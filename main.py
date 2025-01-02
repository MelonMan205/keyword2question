'''
Filename: main.py
Contains all secondary functions with optimizations for performance.
'''
import requests
from io import BytesIO
import fitz
import base64
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from functools import lru_cache

verbose = False

class SemanticSearch:
    _instance = None
    
    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.batch_size = batch_size
        return cls._instance
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        if self.model is None:
            self.model = SentenceTransformer(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.batch_size = batch_size

    @lru_cache(maxsize=128)
    def compare_text_batch(self, texts: tuple, keyword: str, threshold: float) -> List[bool]:
        """
        Compare multiple texts with keyword using semantic similarity.
        Note: texts must be a tuple for caching to work
        """
        with torch.no_grad():
            batch_size = self.batch_size
            texts_list = list(texts)  # Convert back to list for processing
            batches = [texts_list[i:i+batch_size] for i in range(0, len(texts_list), batch_size)]
            results = []

            for batch in batches:
                pairs = [[text, keyword] for text in batch]
                flat_texts = [item for pair in pairs for item in pair]
                embeddings = self.model.encode(flat_texts, convert_to_tensor=True)
                text_embeddings = embeddings[::2]
                keyword_embeddings = embeddings[1::2]
                similarities = torch.nn.functional.cosine_similarity(text_embeddings, keyword_embeddings)
                batch_results = similarities > threshold
                results.extend(batch_results)

            return results

    @lru_cache(maxsize=1024)
    def compare_text(self, text: str, keyword: str, threshold: float) -> bool:
        """
        Compare text with keyword using semantic similarity
        Returns True if semantically similar above threshold
        """
        with torch.no_grad():
            embeddings = self.model.encode([text, keyword], convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0), 
                embeddings[1].unsqueeze(0)
            )
            return bool(similarity > threshold)
    
    @lru_cache(maxsize=128)
    def find_best_subject_match(self, query_subject: str, available_subjects: tuple, threshold: float = 0.3):
        """Find the closest matching subject from available subjects"""
        with torch.no_grad():
            query_embedding = self.model.encode(query_subject, convert_to_tensor=True)
            subject_embeddings = self.model.encode(available_subjects, convert_to_tensor=True)
            
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                subject_embeddings
            )
            
            best_match_idx = torch.argmax(similarities).item()
            best_match_score = similarities[best_match_idx].item()
            
            if best_match_score > threshold:
                print(f"Best match for '{query_subject}': {available_subjects[best_match_idx]}") if verbose else None
                return available_subjects[best_match_idx]
            return None

semantic_searcher = SemanticSearch()

def normalise(value, a, b):
    return a + ((value - 0)*(b-a)) / (100 - 0)

def formatQuestions(questions):
    res = []
    for question in questions:
        content = question['content']
        # formatted_content = content.replace("\n","<br>").replace("\t","&emsp;")
        
        # html_content = f"""
        # <div class="question">
        #     <h3>Question {question['question_number']}{question['question_data']}</h3>
        #     <p>{formatted_content}</p>
        # </div>
        # """
        html_content = f"""
        <div class="question">
            <h3>Question {question['question_number']}{question['question_data']}</h3>
        </div>
        """
        
        question["formatted"] = html_content
        res.append(question)
    return res

async def fetchPDF(url: str, session: aiohttp.ClientSession) -> bytes:
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                if content.startswith(b'%PDF'):
                    return content
                print(f"Invalid PDF content from {url}") if verbose else None
            print(f"Error accessing PDF URL: {response.status}") if verbose else None
        return None
    except Exception as e:
        print(f"Error accessing PDF URL: {e}")
        return None

async def fetchallPDFs(urls: List[str], session: aiohttp.ClientSession):
    tasks = [fetchPDF(url, session) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_pdfs = [pdf for pdf in results if isinstance(pdf, bytes) and pdf is not None]
    return valid_pdfs

def process_page(page, page_num, page_year_mapping, scaled_threshold, keyword, subject):
    """Process single PDF page and return questions found"""
    text = page.get_text()
    questions = []
    
    if not text.strip():
        return [], 0, 0

    year = next((y for s, e, y in page_year_mapping if s <= page_num <= e), "Unknown Year")
    split_pairs = re.findall(r'\nQuestion\s+(\d+)\s*(.*?)(?=\nQuestion\s+\d+|$)', text, re.DOTALL)
    print(f"Page split_pairs: {split_pairs}") if verbose else None

    contents = [content.strip() for _, content in split_pairs]
    # Convert list to tuple for caching
    valid_indices = semantic_searcher.compare_text_batch(tuple(contents), keyword, scaled_threshold)

    valid_pairs = [(num, content) 
                   for (num, content), is_valid in zip(split_pairs, valid_indices) 
                   if is_valid]
    
    print(f"Page split_questions (valid_question): {valid_pairs}") if verbose else None
    
    for q_num, question in valid_pairs:
        pix = page.get_pixmap(dpi=300)
        image_bytes = pix.tobytes("png")
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        print(f"Question Number (DETECTED): {q_num}") if verbose else None
        questions.append({
            "question_number": q_num,
            "content": question,
            "question_data": f': {year} {subject} Exam',
            "images": [f"data:image/png;base64,{image_base64}"],
            "formatted": f"<p>{question}</p>"
        })
    
    return questions, len(split_pairs), len(valid_pairs)

async def returnQuestions(subject, keyword, start, end, threshold, session: aiohttp.ClientSession):
    try:
        total_qs = 0
        total_valid_qs = 0
        all_questions = []
        
        subject = str(subject) if subject else ""
        scaled_threshold = normalise(threshold, 0.15, 0.35)

        pdf_document = fitz.open()
        # Convert year range to tuple for hashability
        year_range = (int(start), int(end))
        urls, years, subject = getSearchURL(subject, tuple(year_range))
        page_year_mapping = []
        print(f"URLS: {urls}") if verbose else None
        
        if not urls:
            return None, 200
        
        # Fetch PDFs concurrently
        pdfs = await fetchallPDFs(urls, session)

        if not pdfs:
            return [], 200

        # Process PDFs
        for pdf_data, year in zip(pdfs, years):
            if pdf_data:
                stream = BytesIO(pdf_data)
                open_pdf = fitz.open(stream=stream, filetype="pdf")
                start_page = pdf_document.page_count
                pdf_document.insert_pdf(open_pdf)
                end_page = pdf_document.page_count - 1
                page_year_mapping.append((start_page, end_page, year))
                open_pdf.close()

        if not pdf_document.page_count:
            return [], 200

        # Process pages concurrently
        with ThreadPoolExecutor() as executor:
            futures = []
            for page_num in range(1, pdf_document.page_count):
                future = executor.submit(
                    process_page,
                    pdf_document[page_num],
                    page_num, 
                    tuple(page_year_mapping),  # Convert to tuple for hashability
                    scaled_threshold,
                    keyword,
                    subject
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                page_questions, page_total, page_valid = future.result()
                all_questions.extend(page_questions)
                total_qs += page_total
                total_valid_qs += page_valid

        pdf_document.close()
        formatted_questions = formatQuestions(all_questions)
        metrics_data = metrics(total_qs, total_valid_qs)
        
        return formatted_questions if formatted_questions else [], metrics_data

    except Exception as e:
        print(f"Error in returnQuestions: {e}")
        import traceback
        traceback.print_exc()
        return [], 200

def metrics(qs, vqs):
    return {
        'valid_question_perc': (round(((vqs/qs)*100 if qs != 0 else 0))),
        'total_questions': qs
    }

from databaseapi import getURL, generateRange
def getSearchURL(subject: str, year_range: list):
    try:
        urls, year_range, subject = getURL(subject, generateRange(year_range))
        return urls, year_range, subject
        
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL: {e}") if verbose else None
        return None
