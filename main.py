'''
Filename: main.py

Contains all secondary functions. NOT flasks.py (app.py).
Modifies Helper Functions.

'''
import requests
import io
from io import BytesIO
from PyPDF2 import PdfReader
#from fitz import get_pixmap
#from pdf2image import convert_from_path
from PIL import Image
import pytesseract
#import spacy
#from spacy.matcher import PhraseMatcher
import os
#nlp = spacy.load('en_core_web_sm')
import fitz
import base64
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


verbose = False
#res time with verbose = 7672 ms, wo = 6858 ms

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a lightweight model.
        all-MiniLM-L6-v2 is very small (~80MB) and fast while maintaining good accuracy
        """
        self.model = SentenceTransformer(model_name)
        
    def compare_text(self, text: str, keyword: str, threshold: float) -> bool:
        """
        Compare text with keyword using semantic similarity
        Returns True if semantically similar above threshold
        """
        # Encode both texts to get embeddings
        embeddings = self.model.encode([text, keyword], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), 
                                                         embeddings[1].unsqueeze(0))
        
        return bool(similarity > threshold)
    
    def find_best_subject_match(self, query_subject: str, available_subjects: list, threshold: float = 0.3):
        """Find the closest matching subject from available subjects in data dictionary"""
        query_embedding = self.model.encode(query_subject, convert_to_tensor=True)
        subject_embeddings = self.model.encode(available_subjects, convert_to_tensor=True)
        
        # Calculate similarities with all available subjects
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            subject_embeddings
        )
        
        best_match_idx = torch.argmax(similarities).item()
        best_match_score = similarities[best_match_idx].item()
        
        if best_match_score > threshold:
            print(f"Best match for '{query_subject}': {available_subjects[best_match_idx]}") if verbose else None
            subject = available_subjects[best_match_idx]
            return subject
        return None
semantic_searcher = SemanticSearch()

def normalise(value, a, b):
    return a + ((value - 0)*(b-a)) / (100 - 0)

def formatQuestions(questions):
    res = []
    for question in questions:
        content = question['content']

        formatted_content = content.replace("\n","<br>").replace("\t","&emsp;")

        html_content = f"""
        <div class="question">
            <h3>Question {question['question_number']}{question['question_data']}</h3>
            <p>{formatted_content}</p>
        </div>
        """


        html_content = f"""
        <div class="question">
            <h3>Question {question['question_number']}{question['question_data']}</h3>
        </div>
        """
        #add {date and question number VCAA Exam x}

        question["formatted"] = html_content
        res.append(question)
    return res


def fetchPDF(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        pdf_data = BytesIO(response.content)
        return pdf_data
    except requests.exceptions.RequestException as e:
        print(f"Error accessing PDF URL: {e}")
        return None

def process_page(page, page_num, page_year_mapping, scaled_threshold, keyword, subject):
    """Process single PDF page and return questions found"""
    text = page.get_text()
    questions = []
    
    if not text.strip():
        return [], 0, 0

    year = next((y for s, e, y in page_year_mapping if s <= page_num <= e), "Unknown Year")
    split_pairs = re.findall(r'\nQuestion\s+(\d+)\s*(.*?)(?=\nQuestion\s+\d+|$)', text, re.DOTALL)
    print(f"Page split_pairs: {split_pairs}") if verbose else None
    valid_pairs = [(num, content.strip()) 
                   for num, content in split_pairs 
                   if semantic_searcher.compare_text(content.strip(), keyword, scaled_threshold)]
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

def returnQuestions(subject, keyword, start, end, threshold):
    try:
        total_qs = 0
        total_valid_qs = 0
        all_questions = []
        
        subject = str(subject) if str(subject) else subject
        scaled_threshold = normalise(threshold, 0.15, 0.35)

        pdf_document = fitz.open()
        urls, years, subject = getSearchURL(subject, [int(start),int(end)])
        page_year_mapping = []
        print(f"URLS: {urls}") if verbose else None
        if not urls:
            return None, 200
        
        # Fetch PDFs concurrently
        with ThreadPoolExecutor() as executor:
            pdfs = list(executor.map(fetchPDF, urls))

        # Process PDFs
        for pdf_data, year in zip(pdfs, years):
            if pdf_data:
                open_pdf = fitz.open(stream=pdf_data, filetype="pdf")
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
                    page_year_mapping,
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
        if os.path.exists("/temp.pdf"):
            os.remove("/temp.pdf")

        formatted_questions = formatQuestions(all_questions)
        metrics_data = metrics(total_qs, total_valid_qs)
        
        return formatted_questions if formatted_questions else [], metrics_data

    except requests.exceptions.RequestException as e:
        print(f"Error accessing PDF URL: {e}")
        return [], 200
    except Exception as e:
        print(f"Error other: {e}")
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
        
