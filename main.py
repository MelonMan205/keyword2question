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

verbose = False
#res time with verbose = 7672 ms, wo = 6858 ms

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a lightweight model.
        all-MiniLM-L6-v2 is very small (~80MB) and fast while maintaining good accuracy
        """
        self.model = SentenceTransformer(model_name)
        
    def compare_text(self, text: str, keyword: str, threshold: float = 0.2) -> bool:
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


def returnQuestions(subject, keyword, start, end): 
    try:
        semantic_searcher = SemanticSearch()

        questions = []
        qs = 0
        vqs = 0
        subject = str(subject) if str(subject) else subject
        
        pdf_document = fitz.open()
        urls, years, subject = getSearchURL(subject, [int(start),int(end)])
        page_year_mapping = []
        print(f"URLS: {urls}") if verbose else None
        
        if urls == []:
            return None, 200
        

        for url, year in zip(urls, years):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
            
            
            
                pdf_data = BytesIO(response.content) #load pdf data into byte stream
                
                open = fitz.open(stream=pdf_data, filetype="pdf")

                start_page = pdf_document.page_count
                
                pdf_document.insert_pdf(open)
                
                end_page = pdf_document.page_count - 1
                
                page_year_mapping.append((start_page, end_page, year))
                open.close()
            except requests.exceptions.RequestException as e:
                print(f"Error accessing PDF URL: {e}")
                continue
        if not pdf_document.page_count:
            return [], 200

        #keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        question_number = None
        for page_num in range(1, pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            question_data = None #implement after searchURL logic
            if not text.strip():
                continue
            
            year = next((y for s, e, y in page_year_mapping if s <= page_num <= e), "Unknown Year")
            split_pairs = re.findall(r'\nQuestion\s+(\d+)\s*(.*?)(?=\nQuestion\s+\d+|$)', text, re.DOTALL)
            print(f"Page split_pairs: {split_pairs}") if verbose else None

            # Process pairs to get valid questions while preserving numbers
            valid_pairs = [(num, content.strip()) 
                          for num, content in split_pairs 
                          if semantic_searcher.compare_text(content.strip(), keyword)]


            valid_questions = [content for _, content in valid_pairs]
            question_numbers = [num for num, _ in valid_pairs]

            print(f"Page split_questions (valid_question): {valid_questions}") if verbose else None
            qs += len(split_pairs)
            vqs += len(valid_questions)

            for q_num, question in zip(question_numbers, valid_questions):
                question_number = q_num  # Already have clean number
                print(f"Question Number (DETECTED): {question_number}") if verbose else None
                
                images = []
                
                pix = page.get_pixmap(dpi=300)  # Render full page
                image_bytes = pix.tobytes("png")

                #FALLBACK: FULL PAGE IMAGE
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                images.append(f"data:image/png;base64,{image_base64}")

                questions.append({
                    "question_number": question_number,
                    "content": question.strip(),
                    "question_data": f': {year} {subject} Exam', #implement actual date after
                    "images": images,
                    "formatted": f"<p>{question.strip()}</p>" #question??  
                })
                
                questions = formatQuestions(questions)
                
        pdf_document.close()
        if os.path.exists("/temp.pdf"):
            os.remove("/temp.pdf")

        return questions, metrics(qs, vqs)

    except requests.exceptions.RequestException as e:
        print(f"Error accessing PDF URL: {e}")
        return None

    except Exception as e:
        print(f"Error other: {e}")
        return None
    
    return []

def metrics(qs, vqs):
    return {
        'valid_question_perc': (round(((vqs/qs)*100 if qs != 0 else None))),
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
        
    
def detectQuestionArea(image_bytes, keyword="Question"):
    """
    Detect the bounding box of the specified keyword in an image using Tesseract OCR.
    """
    img = Image.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        if keyword.lower() in data['text'][i].lower():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            return (x, y, x + w, y + h)
    return None

def compute_similarity(a, b):
    """
    Compute the similarity ratio between two strings using SequenceMatcher.
    Args:
        a (str): First string.
        b (str): Second string.
    Returns:
        float: Similarity ratio (0 to 1).
    """
    return SequenceMatcher(None, a, b).ratio()

def crop_question_area(image_bytes, bounding_box):
    """
    Crop the image to the specified bounding box.
    """
    img = Image.open(io.BytesIO(image_bytes))
    cropped_img = img.crop(bounding_box)
    cropped_io = io.BytesIO()
    cropped_img.save(cropped_io, format="PNG")
    return cropped_io.getvalue()


def crop_to_question_content(image_bytes, question_content, similarity_threshold=0.01):
    """
    Crop the image dynamically to include the entire question content based on similarity to `question_content`.

    Args:
        image_bytes: Bytes of the original image.
        question_content: The textual content of the question to match.
        similarity_threshold: Minimum similarity ratio for matching text blocks (0 to 1).

    Returns:
        Cropped image bytes or the full image if no content matches.
    """
    img = Image.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(data["text"])


    matched_boxes = []

    for i in range(n_boxes):
        detected_text = data["text"].join(' ')
        confidence = int(data["conf"])
        if confidence >= 0 and detected_text:
            similarity = compute_similarity(detected_text, question_content)
            print(f"Detected: '{detected_text}', Similarity: {similarity}, Confidence: {confidence}")

    for i in range(n_boxes):
        detected_text = data["text"].join(' ')
        confidence = int(data["conf"])

        # Skip low-confidence detections or empty text
        if confidence < 70 or not detected_text:
            continue

        # Compute similarity between detected text and question content
        similarity = compute_similarity(detected_text, question_content)

        if similarity >= similarity_threshold:
            # Add bounding box for matched content
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            matched_boxes.append((x, y, x + w, y + h))

    if matched_boxes:
        # Combine matched bounding boxes into a single crop area
        min_x = min(box[0] for box in matched_boxes)
        min_y = min(box[1] for box in matched_boxes)
        max_x = max(box[2] for box in matched_boxes)
        max_y = max(box[3] for box in matched_boxes)

        # Expand the crop area dynamically (add padding)
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img.width, max_x + padding)
        max_y = min(img.height, max_y + padding)

        expanded_box = (min_x, min_y, max_x, max_y)
        return crop_question_area(image_bytes, expanded_box)

    else:
        print(f"No sufficiently similar text blocks found. Returning full image.")
        return image_bytes



def getquestion(subject: str, keywords: str):
    
    return None

