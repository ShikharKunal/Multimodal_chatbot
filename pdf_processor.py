from typing import Any, List, Dict, Union
from pydantic import BaseModel
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import os

# Path to the PDF file
image_output_dir_path = './extracted_images'

# Ensure output directory exists
if not os.path.exists(image_output_dir_path):
    os.makedirs(image_output_dir_path)

# Function to extract text using pdfplumber
def extract_text(pdf) -> str:
    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text

# Function to extract images using PyMuPDF
def extract_images(pdf_document, output_dir):
    images = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save image
            image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            images.append(image_path)
    


# Function to split text into chunks with overlap
def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def preprocess(pdf_document):
    # Extract elements from PDF
    texts = extract_text(pdf_document)
    images = extract_images(pdf_document, image_output_dir_path)

    category_counts = {
        "text": 0,
        "image": 0
    }

    # Split text into chunks with overlap
    chunk_size = 512
    overlap = 200
    text_chunks = split_text_into_chunks(texts, chunk_size, overlap)

    # Count text chunks
    category_counts['text'] = len(text_chunks)

    # Count images
    category_counts['image'] = len(images)
    print("extraction done")

    return text_chunks, images, category_counts