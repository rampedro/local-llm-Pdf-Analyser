from flask import Flask, request, render_template, jsonify, send_file
import fitz  # PyMuPDF
import os
import torch
import numpy as np
import logging
import sys
from sentence_transformers import SentenceTransformer
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from prompts import CATEGORY_EXTRACTION_PROMPT
import base64

app = Flask(__name__)

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

# Function to send image to Ollama for analysis
def send_image_to_ollama(image_base64):
    response = ollama.generate(model="granite3.1-dense:2b", prompt="Analyze this image", images=[image_base64])
    return response["response"]

# Function to extract and categorize text
def extract_categories_from_text(text):
    prompt = CATEGORY_EXTRACTION_PROMPT.format(context_str=text)
    response = ollama.generate(model="granite3.1-dense:2b", prompt=prompt)
    return response["response"]

# Route for uploading PDF/DOCX and processing it
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Extract text
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            # Handle DOCX extraction here (similar to PDF)
            pass

        # Process images
        images = extract_images_from_pdf(file_path)
        base64_images = [base64.b64encode(img).decode("utf-8") for img in images]

        # Extract categories using the model
        categories = extract_categories_from_text(text)

        # Pass the 'file' object to the template
        return render_template("index.html", text=text, images=base64_images, categories=categories, file=file)

    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join("uploads", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
