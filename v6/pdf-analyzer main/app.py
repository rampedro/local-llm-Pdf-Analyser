from flask import Flask, request, render_template, jsonify, send_file
import fitz  # PyMuPDF
import requests
import json
from io import BytesIO
from PIL import Image
import os
import torch
import sys
from sentence_transformers import SentenceTransformer
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from prompts import CATEGORY_EXTRACTION_PROMPT
import base64
import traceback

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"[INFO] Extracting text from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    print(f"[INFO] Extracted text length: {len(text)} characters")
    return text

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    print(f"[INFO] Extracting images from PDF: {pdf_path}")
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
    print(f"[INFO] Total images extracted: {len(images)}")
    return images

# Function to send image to Ollama for analysis with prints
def send_image_to_ollama(image_base64):
    print("[INFO] Sending image to Ollama for analysis...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "moondream",
        "prompt": "Analyze this image",
        "images": [image_base64]
    }

    try:
        response = requests.post(url, json=payload)
        print(f"[INFO] Ollama Response Status: {response.status_code}")
        print(f"[DEBUG] Ollama Raw Response: {response.text}")

        if response.status_code == 200:
            lines = response.text.splitlines()
            sentence_parts = []

            for idx, line in enumerate(lines):
                try:
                    response_json = json.loads(line)
                    word = response_json.get("response", "")
                    print(f"[DEBUG] Word {idx}: {word}")
                    sentence_parts.append(word)
                except Exception as e:
                    print(f"[ERROR] Failed to decode line {idx}: {line}")
                    traceback.print_exc()

            full_sentence = ''.join(sentence_parts).strip()
            print(f"[RESULT] Full analyzed sentence: {full_sentence}")
            return full_sentence
        else:
            print(f"[ERROR] Ollama returned error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print("[ERROR] Exception occurred while sending image to Ollama")
        traceback.print_exc()
        return "Error during image analysis"

# Function to extract and categorize text
def extract_categories_from_text(text):
    print("[INFO] Extracting categories from text...")
    try:
        prompt = CATEGORY_EXTRACTION_PROMPT.format(context_str=text)
        response = ollama.generate(model="granite3.1-dense:2b", prompt=prompt)
        result = response.get("response", "No categories extracted")
        print(f"[RESULT] Categories Extracted: {result}")
        return result
    except Exception as e:
        print("[ERROR] Failed to extract categories")
        traceback.print_exc()
        return "Error extracting categories"

# Route for uploading PDF/DOCX and processing it
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("[INFO] Upload received.")
            file = request.files["file"]
            file_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(file_path)
            print(f"[INFO] File saved to: {file_path}")

            image_selection = request.form.get("image_selection", "all")
            print(f"[INFO] Image selection option: {image_selection}")

            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            else:
                print("[WARN] Non-PDF file received, no text extraction.")
                text = ""

            images = extract_images_from_pdf(file_path)
            base64_images = [base64.b64encode(img).decode("utf-8") for img in images]

            processed_images = []

            if image_selection == "all":
                selected_indexes = list(range(len(images)))
            elif image_selection == "first":
                selected_indexes = [0]
            elif image_selection == "second" and len(images) > 1:
                selected_indexes = [1]
            elif image_selection == "third" and len(images) > 2:
                selected_indexes = [2]
            elif image_selection.isdigit():
                idx = int(image_selection) - 1
                if 0 <= idx < len(images):
                    selected_indexes = [idx]
                else:
                    selected_indexes = []
            else:
                selected_indexes = []

            print(f"[INFO] Selected image indexes for processing: {selected_indexes}")

            for idx in selected_indexes:
                print(f"[INFO] Processing image {idx+1}")
                image_base64 = base64.b64encode(images[idx]).decode("utf-8")
                analysis_response = send_image_to_ollama(image_base64)
                processed_images.append((idx + 1, analysis_response))

            categories = extract_categories_from_text(text)

            return render_template(
                "index.html",
                text=text,
                images=base64_images,
                processed_images=processed_images,
                categories=categories,
                file=file.filename
            )

        except Exception as e:
            print("[ERROR] Exception occurred during POST request")
            traceback.print_exc()
            return "An error occurred while processing the file."

    print("[INFO] Serving main page (GET request)")
    return render_template("index.html")

# Route for downloading files
@app.route("/download/<filename>")
def download_file(filename):
    print(f"[INFO] Download request for: {filename}")
    return send_file(os.path.join("uploads", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
