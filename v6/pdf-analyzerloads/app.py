from flask import Flask, request, render_template, jsonify, send_file
import fitz  # PyMuPDF
import requests
import json
import os
import base64
import traceback
import torch
from sentence_transformers import SentenceTransformer
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from prompts import CATEGORY_EXTRACTION_PROMPT

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------- Utilities -----------

def save_uploaded_file(file_storage):
    """Save uploaded file to disk and return path."""
    filename = file_storage.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(path)
    print(f"[✅] File saved: {path}")
    return path

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    print(f"[📄] Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    print(f"[📄] Text extracted: {len(text)} characters")
    return text

def extract_images_from_pdf(pdf_path):
    """Extract all images from a PDF file."""
    print(f"[🖼️] Extracting images from: {pdf_path}")
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        img_list = page.get_images(full=True)
        for img in img_list:
            base_image = doc.extract_image(img[0])
            images.append(base_image["image"])
    print(f"[🖼️] {len(images)} images extracted.")
    return images

def send_image_to_ollama(image_base64):
    """Analyze an image using Ollama."""
    print(f"[🤖] Sending image to Ollama for analysis...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "moondream",
        "prompt": "Analyze this image",
        "images": [image_base64]
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            parts = []
            for line in response.text.splitlines():
                try:
                    parts.append(json.loads(line).get("response", ""))
                except json.JSONDecodeError:
                    print(f"[⚠️] Skipped invalid JSON line: {line}")
            result = ''.join(parts).strip()
            print(f"[🤖] Image analysis done.")
            return result
        else:
            print(f"[❌] Ollama error {response.status_code}: {response.text}")
            return "Error analyzing image"
    except Exception as e:
        print("[❌] Failed sending image to Ollama")
        traceback.print_exc()
        return "Error during image analysis"

def extract_categories_from_text(text):
    """Extract categories from text using Ollama."""
    print(f"[🗂️] Extracting categories...")
    try:
        prompt = CATEGORY_EXTRACTION_PROMPT.format(context_str=text)
        response = ollama.generate(model="granite3.1-dense:2b", prompt=prompt)
        categories = response.get("response", "No categories extracted")
        print(f"[🗂️] Categories extracted.")
        return categories
    except Exception as e:
        print("[❌] Failed extracting categories")
        traceback.print_exc()
        return "Error extracting categories"

# ----------- Routes -----------

@app.route("/", methods=["GET"])
def home():
    """Serve main page."""
    print("[🌐] Home page requested")
    return render_template("index.html")

@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    """Start the analysis: extract images and analyze them first."""
    try:
        print("[📥] Start analysis request received")
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_path = save_uploaded_file(file)
        images = extract_images_from_pdf(file_path)

        # Prepare to process and return the first image immediately
        processed_images = []

        if len(images) > 0:
            img_bytes = images[0]
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            analysis = send_image_to_ollama(img_b64)
            processed_images.append((1, img_b64, analysis))

            # Send first image and its analysis immediately
            return jsonify({"processed_images": processed_images, "loading": True})
        
        # If no images are found, return an error
        return jsonify({"error": "No images found in PDF"}), 400

    except Exception as e:
        print("[❌] Exception during image extraction and analysis")
        traceback.print_exc()
        return jsonify({"error": "Image extraction and analysis failed"}), 500

@app.route("/finish_processing", methods=["POST"])
def finish_processing():
    """Finish the full document processing: extract text and categories."""
    try:
        print("[📥] Finish processing request received")
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_path = save_uploaded_file(file)
        text = extract_text_from_pdf(file_path)
        categories = extract_categories_from_text(text)

        return jsonify({
            "filename": file.filename,
            "text": text,
            "categories": categories
        })
    except Exception as e:
        print("[❌] Exception during text extraction and category processing")
        traceback.print_exc()
        return jsonify({"error": "Text extraction or category extraction failed"}), 500

@app.route("/download/<filename>")
def download_file(filename):
    """Allow file download."""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            print(f"[📥] Downloading file: {filename}")
            return send_file(file_path, as_attachment=True)
        else:
            print(f"[⚠️] Download failed, file not found: {filename}")
            return "File not found", 404
    except Exception as e:
        print("[❌] Exception during file download")
        traceback.print_exc()
        return "Error during file download", 500

# ----------- Main -----------

if __name__ == "__main__":
    app.run(debug=True)
