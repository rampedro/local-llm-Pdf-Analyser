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

# ------------- Helper Functions -------------

def extract_text_from_pdf(pdf_path):
    print(f"[INFO] Extracting text from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    print(f"[INFO] Extracted text length: {len(text)} characters")
    return text

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
            return "Error in model response"
    except Exception as e:
        print("[ERROR] Exception occurred while sending image to Ollama")
        traceback.print_exc()
        return "Error during image analysis"

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

# ------------- Routes -------------

@app.route("/", methods=["GET"])
def home():
    print("[INFO] Serving main page (GET)")
    return render_template("index.html")

# STEP 1: Extract only TEXT and show first card
@app.route("/extract_text", methods=["POST"])
def extract_text():
    try:
        print("[INFO] Extract text POST received")
        file = request.files["file"]
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        print(f"[INFO] File saved to: {file_path}")

        text = extract_text_from_pdf(file_path)
        categories = extract_categories_from_text(text)

        return jsonify({
            "filename": file.filename,
            "text": text,
            "categories": categories
        })
    except Exception as e:
        print("[ERROR] Failed extract_text")
        traceback.print_exc()
        return jsonify({"error": "Failed text extraction"}), 500

# STEP 2: After button click, extract IMAGES and analyze
@app.route("/extract_images", methods=["POST"])
def extract_images():
    try:
        print("[INFO] Extract Images POST received")
        file = request.files["file"]
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        print(f"[INFO] File saved to: {file_path}")

        images = extract_images_from_pdf(file_path)
        base64_images = [base64.b64encode(img).decode("utf-8") for img in images]

        processed_images = []
        for idx, img_data in enumerate(base64_images):
            analysis = send_image_to_ollama(img_data)
            processed_images.append((idx + 1, img_data, analysis))

        return jsonify({"processed_images": processed_images})
    except Exception as e:
        print("[ERROR] Exception in extract_images")
        traceback.print_exc()
        return jsonify({"error": "Failed to extract images"}), 500


@app.route("/download/<filename>")
def download_file(filename):
    print(f"[INFO] Download request for file: {filename}")
    return send_file(os.path.join("uploads", filename), as_attachment=True)

# ------------- Main -------------
if __name__ == "__main__":
    app.run(debug=True)
