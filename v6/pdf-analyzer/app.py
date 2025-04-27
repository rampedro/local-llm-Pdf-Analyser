from flask import Flask, request, render_template, jsonify, send_file
import fitz  # PyMuPDF
import requests
import json
import os
import base64
import traceback
import uuid
import ollama
from prompts import CATEGORY_EXTRACTION_PROMPT
import logging

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- State Management ----
processing_sessions = {}

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

# ----------- Category Extraction using Ollama Embeddings -----------

def extract_categories_from_text(text):
    """Extract categories from text using Ollama embeddings."""
    print(f"[🗂️] Extracting categories...")
    
    try:
        # Step 1: Generate embeddings for the text using Ollama
        response = ollama.embed(model="mxbai-embed-large", input=text)
        embeddings = response["embeddings"]

        # Log embeddings for debugging purposes
        logging.info(f"Text embedding shape: {len(embeddings)}")

        # Step 2: Prepare the prompt for category extraction
        prompt = CATEGORY_EXTRACTION_PROMPT.format(context_str=text)

        # Step 3: Use Ollama's model to generate a response based on the embedding
        response = ollama.generate(model="granite3.1-dense:2b", prompt=prompt)

        # Extract categories from the response
        categories = response.get("response", "No categories extracted")
        print(f"[🗂️] Categories extracted.")
        return categories

    except Exception as e:
        # Handle errors gracefully and log them
        print("[❌] Failed extracting categories")
        logging.error(f"Error during category extraction: {str(e)}")
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
    """Start the analysis: prepare the session."""
    try:
        print("[📥] Start analysis request received")
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        session_id = str(uuid.uuid4())
        file_path = save_uploaded_file(file)
        images = extract_images_from_pdf(file_path)

        if not images:
            return jsonify({"error": "No images found in PDF"}), 400

        processing_sessions[session_id] = {
            "file_path": file_path,
            "images": images,
            "current_index": 0,
            "text_done": False,
            "categories_done": False,
            "text": "",
            "categories": ""
        }

        return jsonify({"session_id": session_id})

    except Exception as e:
        print("[❌] Exception during start_analysis")
        traceback.print_exc()
        return jsonify({"error": "Start analysis failed"}), 500

@app.route("/process_next", methods=["POST"])
def process_next():
    """Process the next step in the session."""
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        session = processing_sessions.get(session_id)
        if not session:
            return jsonify({"error": "Invalid session"}), 400

        if session["current_index"] < len(session["images"]):
            img_bytes = session["images"][session["current_index"]]
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            analysis = send_image_to_ollama(img_b64)
            session["current_index"] += 1
            return jsonify({"step": "image", "image_b64": img_b64, "analysis": analysis, "done": False})
        
        elif not session["text_done"]:
            session["text"] = extract_text_from_pdf(session["file_path"])
            session["text_done"] = True
            return jsonify({"step": "text", "text": session["text"], "done": False})

        elif not session["categories_done"]:
            session["categories"] = extract_categories_from_text(session["text"])
            session["categories_done"] = True
            return jsonify({"step": "categories", "categories": session["categories"], "done": True})

        else:
            return jsonify({"done": True})

    except Exception as e:
        print("[❌] Exception during process_next")
        traceback.print_exc()
        return jsonify({"error": "Processing step failed"}), 500

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
