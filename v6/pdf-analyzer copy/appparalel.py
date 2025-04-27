from flask import Flask, request, render_template, send_file
import fitz  # PyMuPDF
import os
import base64
import logging
import sys
import ollama
from concurrent.futures import ThreadPoolExecutor
from prompts import CATEGORY_EXTRACTION_PROMPT

app = Flask(__name__)

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread pool for concurrent image and text extraction
executor = ThreadPoolExecutor(max_workers=4)

# Function to extract text from PDF (optimized with `get_text` in batch)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to extract images from PDF (optimized)
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    image_analysis = []
    image_promises = []  # Store future objects for concurrent processing
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            # Process images concurrently
            image_promises.append(executor.submit(send_image_to_ollama, image_base64))
            images.append(image_base64)

    # Wait for all image analysis to complete
    for future in image_promises:
        image_analysis.append(future.result())

    return images, image_analysis

# Function to send image to Ollama for analysis
def send_image_to_ollama(image_base64):
    response = ollama.generate(model="granite3.1-dense:2b", prompt="Analyze this image", images=[image_base64])
    return response["response"]

# Function to extract and categorize text (cached to avoid reprocessing)
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

        # Extract text and images in parallel
        text_future = executor.submit(extract_text_from_pdf, file_path)
        images_future = executor.submit(extract_images_from_pdf, file_path)

        text = text_future.result()
        images, image_analysis = images_future.result()

        # Extract categories using the model
        categories = extract_categories_from_text(text)

        # Pass images, analysis, and file info to the template
        return render_template("index.html", text=text, images=images, image_analysis=image_analysis, categories=categories, file=file)

    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join("uploads", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
