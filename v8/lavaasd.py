import fitz  # PyMuPDF
import base64
import requests
import json
from io import BytesIO
from PIL import Image
import os
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_images_and_figures(pdf_path, output_dir='extracted_images'):
    pdf_document = fitz.open(pdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = []
    figure_descriptions = {}

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        
        figure_matches = re.findall(
            r'(Figure (\d+))[:\s]*(.*?)(?=\nFigure \d+|$)', 
            text, re.DOTALL | re.IGNORECASE
        )

        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            image_filename = f'{output_dir}/image_{page_num + 1}_{img_index + 1}.png'
            image.save(image_filename)

            for match in figure_matches:
                figure_label, figure_number, description = match

                if f'Figure {figure_number}' in figure_label:
                    figure_descriptions[figure_number] = figure_descriptions.get(figure_number, [])
                    figure_descriptions[figure_number].append({
                        'image': image_filename,
                        'description': description.strip()
                    })


            images.append(image_filename)

    return figure_descriptions, images

def extract_keywords(description):
    words = re.findall(r'\b\w+\b', description)
    filtered_words = [w.lower() for w in words if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(filtered_words[:15])  # Limit to top 15

def clean_summary(description, max_words=200):
    words = description.strip().split()
    summary = " ".join(words[:max_words])
    return summary

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def send_image_to_model(image_base64, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "moondream",
        "prompt": prompt,
        "images": [image_base64]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            full_response_text = ""
            for line in response.text.splitlines():
                try:
                    response_json = json.loads(line)
                    full_response_text += response_json.get("response", "")
                except json.JSONDecodeError:
                    continue
            print("Model response:", full_response_text.strip())
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def process_pdf_and_send_to_model(pdf_path):
    figure_descriptions, images = extract_images_and_figures(pdf_path)
    
    for figure_number, figure_data in figure_descriptions.items():

        for figure in figure_data:
            image_filename = figure['image']
            description = figure['description']
            
            keywords = extract_keywords(description)
            summary = clean_summary(description)
            #print(figure_number)
            #print(summary)

            prompt = (
                f"Identify and describe the visualization techniques and possible user interaction patterns used in this?\n\n"
                f"Context summary: {summary}\n"
                f"Keywords: {keywords}"
            )

            print(f"\n>>> Processing Figure {figure_number}: {image_filename}")
            image = Image.open(image_filename)
            img_base64 = image_to_base64(image)

            send_image_to_model(img_base64, prompt)

            break

    print("\n✅ Image processing completed!")

# Example usage
pdf_path = "docs/asd.pdf"
process_pdf_and_send_to_model(pdf_path)
