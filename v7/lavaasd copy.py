import fitz  # PyMuPDF
import base64
import requests
import json
from io import BytesIO
from PIL import Image
import os
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Function to extract images and figure descriptions from the PDF
def extract_images_and_figures(pdf_path, output_dir='extracted_images'):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize storage for extracted images, captions, and figure mappings
    images = []
    figure_descriptions = {}

    # Loop through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)

        # Extract text from the page
        text = page.get_text("text")
        
        # Look for "Figure X" and associated descriptions, including multiple lines
        figure_matches = re.findall(r'(Figure (\d+))[:\s]*(.*?)(?=\nFigure \d+|$)', text, re.DOTALL | re.IGNORECASE)

        # Extract image objects from the page
        image_list = page.get_images(full=True)
        
        # For each image on the page, extract it and save
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            # Save the image with a unique filename
            image_filename = f'{output_dir}/image_{page_num + 1}_{img_index + 1}.png'
            image.save(image_filename)
            
            # Look for descriptions related to the figure
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

# Function to extract keywords from the description
def extract_keywords(description):
    # Split the description into words and remove stopwords
    words = description.split()
    filtered_words = [word.lower() for word in words if word.lower() not in ENGLISH_STOP_WORDS and len(word) > 2]
    return " ".join(filtered_words)

# Function to convert image to base64
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

# Function to send image and description to the Moondream model
def send_image_to_model(image_base64, description):
    url = "http://localhost:11434/api/generate"  # Model endpoint
    payload = {
        "model": "moondream",  # Specify your model here
        "prompt": description,  # Use extracted description as the prompt
        "images": [image_base64]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)  # Set a timeout (e.g., 30 seconds)
        
        if response.status_code == 200:
            lines = response.text.splitlines()
            full_response_text = ""  
            
            for line in lines:
                try:
                    response_json = json.loads(line)
                    full_response_text += response_json.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")
            
            print("Response description:", full_response_text.strip())
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Main function to process the PDF and send to the model
def process_pdf_and_send_to_model(pdf_path):
    # Extract images and their descriptions from the PDF
    figure_descriptions, images = extract_images_and_figures(pdf_path)
    
    # Loop over the extracted figures and their descriptions
    for figure_number, figure_data in figure_descriptions.items():
        for figure in figure_data:
            image_filename = figure['image']
            description = figure['description']
            
            # Extract keywords from the description
            keywords = extract_keywords(description)
            
            # Construct the dynamic prompt with the fixed phrase
            prompt = f"detail visualization and interaction in this image ? hints: {keywords}"

            print(f"Processing figure {figure_number}: Image {image_filename}")
            
            # Open the extracted image
            image = Image.open(image_filename)
            img_base64 = image_to_base64(image)
            
            # Send the image and its description to the model
            send_image_to_model(img_base64, prompt)

    print("Image processing completed successfully!")

# Example usage
pdf_path = "docs/asd.pdf"  # Replace with the path to your PDF
process_pdf_and_send_to_model(pdf_path)
