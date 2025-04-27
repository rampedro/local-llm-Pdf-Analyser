import fitz  # PyMuPDF
import base64
import requests
import json  # Correcting the issue with json decoding
from io import BytesIO
from PIL import Image

# Function to extract images from the PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
    
    return images

# Function to convert image to base64 string
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

# Function to send image to an object recognition model (adjust URL and payload)
def send_image_to_model(image_base64):
    url = "http://localhost:11434/api/generate"  # Example URL; replace with correct model endpoint
    payload = {
        "model": "moondream",  # Specify your model here, like BLIP or any other visual-text model
        "prompt": "only mention how each visualization shows diffrerent information",
        "images": [image_base64]
    }

    response = requests.post(url, json=payload)

    # Print the raw response content to help debug
    print("Raw response content:", response.text)

    if response.status_code == 200:
        # Split the response text into individual JSON objects
        lines = response.text.splitlines()
        for line in lines:
            try:
                response_json = json.loads(line)  # Decode each line as a separate JSON
                print("Response description:", response_json["response"])  # Extract the description from each line
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
    else:
        print(f"Error: {response.status_code}")

# Main function to handle the process
def process_pdf_and_send_to_model(pdf_path):
    images = extract_images_from_pdf(pdf_path)
    
    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1}")
        img_base64 = image_to_base64(image)
        send_image_to_model(img_base64)

    # Add a final sentence after all images are processed
    print("Image processing completed successfully!")

# Example usage
pdf_path = "docs/asd.pdf"  # Replace with the path to your PDF
process_pdf_and_send_to_model(pdf_path)
