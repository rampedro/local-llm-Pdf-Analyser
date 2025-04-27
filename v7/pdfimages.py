import fitz  # PyMuPDF
import re
from PIL import Image
from io import BytesIO
import os

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

# Example usage:
pdf_path = 'docs/asd.pdf'
figure_descriptions, images = extract_images_and_figures(pdf_path)

# Print the extracted figure descriptions
print("Extracted Figure Descriptions:")
for figure, img_files in figure_descriptions.items():
    print(f"Figure {figure}:")
    for img in img_files:
        print(f"  Image: {img['image']}")
        print(f"  Description: {img['description']}")

print(f"Total Images Extracted: {len(images)}")
