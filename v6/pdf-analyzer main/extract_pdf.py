import fitz  # PyMuPDF
import os

def extract_text_and_images(pdf_path, output_img_dir="static/images"):
    doc = fitz.open(pdf_path)
    os.makedirs(output_img_dir, exist_ok=True)

    full_text = []
    all_image_paths = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()
        full_text.append(text.strip())

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"page{page_number+1}_img{img_index+1}.{ext}"
            path = os.path.join(output_img_dir, filename)

            with open(path, "wb") as f:
                f.write(image_bytes)

            all_image_paths.append(path)

    return "\n\n".join(full_text), all_image_paths
