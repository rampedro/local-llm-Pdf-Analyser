import os
import sys
from agent import load_existing_data, save_training_data, train_gam, init_agent

DOC_DIR = "./docs"
import fitz  # PyMuPDF

def extract_categories_from_file(file_path):
    print(f"Extracting categories from file: {file_path}")
    
    # Check if the file is a PDF
    if file_path.lower().endswith(".pdf"):
        try:
            # Open the PDF file
            doc = fitz.open(file_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # Load page by index
                text += page.get_text()  # Extract text from the page
            
            print(f"Extracted text from PDF: {text[:100]}...")  # Print first 100 characters for debug
            # Process the extracted text (e.g., category extraction, NLP, etc.)
            
        except Exception as e:
            print(f"Error while extracting text from PDF: {e}")
            return
    
    # Add logic for other file types if needed (e.g., DOCX, TXT)
    elif file_path.lower().endswith(".docx"):
        # Handle DOCX file here (use python-docx or similar library)
        pass
    else:
        print(f"Unsupported file type: {file_path}")


def main():
    while True:
        print("=== PDF/DOCX Category Extraction & Research Helper ===")
        print("1) Extract categories from a file")
        print("2) Analyze text stats (toy GAM demo)")
        print("0) Exit")
        choice = input("Choose: ")

        if choice == "1":
            file_path = input("Enter the path of the file to extract categories from: ")
            print("Extracting categories from the file...")
            extract_categories_from_file(file_path)
        
        elif choice == "2":
            print("Analyzing text stats (toy GAM demo)...")
            # Call your text stats demo function here.
        
        elif choice == "0":
            print("Exiting...")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()

