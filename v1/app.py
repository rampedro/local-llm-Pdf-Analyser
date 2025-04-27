import os
import sys
from agent import load_existing_data, save_training_data, train_gam, init_agent

DOC_DIR = "./docs"

def main():
    if not os.path.isdir(DOC_DIR):
        os.makedirs(DOC_DIR)
        print(f"Created docs folder at {DOC_DIR}. Place your PDF/DOCX files there and restart.")
        sys.exit(0)

    print("=== PDF/DOCX Category Extraction & Demo App ===")
    print("1) Extract categories from a file")
    print("2) Show & train on text-length data (toy GAM demo)")
    print("0) Exit")
    choice = input("Choose: ").strip()

    if choice == "1":
        fname = input("Filename (with .pdf/.docx): ").strip()
        path = os.path.join(DOC_DIR, fname)
        if not os.path.exists(path):
            print("  ❌ file not found.")
            return
        print("  🚀 Initializing LLM agent…")
        qe = init_agent(DOC_DIR)
        print("  📄 Extracting categories…")
        res = qe.query("Extract categories from this document.")
        print("\n=== Extracted ===\n")
        print(res.response)

    elif choice == "2":
        print("  Loading all docs…")
        X, y, fns = load_existing_data(DOC_DIR)
        print(f"  Found {len(fns)} documents.")
        save_training_data(X, y, DOC_DIR)
        print("  Fitting a toy GAM…")
        gam, scaler, metrics = train_gam(X, y)
        print(f"→ R² = {metrics['r2']:.3f}, MSE = {metrics['mse']:.1f}")

    elif choice == "0":
        sys.exit(0)
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
