import os
import logging
import sys
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import fitz  # For handling PDFs
import docx  # For handling DOCX files

from prompts import CATEGORY_EXTRACTION_PROMPT

logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_data(doc_dir):
    """
    Walk docs/, pull out text lengths (X) and filename lengths (y).
    Returns X (np.array), y (np.array), and filenames list.
    """
    X, y, filenames = [], [], []
    for fn in os.listdir(doc_dir):
        path = os.path.join(doc_dir, fn)
        text = ""
        if fn.lower().endswith(".txt"):
            with open(path, "r") as f: text = f.read()
        elif fn.lower().endswith(".pdf"):
            pdf = fitz.open(path)
            for page in pdf: text += page.get_text()
            pdf.close()
        elif fn.lower().endswith(".docx"):
            doc = docx.Document(path)
            for p in doc.paragraphs: text += p.text
        else:
            continue

        if text.strip():
            X.append(len(text))
            y.append(len(fn))
            filenames.append(fn)
    return np.array(X), np.array(y), filenames

def save_training_data(X, y, doc_dir):
    out = os.path.join(doc_dir, "training_data.json")
    with open(out, "w") as f:
        json.dump({"X": X.tolist(), "y": y.tolist()}, f)
    logging.info(f"Saved training data → {out}")

def train_gam(X, y):
    """
    Example: fit a 1D smoothing GAM on X→y, return model and metrics.
    """
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    gam = LinearGAM(s(0)).fit(X_train_s, y_train)
    preds = gam.predict(X_test_s)
    r2   = r2_score(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    return gam, scaler, {"r2": r2, "mse": mse}

def init_agent(doc_dir, model_name="BAAI/bge-small-en-v1.5"):
    """
    Build a llama_index VectorStoreIndex over ALL docs in doc_dir,
    return a query_engine that uses CATEGORY_EXTRACTION_PROMPT.
    """
    llm = Ollama(model="granite3-moe", request_timeout=300.0)
    embed = HuggingFaceEmbedding(model_name=model_name)
    Settings.llm = llm
    Settings.embed_model = embed

    reader = SimpleDirectoryReader(input_dir=doc_dir, recursive=True)
    docs   = reader.load_data()
    logging.info(f"Loaded {len(docs)} documents into memory")

    client     = chromadb.EphemeralClient()
    collection = client.create_collection("pdf_extract")
    vs         = ChromaVectorStore(chroma_collection=collection)

    idx = VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=embed
    )

    tpl = PromptTemplate(CATEGORY_EXTRACTION_PROMPT)
    return idx.as_query_engine(text_qa_template=tpl, similarity_top_k=3)
