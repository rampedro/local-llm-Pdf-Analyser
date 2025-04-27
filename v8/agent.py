import os
import logging
import sys
import numpy as np
import json
import chromadb
import ollama  # Import Ollama for embeddings
from llama_index.llms.ollama import Ollama  # Use the old style LLM import
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX
from prompts import CATEGORY_EXTRACTION_PROMPT

logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_data(doc_dir):
    """
    Pull out text lengths and filenames from all docs in dir.
    """
    X, y, filenames = [], [], []
    for fn in os.listdir(doc_dir):
        path = os.path.join(doc_dir, fn)
        text = ""
        if fn.lower().endswith(".txt"):
            with open(path, "r") as f: text = f.read()
        elif fn.lower().endswith(".pdf"):
            try:
                pdf = fitz.open(path)
                for page in pdf:
                    text += page.get_text()
                pdf.close()
            except Exception as e:
                logging.warning(f"Error reading PDF {fn}: {e}")
                continue
        elif fn.lower().endswith(".docx"):
            try:
                doc = docx.Document(path)
                for p in doc.paragraphs:
                    text += p.text
            except Exception as e:
                logging.warning(f"Error reading DOCX {fn}: {e}")
                continue
        else:
            continue

        if text.strip():
            X.append(len(text))
            y.append(len(fn))  # Can be replaced with metadata field
            filenames.append(fn)
    return np.array(X), np.array(y), filenames

def save_training_data(X, y, doc_dir):
    out = os.path.join(doc_dir, "training_data.json")
    with open(out, "w") as f:
        json.dump({"X": X.tolist(), "y": y.tolist()}, f)
    logging.info(f"Saved training data → {out}")

def train_gam(X, y):
    """
    Train a GAM on document text length → filename length (for demo).
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
    Set up Ollama LLM + vector store using llama_index.
    """
    # LLM setup (old style)
    llm = Ollama(model="granite3.1-dense:2b", request_timeout=300.0)
    embed = HuggingFaceEmbedding(model_name=model_name)
    Settings.llm = llm
    Settings.embed_model = embed

    # Load documents
    reader = SimpleDirectoryReader(input_dir=doc_dir, recursive=True)
    docs   = reader.load_data()
    logging.info(f"Loaded {len(docs)} documents into memory")

    # Set up ChromaDB for storing document embeddings
    client = chromadb.EphemeralClient()
    collection = client.create_collection("pdf_extract")

    # Step 1: Embedding Stage - Embed each document and add to Chroma collection
    for i, doc in enumerate(docs):
        response = ollama.embed(
            model="mxbai-embed-large", 
            input=doc.text  # Using doc.text, assuming doc is a document object with a 'text' attribute
        )
        embeddings = response["embeddings"]
        collection.add(
            ids=[str(i)], 
            embeddings=embeddings, 
            documents=[doc.text]
        )
    
    # Step 2: Retrieval Stage - Use query embedding to retrieve relevant documents
    def retrieve_relevant_document(prompt):
        # Generate an embedding for the query input
        response = ollama.embed(model="oscardp96/medcpt-article", input=prompt)
        query_embedding = response["embeddings"]

        # Retrieve the most relevant document
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        data = results['documents'][0][0]  # Assuming you want the first document returned
        return data
    
    # Step 3: Generation Stage - Generate response using the LLM with retrieved data
    def generate_response(prompt, data):
        output = ollama.generate(
            model="llama3.2:1b-text-q2_K",
            prompt=f"context: {data}. Main insturction: {prompt}"
        )
        return output['response']

    # Example retrieval and generation
    example_prompt = "You are an intelligent assistant. Your task is to extract high-level information from Computer Science articles and categorize it into predefined topics."
    retrieved_data = retrieve_relevant_document(example_prompt)
    generated_output = generate_response(example_prompt, retrieved_data)
    logging.info(f"Generated Output: {generated_output}")

    # Set up the vector store and index documents for general query capabilities
    vs = ChromaVectorStore(chroma_collection=collection)
    idx = VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=embed
    )

    tpl = PromptTemplate(CATEGORY_EXTRACTION_PROMPT)
    return idx.as_query_engine(text_qa_template=tpl, similarity_top_k=3)
