import os
from dotenv import load_dotenv
from utils import load_pdf, split_text, create_embeddings, store_in_faiss

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load PDF from: data/insurance    XII.pdf
    pdf_path = "data/insurance XII.pdf"
    text = load_pdf(pdf_path)
    
    # Split text into chunks
    chunks = split_text(text)
    
    if not chunks:
        print("No text extracted from PDF")
        return
    
    # Create embeddings
    embeddings = create_embeddings()
    
    # Store chunks in FAISS
    vector_store = store_in_faiss(chunks, embeddings)
    
    # Print output messages safely if vector database was created
    if vector_store is not None:
        print(f"Chunks created: {len(chunks)}")
        print("FAISS vector database created successfully!")

if __name__ == "__main__":
    main()
