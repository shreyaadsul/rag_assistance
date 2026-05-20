import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.vector_manager import add_to_vector_store
from backend.metadata_manager import add_document, document_exists

def load_pdf(file_path):
    """
    Reads a PDF file, extracts text, and collects metadata.
    Returns a list of dictionaries with text and metadata per page.
    """
    pages_data = []
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pages_data
        
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                if extracted_text and extracted_text.strip():
                    pages_data.append({
                        "text": extracted_text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": i + 1
                        }
                    })
    except Exception as e:
        print(f"Error loading PDF: {e}")
        
    return pages_data

def split_text(pages_data, chunk_size=500, chunk_overlap=100):
    """
    Splits text from pages into chunks using RecursiveCharacterTextSplitter
    and assigns a unique chunk_id to each chunk's metadata.
    """
    if not pages_data:
        return [], []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    metadatas = []
    chunk_counter = 1
    
    for page_data in pages_data:
        page_chunks = text_splitter.split_text(page_data["text"])
        
        for chunk in page_chunks:
            chunks.append(chunk)
            chunk_meta = page_data["metadata"].copy()
            chunk_meta["chunk_id"] = chunk_counter
            metadatas.append(chunk_meta)
            chunk_counter += 1
            
    return chunks, metadatas

def process_and_store_pdf(file_path):
    """
    Pipeline Helper: Loads the PDF, applies smart chunking, 
    adds to the unified vector store, and updates metadata storage.
    """
    filename = os.path.basename(file_path)
    
    if document_exists(filename):
        print(f"[INFO] Document '{filename}' is already indexed. Skipping.")
        return
        
    print(f"[INFO] Extracting text content from uploaded PDF: {filename}")
    pages_data = load_pdf(file_path)
    if not pages_data:
        raise ValueError("No extractable text found in the uploaded PDF document.")
        
    print("[INFO] Applying smart chunking via RecursiveCharacterTextSplitter...")
    chunks, metadatas = split_text(pages_data, chunk_size=500, chunk_overlap=100)
    if not chunks:
        raise ValueError("Failed to generate document chunks.")
        
    print("[INFO] Adding chunks to unified FAISS database...")
    add_to_vector_store(chunks, metadatas)
    
    # Update Metadata Manager
    add_document(filename, len(chunks), "Indexed")
    print(f"[SUCCESS] '{filename}' indexed successfully ({len(chunks)} chunks).")
