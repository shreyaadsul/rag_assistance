import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf(file_path):
    # Open the PDF file in read-binary mode
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Extract text from all pages
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        print(f"Error loading PDF: {e}")
    return text

def split_text(text, chunk_size=500):
    # Handle empty text safely
    if not text:
        return []
    
    # Split text into fixed-size chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_in_faiss(chunks, embeddings):
    # Safely handle empty chunks
    if not chunks:
        return None
        
    # Create vector database from chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store
