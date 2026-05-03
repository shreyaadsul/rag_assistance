import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

def load_pdf(file_path):
    """
    Reads a PDF file, extracts text, and collects metadata.
    Returns a list of dictionaries with text and metadata per page.
    """
    pages_data = []
    
    # Handle the case where file might not exist
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pages_data
        
    filename = os.path.basename(file_path)
    
    try:
        # Open the PDF file in read-binary mode
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages and store with metadata
            for i, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                
                # Check for empty or None text
                if extracted_text and extracted_text.strip():
                    pages_data.append({
                        "text": extracted_text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": i + 1  # 1-indexed page number
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
    # Handle empty input safely
    if not pages_data:
        return [], []
    
    # Initialize the smarter text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    metadatas = []
    chunk_counter = 1
    
    for page_data in pages_data:
        # Split the text of the current page
        page_chunks = text_splitter.split_text(page_data["text"])
        
        for chunk in page_chunks:
            chunks.append(chunk)
            
            # Copy base metadata (source, page) and add chunk_id
            chunk_meta = page_data["metadata"].copy()
            chunk_meta["chunk_id"] = chunk_counter
            metadatas.append(chunk_meta)
            
            chunk_counter += 1
            
    return chunks, metadatas

def create_embeddings():
    """
    Creates embeddings using HuggingFaceEmbeddings (updated import).
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_in_faiss(chunks, metadatas, embeddings):
    """
    Creates a FAISS vector store from text chunks and their metadata.
    """
    # Safely handle empty chunks
    if not chunks:
        return None
        
    # Create vector database from chunks, passing metadata explicitly
    vector_store = FAISS.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        metadatas=metadatas
    )
    return vector_store

def create_llm():
    """
    Initializes the Language Model for generating answers using Groq.
    Uses ChatGroq with llama-3.1-8b-instant (updated from decommissioned llama3-8b-8192) 
    and a temperature of 0 for deterministic outputs.
    """
    # Check for the GROQ_API_KEY to provide a clear error message if missing
    if not os.getenv("GROQ_API_KEY"):
        print("\n[ERROR] GROQ_API_KEY is missing from your .env file!")
        print("Please create a free API key at https://console.groq.com/keys and add it to .env")
        exit(1)
        
    # Model changed to llama-3.1-8b-instant due to llama3-8b-8192 deprecation
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def evaluate_answer(context, answer):
    """
    Evaluates the generated answer against the retrieved context to check for factual correctness,
    completeness, and hallucination. Uses the same Groq LLM.
    """
    eval_prompt = f"""You are an AI evaluator.

Given:
Context:
{context}

Answer:
{answer}

Evaluate the answer based ONLY on the context.

Check:
1. Is the answer factually correct?
2. Is it fully supported by the context?
3. Is any important information missing?

Give output in this format:
Score: <number from 0 to 10>
Verdict: <Correct / Partially Correct / Incorrect>
Reason: <short explanation>"""

    llm = create_llm()
    eval_response = llm.invoke(eval_prompt)
    return eval_response.content
