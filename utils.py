import os
import json
import re
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

def parse_evaluation_response(text):
    """
    Robust reusable parsing logic to extract evaluation metrics from LLM output.
    Returns a structured dictionary.
    """
    result = {
        "correctness_score": 0,
        "confidence_score": 0,
        "hallucination_risk": "Medium",
        "completeness": "Medium",
        "verdict": "Unreliable",
        "detailed_reason": "Failed to parse evaluation metrics properly."
    }
    
    if not text:
        return result

    # Attempt to find and parse JSON block
    try:
        json_match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "correctness_score" in parsed:
                result["correctness_score"] = int(parsed["correctness_score"])
            elif "Correctness Score" in parsed:
                result["correctness_score"] = int(parsed["Correctness Score"])
                
            if "confidence_score" in parsed:
                val = str(parsed["confidence_score"]).replace('%', '')
                result["confidence_score"] = int(float(val))
            elif "Confidence Score" in parsed:
                val = str(parsed["Confidence Score"]).replace('%', '')
                result["confidence_score"] = int(float(val))
                
            for key in ["hallucination_risk", "Hallucination Risk"]:
                if key in parsed:
                    val = str(parsed[key]).capitalize()
                    if any(x in val for x in ["Low", "Medium", "High"]):
                        result["hallucination_risk"] = [x for x in ["Low", "Medium", "High"] if x in val][0]
                    else:
                        result["hallucination_risk"] = val
                        
            for key in ["completeness", "Completeness"]:
                if key in parsed:
                    val = str(parsed[key]).capitalize()
                    if any(x in val for x in ["Low", "Medium", "High"]):
                        result["completeness"] = [x for x in ["Low", "Medium", "High"] if x in val][0]
                    else:
                        result["completeness"] = val
                        
            for key in ["verdict", "Verdict"]:
                if key in parsed:
                    val = str(parsed[key])
                    for v in ["Reliable", "Partially Reliable", "Unreliable"]:
                        if v.lower() in val.lower():
                            result["verdict"] = v
                            break
                    else:
                        result["verdict"] = val
                        
            for key in ["detailed_reason", "Detailed Reason", "reason", "Reason"]:
                if key in parsed:
                    result["detailed_reason"] = str(parsed[key]).strip()
                    break
            return result
    except Exception as e:
        print(f"[WARN] JSON parsing failed, falling back to regex parsing: {e}")

    # Fallback line/regex parsing if output is plain text
    try:
        corr_match = re.search(r'Correctness Score.*?(\d+)', text, re.IGNORECASE)
        if corr_match:
            result["correctness_score"] = int(corr_match.group(1))
            
        conf_match = re.search(r'Confidence Score.*?(\d+)', text, re.IGNORECASE)
        if conf_match:
            result["confidence_score"] = int(conf_match.group(1))
            
        hall_match = re.search(r'Hallucination Risk.*?(Low|Medium|High)', text, re.IGNORECASE)
        if hall_match:
            result["hallucination_risk"] = hall_match.group(1).capitalize()
            
        comp_match = re.search(r'Completeness.*?(Low|Medium|High)', text, re.IGNORECASE)
        if comp_match:
            result["completeness"] = comp_match.group(1).capitalize()
            
        verd_match = re.search(r'Verdict.*?(Reliable|Partially Reliable|Unreliable)', text, re.IGNORECASE)
        if verd_match:
            val = verd_match.group(1).lower()
            if "partially" in val:
                result["verdict"] = "Partially Reliable"
            elif "unreliable" in val:
                result["verdict"] = "Unreliable"
            else:
                result["verdict"] = "Reliable"
                
        reason_match = re.search(r'Detailed Reason.*?:(.*)', text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            result["detailed_reason"] = reason_match.group(1).strip()
        else:
            reason_match2 = re.search(r'Reason.*?:(.*)', text, re.IGNORECASE | re.DOTALL)
            if reason_match2:
                result["detailed_reason"] = reason_match2.group(1).strip()
            else:
                result["detailed_reason"] = text.strip()
    except Exception as e:
        print(f"[ERROR] Fallback regex parsing error: {e}")
        result["detailed_reason"] = text.strip()

    return result

def evaluate_answer(context, answer):
    """
    Evaluates the generated answer against the retrieved context to check for factual correctness,
    confidence, hallucination risk, completeness, and overall verdict. Uses the same Groq LLM.
    Returns a structured dictionary of evaluation metrics.
    """
    eval_prompt = f"""You are an advanced RAG evaluation system.
Evaluate the generated answer strictly based on the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate:
1. Correctness Score (0-10)
2. Confidence Score (0-100%)
3. Hallucination Risk (Low / Medium / High)
4. Completeness (Low / Medium / High)
5. Verdict (Reliable / Partially Reliable / Unreliable)
6. Detailed Reason

Rules:
- High confidence only if answer is fully supported by context
- Detect hallucinations and unsupported claims
- Be strict and realistic
- Do not always give high scores
- Assign lower confidence for ambiguous or incomplete answers

Provide your output as a valid JSON object with EXACTLY the following keys:
{{
  "correctness_score": <int 0-10>,
  "confidence_score": <int 0-100>,
  "hallucination_risk": "<Low | Medium | High>",
  "completeness": "<Low | Medium | High>",
  "verdict": "<Reliable | Partially Reliable | Unreliable>",
  "detailed_reason": "<string explanation>"
}}
Do not include any markdown formatting or codeblocks outside the JSON object."""

    try:
        llm = create_llm()
        eval_response = llm.invoke(eval_prompt)
        return parse_evaluation_response(eval_response.content)
    except Exception as e:
        print(f"[ERROR] LLM Evaluation error: {e}")
        return {
            "correctness_score": 0,
            "confidence_score": 0,
            "hallucination_risk": "High",
            "completeness": "Low",
            "verdict": "Unreliable",
            "detailed_reason": f"Evaluation engine exception: {str(e)}"
        }
