import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import the existing real RAG backend functions preserved in utils.py
from utils import load_pdf, split_text, create_embeddings, store_in_faiss, create_llm, evaluate_answer

# Load environment variables (e.g., GROQ_API_KEY) from .env file
load_dotenv()

# Initialize the Flask web application
app = Flask(__name__)

# Configure upload folder and permitted file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists before starting the app
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global in-memory caches to prevent redundant loading overhead
GLOBAL_VECTOR_STORES = {}
GLOBAL_EMBEDDINGS = None

def get_or_create_embeddings():
    """
    Returns a cached instance of HuggingFaceEmbeddings to ensure 
    the model isn't redundantly reloaded into memory on every request.
    """
    global GLOBAL_EMBEDDINGS
    if GLOBAL_EMBEDDINGS is None:
        GLOBAL_EMBEDDINGS = create_embeddings()
    return GLOBAL_EMBEDDINGS

def process_and_store_pdf(file_path):
    """
    Pipeline Helper: Loads the PDF, applies smart chunking, generates embeddings,
    initializes the FAISS vector database, and maps it in memory.
    """
    filename = os.path.basename(file_path)
    
    # Return cached FAISS vector database immediately if previously built
    if filename in GLOBAL_VECTOR_STORES:
        return GLOBAL_VECTOR_STORES[filename]
        
    print(f"[INFO] Extracting text content from uploaded PDF: {filename}")
    pages_data = load_pdf(file_path)
    if not pages_data:
        raise ValueError("No extractable text found in the uploaded PDF document.")
        
    print("[INFO] Applying smart chunking via RecursiveCharacterTextSplitter...")
    chunks, metadatas = split_text(pages_data, chunk_size=500, chunk_overlap=100)
    if not chunks:
        raise ValueError("Failed to generate document chunks.")
        
    print("[INFO] Generating HuggingFace embeddings and initializing FAISS database...")
    embeddings = get_or_create_embeddings()
    vector_store = store_in_faiss(chunks, metadatas, embeddings)
    
    if vector_store is None:
        raise ValueError("FAISS vector database creation failed.")
        
    # Store globally for high-speed local retrievals
    GLOBAL_VECTOR_STORES[filename] = vector_store
    print(f"[SUCCESS] FAISS vector database ready for '{filename}' ({len(chunks)} chunks cached).")
    return vector_store

def allowed_file(filename):
    """Verifies allowed file extension (.pdf)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    Route: GET /
    Renders the existing, unmodified user interface layout.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Route: POST /upload
    Upload Handling Flow: Saves uploaded document securely, invokes backend 
    chunking/embedding pipeline, and pre-warms the FAISS vector database.
    """
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
        
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected for upload"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Pre-process and cache the PDF in FAISS DB upon initial left-panel upload
            process_and_store_pdf(file_path)
            return jsonify({
                "status": "success", 
                "message": f"Successfully processed '{filename}'! FAISS vector database active and ready for queries."
            })
        except Exception as e:
            print(f"[ERROR] Pipeline preparation error: {e}")
            return jsonify({"status": "error", "message": f"Processing failure: {str(e)}"}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid format. Upload PDF files only."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Route: POST /ask
    Real RAG Execution Flow:
    1. Upload Handling: Safely reads question payload and uploaded document file.
    2. Retrieval: Performs FAISS vector similarity search to extract top relevant chunks.
    3. Groq Generation: Feeds context and question into ChatGroq LLM for real responses.
    4. Evaluation Flow: Invokes existing QA evaluator framework to render scores dynamically.
    """
    # Safe handling for empty queries
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty."}), 400

    # Safe handling for file validation
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "PDF document is required."}), 400
        
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No PDF file selected."}), 400
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 1. Pipeline Verification: Ensure document vector store is ready
        vector_store = process_and_store_pdf(file_path)

        # 2. Retrieval Flow: Search for top k=3 relevant context snippets
        print(f"[INFO] Executing FAISS similarity search for query: '{question}'")
        results = vector_store.similarity_search(question, k=3)

        if not results:
            return jsonify({
                "status": "success",
                "answer": "Unable to generate a reliable response from retrieved context. Try asking a more specific question.",
                "evaluation": "Score: 0/10\nVerdict: Incorrect\nReason: Query context absent from vector database."
            })

        # Append optional retrieved source pages for transparent dynamic UI rendering
        source_pages = []
        for doc in results:
            page_num = doc.metadata.get('page', 'Unknown')
            if page_num not in source_pages:
                source_pages.append(str(page_num))
        
        sources_footer = f"\n\n[Retrieved Context Sources: Page(s) {', '.join(source_pages)}]"

        # Aggregate context strings
        context = "\n\n".join([doc.page_content for doc in results])

        # 3. Groq Generation Flow: Assemble structured LLM prompt
        print("[INFO] Invoking Groq LLM engine to synthesize accurate answer...")
        llm = create_llm()
        
        prompt = f"""You are an AI assistant.

Answer the question using ONLY the provided context.
If the answer is not found in the context, say "Not found".

Be:
- Accurate
- Complete
- Concise

Context:
{context}

Question:
{question}

Answer:
"""
        response = llm.invoke(prompt)
        real_answer = response.content.strip()

        # Improve Empty-State Handling if generation returns "Not found" or empty string
        if not real_answer or "not found" in real_answer.lower():
            final_rendered_answer = "Unable to generate a reliable response from retrieved context. Try asking a more specific question."
        else:
            final_rendered_answer = real_answer + sources_footer

        # 4. Evaluation Flow: Run evaluation framework against combined context
        print("[INFO] Performing multi-dimensional validation evaluation...")
        evaluation_result = evaluate_answer(context, real_answer)

        # Return full payload dynamically consumed by unmodified index.html DOM logic
        return jsonify({
            "status": "success",
            "answer": final_rendered_answer,
            "evaluation": evaluation_result.strip()
        })

    except Exception as e:
        print(f"[ERROR] Live RAG failure: {e}")
        return jsonify({"status": "error", "message": f"Server execution exception: {str(e)}"}), 500

if __name__ == '__main__':
    # Running server locally with hot-reloading active
    app.run(debug=True, host='127.0.0.1', port=5000)
