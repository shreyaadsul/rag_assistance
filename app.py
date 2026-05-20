import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from backend.ingestion import process_and_store_pdf
from backend.retriever import retrieve_context
from backend.evaluator import evaluate_answer
from backend.metadata_manager import get_all_documents, remove_document
from backend.vector_manager import delete_from_vector_store
from backend.llm_manager import create_llm

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/documents', methods=['GET'])
def list_documents():
    """Returns the list of indexed documents and their metadata."""
    docs = get_all_documents()
    return jsonify({"status": "success", "documents": docs})

@app.route('/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Deletes a document from vector store and metadata."""
    try:
        delete_from_vector_store(filename)
        remove_document(filename)
        # Also remove the physical file if it exists
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"status": "success", "message": f"Deleted {filename}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handles multiple PDF uploads and ingestion."""
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
        
    files = request.files.getlist('pdf_file')
    if not files or files[0].filename == '':
        return jsonify({"status": "error", "message": "No file selected for upload"}), 400
        
    processed_files = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                process_and_store_pdf(file_path)
                processed_files.append(filename)
            except Exception as e:
                print(f"[ERROR] Processing error for {filename}: {e}")
                errors.append(f"{filename}: {str(e)}")
        else:
            errors.append(f"{file.filename}: Invalid format")

    if errors and not processed_files:
        return jsonify({"status": "error", "message": " | ".join(errors)}), 500
        
    msg = f"Successfully processed: {', '.join(processed_files)}."
    if errors:
        msg += f" Errors: {', '.join(errors)}"
        
    return jsonify({"status": "success", "message": msg})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Executes the Multi-Document RAG Retrieval and Evaluation."""
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty."}), 400

    doc_filter = request.form.get('doc_filter', 'all').strip()

    try:
        # 1. Retrieval Flow
        print(f"[INFO] Executing similarity search for query: '{question}' with filter: '{doc_filter}'")
        results_with_scores, retrieval_scores, context, source_pages = retrieve_context(question, doc_filter, top_k=5)

        if not results_with_scores:
            return jsonify({
                "status": "success",
                "answer": "Unable to generate a reliable response from retrieved context. Ensure documents are uploaded or try asking a more specific question.",
                "evaluation": {
                    "correctness_score": 0,
                    "confidence_score": 0,
                    "hallucination_risk": "High",
                    "completeness": "Low",
                    "verdict": "Unreliable",
                    "detailed_reason": "Query context absent from vector database."
                },
                "retrieval_scores": []
            })

        sources_footer = f"\n\n[Retrieved Context Sources: {', '.join(source_pages)}]"

        # 2. LLM Generation
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

        if not real_answer or "not found" in real_answer.lower():
            final_rendered_answer = "Unable to generate a reliable response from retrieved context. Try asking a more specific question."
        else:
            final_rendered_answer = real_answer + sources_footer

        # 3. Evaluation Flow
        print("[INFO] Performing multi-dimensional validation evaluation...")
        evaluation_result = evaluate_answer(context, real_answer)

        return jsonify({
            "status": "success",
            "answer": final_rendered_answer,
            "evaluation": evaluation_result,
            "retrieval_scores": retrieval_scores
        })

    except Exception as e:
        print(f"[ERROR] Live RAG failure: {e}")
        return jsonify({"status": "error", "message": f"Server execution exception: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
