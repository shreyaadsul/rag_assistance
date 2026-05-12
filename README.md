# 🚀 Advanced RAG Assistant with Answer Evaluation

An advanced Retrieval-Augmented Generation (RAG) system built using Flask, FAISS, HuggingFace Embeddings, and Groq LLM.

This project allows users to upload PDF documents, ask context-aware questions, retrieve relevant document chunks using vector similarity search, generate AI-powered answers, and evaluate answer correctness using an intelligent evaluation module.

# 🧠 Project Features

## ✅ PDF Upload System

* Upload PDF documents dynamically
* Automatic document processing pipeline

## ✅ Smart Chunking

* Recursive text chunking
* Chunk overlap support
* Metadata preservation

## ✅ Semantic Retrieval

* HuggingFace Embeddings
* FAISS Vector Database
* Top-k similarity search

## ✅ AI Answer Generation

* Groq LLM integration
* Context-grounded responses
* Reduced hallucination approach

## ✅ Answer Evaluation System ⭐

* Score generated answers
* Verdict generation
* Reasoning explanation
* Self-evaluating RAG pipeline

## ✅ Flask Web UI

* Modern dark-themed interface
* Upload PDFs
* Ask questions interactively
* View answers and evaluation results

# ⚙️ System Architecture

```text
User Query
    ↓
Embedding Model
    ↓
FAISS Vector Database
    ↓
Top-K Retrieval
    ↓
Groq LLM (Answer Generation)
    ↓
Answer Evaluation Module
    ↓
Final Output (Answer + Score + Verdict)
```

# 🧪 Research Experiments

## 🔹 Chunk Size Experiment

| Chunk Size | Avg Score |
| ---------- | --------- |
| 300        | 8.67      |
| 500        | 9.33 ⭐    |
| 800        | 9.0       |

Best performance achieved at chunk size = 500.

## 🔹 Top-K Retrieval Experiment

| K Value | Avg Score |
| ------- | --------- |
| 2       | 8.67      |
| 3       | 9.0 ⭐     |
| 5       | 9.0       |

Best retrieval performance achieved at k = 3.

## 🔹 Prompt Engineering

| Prompt Type | Behavior                          |
| ----------- | --------------------------------- |
| Basic       | Detailed responses                |
| Strict      | Controlled and grounded responses |

# 🔥 Research Contribution

This project focuses on improving RAG reliability using:

* retrieval optimization
* answer evaluation
* hallucination reduction
* grounded AI response generation

Unlike traditional RAG systems that only generate answers, this system also evaluates the correctness and contextual grounding of generated responses.

# 🛠️ Technologies Used

* Python
* Flask
* FAISS
* HuggingFace Embeddings
* Groq LLM
* LangChain
* HTML/CSS
* RecursiveCharacterTextSplitter

# 📂 Project Structure

```text
rag_assistant/
│
├── app.py
├── utils.py
├── requirements.txt
├── templates/
├── static/
├── uploads/
└── data/
```

# ▶️ How to Run

## 1. Clone Repository

```bash
git clone <your-repo-link>
cd rag_assistant
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

## 4. Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

# 📄 Research Paper

**Title:**
Improving Retrieval-Augmented Generation Systems using Answer Evaluation and Retrieval Optimization

This research explores:

* retrieval quality
* chunking optimization
* prompt engineering
* answer evaluation systems
* RAG reliability improvements

# 🚀 Future Improvements

* Multi-document support
* Confidence scoring
* Chat memory
* Advanced reranking
* Better summarization pipeline

# 👩‍💻 Author
Shreya Adsul
TY B.Sc. CS (AIML)
Nagindas Khandwala College
