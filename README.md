# 🧠 Advanced RAG AI Assistant with Answer Evaluation

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** that answers questions from documents using vector search and LLMs.

Unlike basic RAG systems, this project introduces an **Answer Evaluation Layer**, making it more reliable, explainable, and research-oriented.

---

## 🚀 Features

- 📄 PDF Document Processing
- 🔍 Smart Chunking (Recursive Text Splitter)
- 🧠 Embeddings using HuggingFace
- 📦 Vector Database (FAISS)
- 🔎 Top-K Retrieval System
- 🤖 LLM-based Answer Generation (Groq - LLaMA 3.1)
- 📊 **Answer Evaluation System (CORE FEATURE)**
  - Score (0–10)
  - Verdict (Correct / Partial / Incorrect)
  - Reasoning
- 🔁 Multi-query interactive system
- 📍 Metadata tracking (source, page, chunk_id)

---

## 🧠 System Architecture


User Question
↓
Embedding Model
↓
FAISS Vector DB
↓
Top-K Relevant Chunks
↓
LLM (Groq)
↓
Generated Answer
↓
Evaluation System (LLM)
↓
Score + Verdict + Reason


---

## ⚙️ Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM (LLaMA 3.1)
- PyPDF2

---

## 📂 Project Structure


rag-assistant/
├── app.py # Main application
├── utils.py # Helper functions
├── data/ # PDF files
├── requirements.txt
└── README.md


---

## 🧪 Experiments (Research Component)

### 🔬 Experiment 1: Chunk Size Optimization

| Chunk Size | Avg Score |
|------------|----------|
| 300        | 8.67     |
| 500        | ⭐ 9.33   |
| 800        | 9.0      |

👉 **Best: 500**

---

### 🔬 Experiment 2: Top-K Retrieval

| K Value | Avg Score |
|--------|----------|
| 2      | 8.67     |
| 3      | ⭐ 9.0    |
| 5      | 9.0      |

👉 **Best: k = 3**

---

### 🔬 Experiment 3: Prompt Engineering

| Prompt Type | Avg Score | Behavior |
|-------------|----------|----------|
| Basic       | ⭐ 9.33   | Detailed |
| Strict      | 9.0      | Controlled |

---

## 🧠 Key Insights

- Moderate chunk sizes (500) give best balance of context + precision  
- Retrieval size (k = 3) provides optimal context without noise  
- Prompt design affects:
  - Completeness
  - Hallucination control  

---

## 🔥 Core Innovation

### ✅ Answer Evaluation System

Unlike traditional RAG:

👉 This system evaluates its own answers

**Output:**

Score: 9
Verdict: Correct
Reason: Answer is supported by context but missing minor detail


---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant
pip install -r requirements.txt
▶️ Run the Project
python app.py

Then:

Enter your query: What is insurance?
📈 Example Output
FINAL ANSWER:
Life insurance provides financial protection...

EVALUATION:
Score: 9
Verdict: Correct
Reason: Accurate but missing minor detail
🎯 Future Improvements
Web UI (Streamlit / React)
Multi-document support
Chat memory
API deployment (FastAPI)

👩‍💻 Author
Shreya Adsul
AIML Student
Interested in AI Systems, NLP, and Automation