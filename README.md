# 🚀 RAG Assistant (Day 1)

A simple **Retrieval-Augmented Generation (RAG)** system that processes documents and prepares them for intelligent question answering.

## 📌 Overview
This project builds the **core pipeline of a RAG system**:
**PDF → Text → Chunks → Embeddings → Vector Database (FAISS)**
It enables storing document knowledge in a way that can later be used for accurate, context-aware AI responses.

## ⚙️ Features
* 📄 PDF text extraction using PyPDF2
* ✂️ Text chunking for efficient processing
* 🧠 Embeddings using HuggingFace (local, no API required)
* 🔍 Vector storage using FAISS
* ⚡ Fast and lightweight setup

## 🛠️ Tech Stack
* Python
* LangChain
* FAISS
* Sentence Transformers
* PyPDF2

## 📂 Project Structure
```
rag-assistant/
│
├── app.py          # Main execution file
├── utils.py        # Core RAG functions
├── data/           # Input PDF files
└── requirements.txt
```

## 🚀 How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Run the project
```
python app.py
```
## ✅ Output
```
Chunks created: XXXX
FAISS vector database created successfully!
```
## 📊 Current Status

✔ Day 1 Completed
* Document processing pipeline built
* Vector database created

## 🔥 Next Steps
* Add question-answering system
* Integrate LLM for responses
* Build chat interface

## 💡 Learning Outcome
* Understanding of RAG architecture
* Working with embeddings and vector databases
* Handling real-world AI pipelines

## 👩‍💻 Author
Shreya Adsul
