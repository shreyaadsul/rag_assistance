import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Global cache for embeddings to avoid reloading
_GLOBAL_EMBEDDINGS = None

def get_or_create_embeddings():
    """
    Returns a cached instance of HuggingFaceEmbeddings.
    """
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _GLOBAL_EMBEDDINGS

def create_llm():
    """
    Initializes the Language Model for generating answers using Groq.
    """
    if not os.getenv("GROQ_API_KEY"):
        print("\n[ERROR] GROQ_API_KEY is missing from your .env file!")
        print("Please create a free API key at https://console.groq.com/keys and add it to .env")
        # In a real app, you might raise an exception here instead of exit()
        raise ValueError("GROQ_API_KEY is missing from your .env file.")
        
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
