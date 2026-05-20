import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.llm_manager import get_or_create_embeddings

FAISS_INDEX_PATH = os.path.join("data", "faiss_index")

_GLOBAL_VECTOR_STORE = None

def get_vector_store():
    """
    Returns the unified FAISS vector store. 
    Loads from disk if it exists, otherwise creates an empty one.
    """
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is not None:
        return _GLOBAL_VECTOR_STORE

    embeddings = get_or_create_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            _GLOBAL_VECTOR_STORE = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("[INFO] Loaded existing FAISS index from disk.")
            return _GLOBAL_VECTOR_STORE
        except Exception as e:
            print(f"[ERROR] Failed to load FAISS index: {e}")
            # Fall back to creating a new one if load fails
            pass
    
    # Create empty FAISS index (LangChain FAISS doesn't have a direct "create empty" without texts,
    # so we initialize with a dummy document and then delete it or just use from_texts with a space)
    # A cleaner approach in LangChain:
    dummy_doc = Document(page_content="dummy", metadata={"source": "dummy"})
    _GLOBAL_VECTOR_STORE = FAISS.from_documents([dummy_doc], embeddings)
    # We leave the dummy document, it won't hurt, but ideally we'd manage the index better.
    # We can delete it by its id if we tracked it, or just ignore it.
    
    return _GLOBAL_VECTOR_STORE

def save_vector_store():
    """Saves the unified vector store to disk."""
    if _GLOBAL_VECTOR_STORE is not None:
        os.makedirs("data", exist_ok=True)
        _GLOBAL_VECTOR_STORE.save_local(FAISS_INDEX_PATH)

def add_to_vector_store(chunks, metadatas):
    """
    Adds new chunks to the unified vector store and saves to disk.
    """
    if not chunks:
        return
        
    vector_store = get_vector_store()
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    save_vector_store()
    return vector_store

def delete_from_vector_store(filename):
    """
    Deletes all chunks associated with a specific document.
    FAISS in Langchain supports delete via doc_ids. We need to find the doc_ids first.
    Since we don't store doc_ids in metadata manager, we have to rebuild or filter the index.
    A simple approach for this prototype: 
    Iterate through the index, keep docs not matching the filename, and rebuild the FAISS store.
    Warning: This is slow for very large databases, but acceptable for this scale.
    """
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None and not os.path.exists(FAISS_INDEX_PATH):
        return
        
    vector_store = get_vector_store()
    
    # Langchain FAISS doesn't expose an easy way to iterate all documents.
    # We will access the underlying docstore.
    try:
        docstore = vector_store.docstore._dict
        ids_to_delete = []
        for doc_id, doc in docstore.items():
            if doc.metadata.get("source") == filename:
                ids_to_delete.append(doc_id)
                
        if ids_to_delete:
            vector_store.delete(ids_to_delete)
            save_vector_store()
            print(f"[INFO] Deleted {len(ids_to_delete)} chunks for {filename} from FAISS index.")
    except Exception as e:
        print(f"[ERROR] Failed to delete from vector store: {e}")
