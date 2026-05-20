from backend.vector_manager import get_vector_store

def retrieve_context(question, doc_filter="all", top_k=5):
    """
    Retrieves the top K relevant chunks for the given question.
    If doc_filter is provided (and not 'all'), it filters by the source filename.
    Returns:
        results_with_scores: list of (Document, score) tuples
        formatted_retrieval_scores: list of dicts for UI rendering
        context_string: concatenated string of all retrieved chunk contents
        source_pages: list of distinct source pages
    """
    vector_store = get_vector_store()
    
    # Check if the store is effectively empty
    if len(vector_store.docstore._dict) == 0:
        return [], [], "", []
        
    search_kwargs = {"k": top_k}
    if doc_filter and doc_filter.lower() != "all":
        search_kwargs["filter"] = {"source": doc_filter}
        
    try:
        results_with_scores = vector_store.similarity_search_with_score(question, **search_kwargs)
    except Exception as e:
        print(f"[ERROR] Similarity search failed: {e}")
        return [], [], "", []

    retrieval_scores = []
    results = []
    source_pages = []

    for idx, (doc, score) in enumerate(results_with_scores, start=1):
        results.append(doc)
        source = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', 'Unknown')
        
        page_ref = f"{source} (Page {page_num})"
        if page_ref not in source_pages:
            source_pages.append(page_ref)
            
        relevance = max(0, min(100, int((1.0 - float(score)) * 100)))
        
        preview = doc.page_content.strip()
        if len(preview) > 120:
            preview = preview[:120] + "..."
            
        retrieval_scores.append({
            "chunk_id": chunk_id,
            "source": source,
            "page": page_num,
            "relevance": relevance,
            "raw_score": float(score),
            "preview": preview,
            "content": doc.page_content
        })
        
    context_string = "\n\n".join([doc.page_content for doc in results])
    return results_with_scores, retrieval_scores, context_string, source_pages
