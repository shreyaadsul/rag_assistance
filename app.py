import os
from dotenv import load_dotenv
from utils import load_pdf, split_text, create_embeddings, store_in_faiss, create_llm

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Define path to the PDF
    pdf_path = "data/insurance XII.pdf"
    
    # 1. Load PDF and get text with initial metadata
    print("Loading PDF...")
    pages_data = load_pdf(pdf_path)
    
    if not pages_data:
        print("No text extracted from PDF or file not found.")
        return
    
    # 2. Split text into chunks and generate chunk-level metadata
    print("Splitting text into chunks...")
    chunks, metadatas = split_text(pages_data, chunk_size=500, chunk_overlap=100)
    
    if not chunks:
        print("No chunks generated.")
        return
        
    # 3. Create embeddings model
    print("Initializing embedding model...")
    embeddings = create_embeddings()
    
    # 4. Store chunks and metadata in FAISS vector database
    print("Creating FAISS vector database...")
    vector_store = store_in_faiss(chunks, metadatas, embeddings)
    
    if vector_store is not None:
        print(f"FAISS vector database created successfully with {len(chunks)} chunks!\n")
    else:
        print("Failed to create FAISS vector store.")
        return
        
    # 5. Retrieval step
    print("--- Retrieval System Ready ---")
    query = input("Enter your search query: ")
    
    if not query.strip():
        print("Empty query provided. Exiting.")
        return
        
    print(f"\nSearching for top 3 results for: '{query}'...")
    
    # Perform similarity search to get top k=3 chunks
    results = vector_store.similarity_search(query, k=3)
    
    # Print the retrieved chunks along with their metadata
    print("\n--- Search Results ---")
    for i, doc in enumerate(results, start=1):
        print(f"Result #{i}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 50)

    # 6. LLM Generation step
    if not results:
        print("\nNo results found in the document to answer your query.")
        return

    print("\nGenerating answer...")
    
    # Combine the retrieved content into a single context string
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Create the prompt combining context and user query
    prompt = f"""Answer the question using ONLY the context below. If the answer is not in the context, say 'Not found'.

Context:
{context}

Question: {query}
"""

    # Initialize the LLM and generate the response
    llm = create_llm()
    response = llm.invoke(prompt)
    
    # Print the final generated answer clearly
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(response.content)
    print("="*50)

if __name__ == "__main__":
    main()
