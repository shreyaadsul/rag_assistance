import os
from dotenv import load_dotenv
from utils import load_pdf, split_text, create_embeddings, store_in_faiss, create_llm, evaluate_answer

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
        
    # 5. Interactive Multi-Query Loop
    print("\n--- Retrieval System Ready ---")
    print("Type 'exit' to stop the program.\n")
    
    # Initialize the LLM once before the loop
    llm = create_llm()
    
    while True:
        query = input("\nEnter your search query: ")
        
        # Handle empty input
        if not query.strip():
            print("Empty query provided. Please try again.")
            continue
            
        # Exit condition
        if query.strip().lower() == 'exit':
            print("Exiting the interactive system. Goodbye!")
            break
            
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
            continue

        print("\nGenerating answer...")
        
        # Combine the retrieved content into a single context string
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Create the prompt combining context and user query
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
{query}

Answer:
"""

        # Generate the response
        response = llm.invoke(prompt)
        answer = response.content
        
        # Print the final generated answer clearly
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print("="*50)
        print(answer)
        print("="*50)

        # 7. Evaluation step
        print("\nEvaluating answer...")
        evaluation = evaluate_answer(context, answer)
        
        print("\n" + "="*50)
        print("EVALUATION RESULT:")
        print("="*50)
        print(evaluation)
        print("="*50)

if __name__ == "__main__":
    main()
