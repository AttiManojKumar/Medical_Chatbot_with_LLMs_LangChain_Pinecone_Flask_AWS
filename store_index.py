from src.helper import (
    INDEX_NAME,
    build_vector_store,
    create_or_get_pinecone_index,
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    get_data_path,
    get_pinecone_client,
    get_relevant_docs,
    get_vector_count,
    load_api_keys,
    load_pdf_file,
    require_key,
    text_split,
    upload_chunks_to_pinecone,
)


def main() -> None:
    """Load the PDF, create embeddings, and store them in Pinecone."""
    print("Step 1: Loading API keys from .env", flush=True)
    keys = load_api_keys()
    pinecone_api_key = require_key(keys, "pinecone")

    print("Step 2: Loading HuggingFace embedding model", flush=True)
    embeddings = download_hugging_face_embeddings()
    print("Embedding dimension:", len(embeddings.embed_query("Hello world")), flush=True)

    print("Step 3: Connecting to Pinecone", flush=True)
    pc = get_pinecone_client(pinecone_api_key)
    index = create_or_get_pinecone_index(pc, INDEX_NAME)

    print("Step 4: Checking existing vectors in Pinecone", flush=True)
    vector_count = get_vector_count(index)
    print("Current Pinecone vector count:", vector_count, flush=True)

    if vector_count == 0:
        print("Step 5: Finding the data folder", flush=True)
        data_path = get_data_path()
        print("Data folder:", data_path, flush=True)

        print("Step 6: Loading PDF files", flush=True)
        extracted_data = load_pdf_file(data_path)
        print("Pages loaded:", len(extracted_data), flush=True)

        print("Step 7: Cleaning document metadata", flush=True)
        filtered_data = filter_to_minimal_docs(extracted_data)
        print("Filtered documents:", len(filtered_data), flush=True)

        print("Step 8: Splitting documents into chunks", flush=True)
        text_chunks = text_split(filtered_data)
        print("Text chunks:", len(text_chunks), flush=True)

        print("Step 9: Uploading chunks to Pinecone", flush=True)
        upload_chunks_to_pinecone(text_chunks, embeddings, INDEX_NAME)
        vector_count = get_vector_count(index)
        print("Upload complete. New vector count:", vector_count, flush=True)
    else:
        print("Step 5: Pinecone already has vectors. Skipping PDF upload.", flush=True)

    print("Step 6: Testing retrieval from Pinecone", flush=True)
    docsearch = build_vector_store(INDEX_NAME, embeddings)
    docs = get_relevant_docs(docsearch, "What is acetaminophen?", top_k=3)

    print("Relevant chunks returned:", len(docs), flush=True)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:700])
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    main()
