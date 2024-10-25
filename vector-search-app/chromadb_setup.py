import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
def init_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("text_search_collection")
    return collection

# Add data to ChromaDB
def add_data_to_chromadb(collection, documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for doc in documents:
        embedding = model.encode(doc["text"])
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            metadatas=[{"text": doc["text"]}]
        )
    print("Data added to ChromaDB!")
