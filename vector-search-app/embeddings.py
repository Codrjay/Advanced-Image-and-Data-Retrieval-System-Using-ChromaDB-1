from sentence_transformers import SentenceTransformer

# Initialize the model for embedding generation
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embedding(text):
    return model.encode(text)
