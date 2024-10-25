# ChromaDB Text-to-Image Vector Search

This project leverages **ChromaDB** and **OpenAI's CLIP** model to perform a **text-to-image vector search**. Users can upload images dynamically or input text queries to retrieve the most similar image based on vector embeddings. Additionally, the interface displays key performance metrics, including **accuracy score** and **query time**.

## Features
- **Dynamic Image and Text Uploading**: Users can upload their own images and input queries, making searches flexible and personalized.
- **Efficient Text-to-Image Search**: The app uses vector embeddings to match text queries with images accurately.
- **Performance Metrics**: Displays the following in real-time:
  - Image ingestion time (formatted to four decimal places)
  - Accuracy score
  - Query processing time

## Technologies Used
- **Python**
- **Torch** (PyTorch) for handling embeddings
- **ChromaDB** as a vector database for efficient storage and retrieval
- **Gradio** for a user-friendly interface
- **Hugging Face Transformers** for the CLIP model

## Getting Started

### Prerequisites
- Python 3.x
- [Google Colab](https://colab.research.google.com/) (recommended for quick setup)
- Install dependencies with `pip install -r requirements.txt`

### Installation
Clone this repository:
```bash
git clone https://github.com/thebadsektor/chromadb-text-to-image.git
cd chromadb-text-to-image
