import librosa
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pandas as pd
import os

# Paths for dataset images and descriptions
dataset_dir = '/workspace/Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB-1/vector-search-app/dataset'
descriptions_file = '/workspace/Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB-1/vector-search-app/descriptions/descriptions.csv'

# Load descriptions from CSV
descriptions_df = pd.read_csv(descriptions_file)
descriptions = descriptions_df.set_index('filename').T.to_dict('list')

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("multimedia_collection")

# Load CLIP model and processor for generating embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to process media files and generate embeddings
def process_media(media_path):
    if media_path.endswith(".jpg") or media_path.endswith(".jpeg"):
        # Process image
        image = Image.open(media_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
    elif media_path.endswith(".mp4"):
        # Process video by taking a frame from the middle
        video = VideoFileClip(media_path)
        frame = video.get_frame(video.duration / 2)  # Get the middle frame
        image = Image.fromarray(frame)
        inputs = processor(images=image, return_tensors="pt", padding=True)
    elif media_path.endswith(".mp3"):
        # Process audio as a spectrogram
        y, sr = librosa.load(media_path, sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize and convert to RGB format
        mel_spectrogram_db = 255 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        mel_spectrogram_db = mel_spectrogram_db.astype(np.uint8)
        mel_spectrogram_db = cv2.resize(mel_spectrogram_db, (224, 224))
        mel_spectrogram_db = np.stack((mel_spectrogram_db,) * 3, axis=-1)
        
        inputs = processor(images=mel_spectrogram_db, return_tensors="pt", padding=True)

    with torch.no_grad():
        media_embedding = model.get_image_features(**inputs).numpy().flatten().tolist()
    return media_embedding

# Function to search media based on a text query
def search_media(query):
    if not query.strip():
        return None, "Please enter a query.", ""

    print(f"\nQuery: {query}")

    # Start tracking query time
    query_start_time = time.time()

    # Prepare media paths from dataset
    media_paths = [os.path.join(dataset_dir, filename) for filename in descriptions.keys()]

    # Process each media and generate embeddings
    media_embeddings = []
    ingestion_start_time = time.time()  # Start ingestion timing
    for media_path in media_paths:
        media_embedding = process_media(media_path)
        media_embeddings.append(media_embedding)
    ingestion_end_time = time.time()
    ingestion_time = ingestion_end_time - ingestion_start_time

    # Add media embeddings to the collection
    collection.add(
        embeddings=media_embeddings,
        metadatas=[{"media": media_path} for media_path in media_paths],
        ids=[str(i) for i in range(len(media_paths))]
    )

    # Generate embedding for query text
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy().flatten().tolist()

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    # Retrieve the matched media file
    if len(results['metadatas']) == 0:
        return None, "No matching media found.", ""

    result_media_path = results['metadatas'][0]['media']
    matched_media_index = int(results['ids'][0])
    matched_media_embedding = media_embeddings[matched_media_index]

    # Calculate accuracy score based on cosine similarity
    accuracy_score = cosine_similarity([matched_media_embedding], [query_embedding])[0][0]
    
    query_end_time = time.time()
    query_time = query_end_time - query_start_time

    # Determine media type for display
    if result_media_path.endswith(".jpg") or result_media_path.endswith(".jpeg"):
        result_media = Image.open(result_media_path)
        media_type = "Image"
    elif result_media_path.endswith(".mp4"):
        result_media = None
        media_type = f"Video: {result_media_path}"
    elif result_media_path.endswith(".mp3"):
        result_media = None
        media_type = f"Audio: {result_media_path}"

    file_name = os.path.basename(result_media_path)

    return result_media, f"Media type: {media_type}\nImage ingestion time: {ingestion_time:.4f} seconds\nAccuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", file_name

# Function to populate the query input box with a suggested query
def populate_query(suggested_query):
    return suggested_query

# Gradio Interface Launch
with gr.Blocks() as gr_interface:
    gr.Markdown("# Multimedia Vector Search with ChromaDB and Local Dataset")
    custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")
    
    with gr.Row():
        submit_button = gr.Button("Submit Query")
        cancel_button = gr.Button("Cancel")

    media_output = gr.Image(type="pil", label="Result Media")
    accuracy_output = gr.Textbox(label="Performance Metrics")

    submit_button.click(fn=search_media, inputs=custom_query, outputs=[media_output, accuracy_output])
    cancel_button.click(fn=lambda: (None, ""), outputs=[media_output, accuracy_output])

# Launch the Gradio interface
gr_interface.launch(share=True)
