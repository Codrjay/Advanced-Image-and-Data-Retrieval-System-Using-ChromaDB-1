# Multimedia Vector Search (CLIP + ChromaDB)

Search images, videos, and audio files using plain-text queries. Media is converted into CLIP embeddings, stored in ChromaDB, and retrieved by vector similarity through a simple Gradio interface.


## Features

- **Multiple media types:** images (JPEG), videos (MP4), and audio (MP3)
- **Text-to-media search:** type a description and get the closest matches
- **CLIP embeddings:** generated with the CLIP model from Hugging Face
- **ChromaDB storage:** fast vector storage and similarity search
- **Gradio interface:** no code needed to run queries

## How It Works

1. Media files are loaded from `dataset/`, with text descriptions read from `descriptions/descriptions.csv`.
2. Each item is converted into an embedding using CLIP.
3. Embeddings are stored in a ChromaDB collection.
4. Your text query is embedded the same way.
5. ChromaDB returns the nearest items by vector similarity, and Gradio displays them.

## Tech Stack

Python, PyTorch, Hugging Face Transformers (CLIP), ChromaDB, Gradio, librosa, MoviePy, OpenCV, Pillow, pandas, scikit-learn

## Project Structure

```
vector-search-app/
├── dataset/                 # Media files (JPEG, MP4, MP3)
├── descriptions/
│   └── descriptions.csv     # Text descriptions for the media files
├── app.py                   # Main application
└── requirements.txt         # Python dependencies
```

## Setup

```bash
git clone https://github.com/Codrjay/Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB-1.git
cd Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB-1/vector-search-app

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

1. Place your media files in `dataset/`.
2. Make sure `descriptions/descriptions.csv` describes those files.
3. Start the app:
   ```bash
   python app.py
   ```
4. Open the local link printed in the terminal and enter a search query.



## Limitations

- Search quality depends on the quality and coverage of `descriptions.csv`.
- Results are only as broad as the files in `dataset/`.
- Large media collections will take longer to embed on first run.

## Possible Improvements

- Persist the ChromaDB collection so embeddings are not rebuilt on every run
- Support image-to-image and audio-to-audio search
- Add result filtering by media type

## Author

**Emmanuel James** ([@Codrjay](https://github.com/Codrjay)), data analyst and AI practitioner.
