import os
import subprocess

from faster_whisper import WhisperModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import CHROMA_DIR, FFMPEG_PATH, EMBEDDING_MODEL


# ---------- Load PDFs ----------
def load_pdfs(pdf_paths):
    pages = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages.extend(loader.load())
    return pages


# ---------- Transcribe YouTube ----------
def transcribe_youtube(url):
    audio_template = "temp_audio.%(ext)s"
    audio_file = "temp_audio.mp3"

    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "--ffmpeg-location", FFMPEG_PATH,
            "-o", audio_template,
            url
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0 or not os.path.exists(audio_file):
        raise RuntimeError(f"YouTube download failed:\n{result.stderr}")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_file)

    text = " ".join(seg.text for seg in segments)

    return Document(
        page_content=text,
        metadata={"source": "youtube", "url": url}
    )


# ---------- Chunk ----------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# ---------- USER-DRIVEN INGEST ----------
def ingest_documents(pdf_paths=None, youtube_url=None):
    if not pdf_paths and not youtube_url:
        raise ValueError("Upload at least one PDF or provide a YouTube URL.")

    docs = []

    if pdf_paths:
        docs.extend(load_pdfs(pdf_paths))

    if youtube_url:
        docs.append(transcribe_youtube(youtube_url))

    chunks = chunk_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="knowledge_base"
    )

    vectordb.persist()
    return len(chunks)
