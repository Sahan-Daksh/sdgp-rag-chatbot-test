import os
import streamlit as st
import tempfile
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy
from whisper import load_model
from groq import Groq
import json
import openai

# API Keys
GROQ_API_KEY = "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"  # Replace with your Groq API key
OPENAI_API_KEY = "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"  # Replace with your OpenAI key
openai.api_key = OPENAI_API_KEY
client = Groq(api_key=GROQ_API_KEY)

# Constants
PDF_PATH = "ugc_student_handbook_2023-2024.pdf"
PROCESSED_DATA = "processed_data.json"

# Model loading
@st.cache_resource
def load_models():
    whisper_model = load_model("base")  # Whisper base model for voice input
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # SentenceTransformer for embeddings
    spacy_model = spacy.load("en_core_web_sm")  # Spacy for NER
    return whisper_model, embed_model, spacy_model

whisper_model, embed_model, spacy_model = load_models()

# Step 1: PDF Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Chunking and Metadata with Spacy
def preprocess_with_metadata(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks, metadata = [], []

    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)

        # Metadata extraction using Spacy
        doc = spacy_model(chunk)
        keywords = list(set(ent.text for ent in doc.ents))
        metadata.append({"chunk_text": chunk, "keywords": keywords})

        start += chunk_size - overlap
    return chunks, metadata

# Step 3: FAISS Index Creation
def generate_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Save preprocessed data
def save_processed_data(chunks, metadata, embeddings, file_path):
    data = {"chunks": chunks, "metadata": metadata, "embeddings": embeddings.tolist()}
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

# Load preprocessed data
def load_preprocessed_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    metadata = data["metadata"]
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, metadata, index

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, chunks, index, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=k)
    return [chunks[i] for i in indices[0]]

# Query OpenAI for Enrichment
def enrich_with_openai(chunk):
    prompt = f"Organize the following text and make it more coherent:\n{chunk}"
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=200, temperature=0.3
    )
    return response.choices[0].text.strip()

# Query the Groq LLM
def query_llm_groq(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that uses the provided context to answer questions."},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model="llama3-8b-8192", messages=messages, temperature=0.5, max_tokens=512
    )
    return response.choices[0].message.content

# Preprocess the PDF if not already done
if not os.path.exists(PROCESSED_DATA):
    with st.spinner("Preprocessing PDF... This might take a moment."):
        raw_text = extract_text_from_pdf(PDF_PATH)
        chunks, metadata = preprocess_with_metadata(raw_text)
        faiss_index, embeddings = generate_faiss_index([meta['chunk_text'] for meta in metadata])
        save_processed_data([meta['chunk_text'] for meta in metadata], metadata, embeddings, PROCESSED_DATA)
else:
    chunks, metadata, faiss_index = load_preprocessed_data(PROCESSED_DATA)

# Streamlit App UI
st.title("Enhanced RAG Assistant: Text and Voice Queries")
st.sidebar.title("Select Input Mode")

# Sidebar choice
query_mode = st.sidebar.radio("Choose Query Mode", ["Voice Input", "Text Input"])

# Voice Query Mode
if query_mode == "Voice Input":
    st.header("Voice Query")
    uploaded_audio = st.file_uploader("Upload an audio file (wav/mp3)", type=["wav", "mp3"])

    if uploaded_audio:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())
            audio_path = temp_audio.name
        
        st.audio(audio_path, format="audio/wav")

        with st.spinner("Transcribing audio..."):
            transcription = whisper_model.transcribe(audio_path).get("text", "")
        os.remove(audio_path)

        if transcription:
            st.success("Transcription:")
            st.write(transcription)

            with st.spinner("Retrieving relevant content..."):
                relevant_chunks = retrieve_relevant_chunks(transcription, chunks, faiss_index)
                context = "\n".join(relevant_chunks)

            response = query_llm_groq(transcription, context)
            st.subheader("Assistant Response:")
            st.write(response)

# Text Query Mode
if query_mode == "Text Input":
    st.header("Text Query")
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if query:
            with st.spinner("Retrieving relevant content..."):
                relevant_chunks = retrieve_relevant_chunks(query, chunks, faiss_index)
                context = "\n".join(relevant_chunks)

            response = query_llm_groq(query, context)
            st.subheader("Assistant Response:")
            st.write(response)
