import streamlit as st
import os
import librosa
import soundfile as sf
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Initialize Groq client
client = Groq(api_key="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")

# Function: Audio Preprocessing
def preprocess_audio(input_path, target_sample_rate=16000):
    try:
        # Load audio file
        y, original_sr = librosa.load(input_path, sr=None)
        # Convert to mono if stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        # Resample
        y_resampled = librosa.resample(y_trimmed, orig_sr=original_sr, target_sr=target_sample_rate)
        # Normalize
        y_normalized = librosa.util.normalize(y_resampled)
        # Save preprocessed audio
        temp_output = input_path.replace('.wav', '_preprocessed.wav')
        sf.write(temp_output, y_normalized, target_sample_rate)
        return temp_output
    except Exception as e:
        st.error(f"Audio preprocessing error: {e}")
        return input_path

# Function: Transcription using Groq
def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language="en",
            )
            return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function: Split text into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function: Generate embeddings
def generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text_chunks), model

# Function: Create FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function: Retrieve relevant chunks
def retrieve_relevant_chunks(query, text_chunks, index, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    return [text_chunks[idx] for idx in indices[0]]

# Streamlit UI
st.title("Audio Transcription and PDF Querying System")

st.sidebar.header("Inputs")
audio_file = st.sidebar.file_uploader("Upload Audio File (MP3/WAV)", type=["mp3", "wav"])
pdf_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if audio_file and pdf_file:
    # Preprocess and Transcribe Audio
    with st.spinner("Processing Audio..."):
        input_audio_path = os.path.join("temp", audio_file.name)
        with open(input_audio_path, "wb") as f:
            f.write(audio_file.read())
        processed_audio = preprocess_audio(input_audio_path)
        transcription = transcribe_audio(processed_audio)

    if transcription:
        st.subheader("Transcription")
        st.write(transcription)

        # Process PDF for Querying
        with st.spinner("Processing PDF..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            text_chunks = split_text_into_chunks(pdf_text)
            embeddings, embedding_model = generate_embeddings(text_chunks)
            faiss_index = create_faiss_index(np.array(embeddings))

        # Retrieve Chunks and Query
        with st.spinner("Fetching relevant information..."):
            relevant_chunks = retrieve_relevant_chunks(transcription, text_chunks, faiss_index, embedding_model)
            context = " ".join(relevant_chunks)
            st.subheader("Relevant Context from PDF")
            st.write(context)

        # Display Response from LLM
        with st.spinner("Generating response from model..."):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "assistant", "content": context},
                        {"role": "user", "content": transcription},
                    ],
                    model="llama3-8b-8192",
                    max_tokens=512,
                )
                st.subheader("Response from Model")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Model query error: {e}")
else:
    st.warning("Please upload both an audio file and a PDF document to proceed.")
