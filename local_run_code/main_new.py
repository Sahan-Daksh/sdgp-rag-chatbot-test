import os
import streamlit as st
import librosa
import soundfile as sf
from sentence_transformers import SentenceTransformer
import faiss
from whisper import load_model
from groq import Groq
import PyPDF2

# Constants for prebuilt assets
BUILT_IN_AUDIO = "./aud3.mp3"
BUILT_IN_PDF = "./Bash.pdf"

# Streamlit app setup
st.title("Smart Assistant: Voice & Text-Based Queries")
st.sidebar.title("Options")

# API Key for Groq
GROQ_API_KEY = "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"      # Replace with your Groq API key
client = Groq(api_key=GROQ_API_KEY)

# Add a toggle switch to choose between Voice and Text mode
query_mode = st.sidebar.radio("Select Query Mode:", ["Voice Input", "Text Input"])

# Cache PDF preprocessing (done once)
@st.cache_data
def preprocess_pdf_once(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return [], None, None

    # Build index and embeddings
    text_chunks = [text[i:i + 300] for i in range(0, len(text), 300)]  # Split into chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return text_chunks, index, model

# Audio preprocessing for Whisper
def preprocess_audio(input_path, target_sample_rate=16000):
    try:
        y, original_sr = librosa.load(input_path, sr=None)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        y_resampled = librosa.resample(y_trimmed, orig_sr=original_sr, target_sr=target_sample_rate)
        y_normalized = librosa.util.normalize(y_resampled)
        temp_output = input_path.replace(".mp3", "_preprocessed.wav")
        sf.write(temp_output, y_normalized, target_sample_rate)
        return temp_output
    except Exception as e:
        st.error(f"Audio preprocessing error: {e}")
        return None

# Query handler using Groq with Llama
def query_handler_groq(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.5,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying Groq Llama model: {e}")
        return None

# Load Whisper model
def load_whisper_model():
    try:
        model = load_model("base")  # Load the base Whisper model
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

whisper_model = load_whisper_model()

# PDF preprocessing (done once)
if query_mode == "Text Input":
    st.info("Text mode selected. Using built-in PDF.")
    with st.spinner("Loading and preprocessing PDF..."):
        text_chunks, faiss_index, embed_model = preprocess_pdf_once(BUILT_IN_PDF)
        st.success("PDF preprocessed successfully!")

# Voice input mode
if query_mode == "Voice Input":
    st.info("Voice mode selected. Using built-in audio.")
    st.audio(BUILT_IN_AUDIO, format="audio/mp3")
    if st.button("Transcribe and Query"):
        with st.spinner("Processing audio..."):
            preprocessed_audio = preprocess_audio(BUILT_IN_AUDIO)
            if preprocessed_audio:
                try:
                    # Use Whisper for transcription
                    result = whisper_model.transcribe(preprocessed_audio)
                    transcription_text = result.get("text", "No transcription available")
                    st.success("Audio transcribed successfully!")
                    st.text("Transcription:")
                    st.write(transcription_text)

                    # Input transcription text as a query
                    query = st.text_input("Ask a question based on the transcription:")
                    if st.button("Query with Transcription"):
                        with st.spinner("Querying AI..."):
                            response = query_handler_groq(query, context=transcription_text)
                            st.write("Assistant Response:")
                            st.write(response)
                except Exception as e:
                    st.error(f"Error during audio transcription or querying: {e}")

                # Cleanup temp files
                os.remove(preprocessed_audio)

# Text input query
if query_mode == "Text Input":
    user_query = st.text_input("Enter your query:")
    if st.button("Query"):
        if user_query:
            with st.spinner("Retrieving and responding..."):
                # Retrieve relevant chunks using FAISS
                query_embedding = embed_model.encode([user_query])
                distances, indices = faiss_index.search(query_embedding, k=3)
                retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
                context = " ".join(retrieved_chunks)

                # Query Llama with context
                response = query_handler_groq(user_query, context)
                st.write("Assistant Response:")
                st.write(response)
