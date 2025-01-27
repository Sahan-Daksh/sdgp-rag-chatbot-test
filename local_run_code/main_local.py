import os
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
from whisper import load_model
from groq import Groq
import PyPDF2

# Constants for prebuilt assets
BUILT_IN_PDF = "./Bash.pdf"

# Streamlit app setup
st.title("Smart Assistant: Voice & Text-Based Queries")
st.sidebar.title("Options")

# API Key for Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

# Add a toggle switch to choose between Voice and Text mode
query_mode = st.sidebar.radio("Select Query Mode:", ["Voice Input", "Text Input"])

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    try:
        model = load_model("base")  # Load the base Whisper model
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

whisper_model = load_whisper_model()

# Load embedding model
@st.cache_resource
def load_embed_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Load the sentence transformer model
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

embed_model = load_embed_model()  # Load model for embedding text

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
    embeddings = embed_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return text_chunks, index

# Preprocess the built-in PDF (done for both Text and Voice modes)
text_chunks, faiss_index = preprocess_pdf_once(BUILT_IN_PDF)
  # Now available globally

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

# Voice input mode
if query_mode == "Voice Input":
    # Audio File Input
    st.header("Upload an Audio File")
    audio_file = st.file_uploader("Upload an audio file (WAV, MP3, etc.)", type=["wav", "mp3"])

    if audio_file:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name
        
        st.audio(audio_path, format="audio/wav")
        
        # Transcribe audio using Whisper
        with st.spinner("Processing audio and transcribing..."):
            try:
                result = whisper_model.transcribe(audio_path)
                transcription_text = result['text'] if 'text' in result else "No transcription available"
                st.success("Transcription completed!")
                st.text("Transcription:")
                st.write(transcription_text)

                # Use transcription as a query
                with st.spinner("Retrieving context and querying assistant..."):
                    # Create the embedding of the transcribed text to use with FAISS
                    query_embedding = embed_model.encode([transcription_text])
                    distances, indices = faiss_index.search(query_embedding, k=3)  # Adjust k as necessary
                    retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
                    context = " ".join(retrieved_chunks)

                    # Query with context
                    response = query_handler_groq(transcription_text, context)  # Correct function
                    st.header("Assistant Response")
                    st.write(response)

            except Exception as e:
                st.error(f"Error during transcription or query: {e}")
            finally:
                # Cleanup temporary file
                os.remove(audio_path)

# Text input mode
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
