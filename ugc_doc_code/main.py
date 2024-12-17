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

# Load models
@st.cache_resource
def load_models():
    whisper_model = load_model("base")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    spacy_model = spacy.load("en_core_web_sm")
    return whisper_model, embed_model, spacy_model

whisper_model, embed_model, spacy_model = load_models()

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chunking and FAISS Indexing
def preprocess_with_metadata(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks

def generate_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def load_or_create_index():
    if os.path.exists(PROCESSED_DATA):
        with open(PROCESSED_DATA, "r", encoding="utf-8") as file:
            data = json.load(file)
        chunks = data["chunks"]
        embeddings = np.array(data["embeddings"])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return chunks, index
    else:
        raw_text = extract_text_from_pdf(PDF_PATH)
        chunks = preprocess_with_metadata(raw_text)
        index, embeddings = generate_faiss_index(chunks)
        with open(PROCESSED_DATA, "w", encoding="utf-8") as file:
            json.dump({"chunks": chunks, "embeddings": embeddings.tolist()}, file)
        return chunks, index

def retrieve_relevant_chunks(query, chunks, index, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def query_groq_llm(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model="llama3-8b-8192", messages=messages, temperature=0.5, max_tokens=512
    )
    return response.choices[0].message.content

# Load FAISS index
chunks, faiss_index = load_or_create_index()

# Streamlit UI
st.title("University Admission Assistant")
st.sidebar.title("Query Options")
mode = st.sidebar.radio("Select Mode", ["Text Input", "Voice Input", "Selection Mode"])

# Text Input Mode
if mode == "Text Input":
    st.header("Text Query")
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if query:
            with st.spinner("Retrieving content..."):
                relevant_chunks = retrieve_relevant_chunks(query, chunks, faiss_index)
                context = "\n".join(relevant_chunks)
                response = query_groq_llm(query, context)
            st.subheader("Assistant Response")
            st.write(response)

# Voice Input Mode
elif mode == "Voice Input":
    st.header("Voice Query")
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name
        st.audio(audio_path)
        with st.spinner("Transcribing audio..."):
            transcription = whisper_model.transcribe(audio_path)["text"]
        os.remove(audio_path)
        st.success("Transcription:")
        st.write(transcription)
        if st.button("Submit Query"):
            with st.spinner("Retrieving content..."):
                relevant_chunks = retrieve_relevant_chunks(transcription, chunks, faiss_index)
                context = "\n".join(relevant_chunks)
                response = query_groq_llm(transcription, context)
            st.subheader("Assistant Response")
            st.write(response)

# Selection Mode (Form-based Query)
elif mode == "Selection Mode":
    st.header("Selection-Based Query")
    z_score = st.text_input("Enter your Z-Score:")
    subject_stream = st.selectbox("Select your subject stream:", ["Science", "Commerce", "Arts"])
    st.subheader("G.C.E. (A/L) Subjects and Grades")
    subjects_grades = []
    for i in range(3):  # Collect 3 subjects
        subject = st.text_input(f"Subject {i+1}", key=f"subject_{i}")
        grade = st.selectbox(f"Grade for Subject {i+1}", ["A", "B", "C", "S", "F"], key=f"grade_{i}")
        if subject:  # Only append if subject is entered
            subjects_grades.append(f"{subject} - {grade}")
    preferred_courses = st.text_area("Preferred Courses of Study (comma-separated):")
    district = st.text_input("Enter your district:")

    if st.button("Submit Form"):
        # Generate query from form inputs
        query = f"""
        Find details about university admissions for a candidate with the following criteria:

        Z-Score: {z_score},  
        G.C.E. (Advanced Level) Subjects and Grades:
        {', '.join(subjects_grades)},
        Preferred Courses of Study: {preferred_courses},  
        District: {district}.  

        List the eligible courses of study, minimum entry requirements, applicable cut-off marks, and any district quota information for the specified criteria.
        """
        # st.subheader("Generated Query")
        # st.write(query)

        with st.spinner("Retrieving content..."):
            relevant_chunks = retrieve_relevant_chunks(query, chunks, faiss_index)
            context = "\n".join(relevant_chunks)
            response = query_groq_llm(query, context)
        st.subheader("Assistant Response")
        st.write(response)
