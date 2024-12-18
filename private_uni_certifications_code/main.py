import os
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate
from reportlab.lib import colors
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to convert CSVs to a single PDF
def csv_to_pdf_proper(csv_files, output_pdf):
    pdf = SimpleDocTemplate(output_pdf, pagesize=landscape(A4))
    elements = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="latin1")
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ])
        table.setStyle(style)
        elements.append(table)
        elements.append(Table([[" "]]))  # Spacer line

    pdf.build(elements)
    return output_pdf

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to generate embeddings
def generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text_chunks)

# Function to create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, text_chunks, index, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    return [text_chunks[idx] for idx in indices[0]]

# Streamlit UI
st.title("Document Query System")
st.sidebar.title("Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    st.sidebar.success(f"Uploaded {len(uploaded_files)} files.")
    output_pdf = "structured_output.pdf"
    st.sidebar.text("Generating PDF...")
    pdf_path = csv_to_pdf_proper(uploaded_files, output_pdf)
    st.sidebar.success("PDF Generated!")

    st.subheader("Extracted PDF Content")
    document_text = extract_text_from_pdf(pdf_path)
    st.text_area("PDF Text", document_text, height=200)

    st.subheader("Generate Embeddings and Query")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_chunks = split_text_into_chunks(document_text)
    embeddings = generate_embeddings(text_chunks)
    faiss_index = create_faiss_index(np.array(embeddings))

    query = st.text_input("Enter your query")
    if query:
        st.text("Searching relevant information...")
        relevant_chunks = retrieve_relevant_chunks(query, text_chunks, faiss_index, embedding_model)
        context = " ".join(relevant_chunks)
        st.text_area("Relevant Chunks", context, height=150)

        # Simulating LLM response for now
        response = f"Simulated response based on: {context}"
        st.text_area("Model Response", response, height=100)