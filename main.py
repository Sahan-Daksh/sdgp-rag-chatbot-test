# import os
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
# import whisper
# import tempfile
# from pydub import AudioSegment
# from gtts import gTTS
# import torchaudio
from groq import Groq

# Securely manage API keys
openai.api_key = "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"
client = Groq(api_key="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")

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error("Error extracting text from PDF: " + str(e))
        return ""

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 3: Generate embeddings for chunks
def generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text_chunks)

# Step 4: Store embeddings in FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Retrieve relevant chunks
def retrieve_relevant_chunks(query, text_chunks, index, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    return [text_chunks[idx] for idx in indices[0]]

# Step 6: Query the language model
def query_llm(query, context):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=512,
            top_p=1,
            stop=None,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error("Error querying the language model: " + str(e))
        return ""

# Audio transcription logic
# def transcribe_audio(audio_file):
#     try:
#         model = whisper.load_model("base")
#         transcription = model.transcribe(audio_file)
#         return transcription["text"]
#     except Exception as e:
#         st.error("Error transcribing audio: " + str(e))
#         return ""

# Generate speech from text using uploaded voice clip
# def generate_speech_from_text(text, uploaded_voice_clip):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_clip:
#             temp_clip.write(uploaded_voice_clip.read())
#             temp_clip_path = temp_clip.name

#         voice_clip = AudioSegment.from_file(temp_clip_path)
#         tts = gTTS(text=text, lang="en")
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
#             tts.save(temp_tts.name)
#             tts_audio = AudioSegment.from_file(temp_tts.name)

#         # Combine TTS with the voice clip properties
#         final_audio = tts_audio.set_frame_rate(voice_clip.frame_rate).set_channels(voice_clip.channels)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as final_output:
#             final_audio.export(final_output.name, format="mp3")
#             return final_output.name
#     except Exception as e:
#         st.error("Error generating speech: " + str(e))
#         return ""

# Main Streamlit app
def main():
    # Page configuration
    st.set_page_config(page_title="AI Assistant with RAG and Voice", layout="wide")

    # Sidebar for interaction mode
    st.sidebar.title("Interaction Mode")
    interaction_mode = st.sidebar.radio("Choose interaction mode:", ("Text", "Voice"))

    # File upload for PDF
    st.subheader("Upload PDF for Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # File upload for voice clip
    st.subheader("Upload Voice Clip for Narration")
    uploaded_voice_clip = st.file_uploader("Upload a voice clip", type=["wav", "mp3"])

    if uploaded_pdf and uploaded_voice_clip:
        document_text = extract_text_from_pdf(uploaded_pdf)

        if document_text:
            text_chunks = split_text_into_chunks(document_text)

            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2")
            faiss_index = create_faiss_index(np.array(embeddings))
            st.success("PDF processed and knowledge base created!")

            if interaction_mode == "Text":
                st.subheader("Text Interaction")
                user_query = st.text_input("Enter your query:")

                if st.button("Ask"):
                    if user_query.strip():
                        relevant_chunks = retrieve_relevant_chunks(user_query, text_chunks, faiss_index, embedding_model)
                        context = " ".join(relevant_chunks)
                        response = query_llm(user_query, context)

                        st.markdown(f"**Assistant:** {response}")

                        # Generate narrated response
                        # narrated_response_path = generate_speech_from_text(response, uploaded_voice_clip)
                        # if narrated_response_path:
                        #     st.audio(narrated_response_path, format="audio/mp3")
                    else:
                        st.warning("Please enter a query.")

            elif interaction_mode == "Voice":
                st.subheader("Voice Interaction")
                uploaded_audio = st.file_uploader("Upload your voice message", type=["wav", "mp3"])

                if uploaded_audio:
                    st.audio(uploaded_audio, format="audio/wav")
                    st.success("Voice message uploaded successfully!")

                    # user_query = transcribe_audio(uploaded_audio)
                    # if user_query:
                    #     relevant_chunks = retrieve_relevant_chunks(user_query, text_chunks, faiss_index, embedding_model)
                    #     context = " ".join(relevant_chunks)
                    #     response = query_llm(user_query, context)

                    #     st.markdown(f"**Assistant:** {response}")

                        # Generate narrated response
                        # narrated_response_path = generate_speech_from_text(response, uploaded_voice_clip)
                        # if narrated_response_path:
                        #     st.audio(narrated_response_path, format="audio/mp3")

if __name__ == "__main__":
    main()