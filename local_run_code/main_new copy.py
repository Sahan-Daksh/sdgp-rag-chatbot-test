
#Code 1
import os
import librosa
import soundfile as sf
import numpy as np

def preprocess_audio(input_path, output_path=None, target_sample_rate=16000):
    """
    Preprocess audio for transcription

    Args:
    - input_path: Path to input audio file
    - output_path: Path to save processed audio (optional)
    - target_sample_rate: Desired sample rate for transcription

    Returns:
    - Processed audio path
    """
    try:
        # Load audio file
        y, original_sr = librosa.load(input_path, sr=None)

        # Convert to mono if stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Trim silence from beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Resample to target sample rate
        y_resampled = librosa.resample(
            y_trimmed,
            orig_sr=original_sr,
            target_sr=target_sample_rate
        )

        # Normalize audio
        y_normalized = librosa.util.normalize(y_resampled)

        # Save preprocessed audio if output path provided
        if output_path:
            sf.write(output_path, y_normalized, target_sample_rate)
            return output_path

        # If no output path, create a temp preprocessed file
        temp_output = input_path.replace('.wav', '_preprocessed.wav')
        sf.write(temp_output, y_normalized, target_sample_rate)

        return temp_output

    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return input_path  # Fallback to original file if preprocessing fails

# Modify your existing script
import os
import sys
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="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"
)

# Specify the path to the audio file
filename = "./aud3.mp3"

# Preprocess the audio
preprocessed_filename = preprocess_audio(filename)

# Open the preprocessed audio file
with open(preprocessed_filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=(preprocessed_filename, file.read()), # Required audio file
      model="whisper-large-v3-turbo", # Required model to use for transcription
      prompt="Specify context or spelling",  # Optional
      response_format="json",  # Optional
      language="en",  # Optional
      temperature=0.0  # Optional
    )
    # Print the transcription text
    varr =transcription.text
    print(transcription.text)

# Optional: Clean up preprocessed file
if preprocessed_filename != filename:
    os.remove(preprocessed_filename)




#Code 2


#code 2
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from groq import Groq

# OpenAI API Key (Replace with your API key)
openai.api_key = "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"

# Initialize Groq client
client = Groq(api_key="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")

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 3: Generate embeddings for chunks
def generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    return embeddings

# Step 4: Store embeddings in FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    index.add(embeddings)
    return index

# Step 5: Retrieve relevant chunks
def retrieve_relevant_chunks(query, text_chunks, index, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 matches
    return [text_chunks[idx] for idx in indices[0]]

# Step 6: Query the language model
def query_llm(client, query, context):
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

# Main Code
pdf_path = "./Bash.pdf"

# Step 1: Extract text
document_text = extract_text_from_pdf(pdf_path)

# Step 2: Split into chunks
text_chunks = split_text_into_chunks(document_text)

# Step 3: Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2")

# Step 4: Create FAISS index
faiss_index = create_faiss_index(np.array(embeddings))

# Step 5: User Query
user_query = varr

# Step 6: Retrieve relevant chunks
relevant_chunks = retrieve_relevant_chunks(user_query, text_chunks, faiss_index, embedding_model)

# Combine chunks into context
context = " ".join(relevant_chunks)

# Step 7: Query the LLM
response = query_llm(client, user_query, context)

print("\nResponse from the model:")
print(response)
