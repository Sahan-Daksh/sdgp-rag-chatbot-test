import streamlit as st
from huggingface_hub import InferenceClient
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Secure API setup
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')

if not GROQ_API_KEY or not HF_API_KEY:
    st.error("Missing API keys. Please set GROQ_API_KEY and HF_API_KEY in .env file")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(api_key=HF_API_KEY)

def get_roadmap_prompt(career):
    messages = [
        {"role": "system", "content": """You are an expert career guidance counselor specialized in creating visual roadmap descriptions.
        Format your response as a clear step-by-step journey that can be visualized."""},
        {"role": "user", "content": f"""Create a detailed prompt for a visual roadmap showing the journey to become a {career}.
        Include:
        - Required education and degrees
        - Essential skills and certifications
        - Career progression steps
        - Major milestones
        Format as a journey flowing from left to right with clear progression points."""}
    ]
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content

st.title("Career Roadmap Generator")
dream_career = st.text_input("Enter your dream career:")

if st.button("Generate Roadmap"):
    if dream_career:
        with st.spinner("Generating your career roadmap..."):
            try:
                # Get detailed prompt from LLM
                detailed_prompt = get_roadmap_prompt(dream_career)
                st.write("Generated Prompt:", detailed_prompt)

                # Generate image using the detailed prompt
                image = hf_client.text_to_image(
                    detailed_prompt,
                    model="black-forest-labs/FLUX.1-dev"
                )

                # Display results
                st.image(image, caption=f"Roadmap for: {dream_career}", use_column_width=True)
                image.save("generated_roadmap.png")
                st.success("Roadmap generated and saved!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a dream career!")