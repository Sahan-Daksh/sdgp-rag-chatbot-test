import streamlit as st
from huggingface_hub import InferenceClient
from groq import Groq

# API setup
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(api_key="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")

def get_roadmap_prompt(career):
    messages = [
        {"role": "system", "content": "You are a career guidance expert. Create detailed visual roadmap descriptions."},
        {"role": "user", "content": f"Create a detailed prompt to generate a visual career roadmap for {career}. Include education path, key skills, certifications, and career progression milestones. Format it as a journey-based visualization."}
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