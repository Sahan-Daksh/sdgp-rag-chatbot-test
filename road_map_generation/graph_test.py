import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Replace <ngrok_url> with the actual Ngrok URL from Colab
NGROK_URL = "https://2187-34-16-168-159.ngrok-free.app/"

st.title("Dream Career Roadmap Generator")
st.write("Enter your dream career, and we will generate a roadmap from zero to hero!")

# Input field for dream career
dream_career = st.text_input("Your Dream Career", "")

if st.button("Generate Roadmap"):
    if dream_career:
        with st.spinner("Generating your career roadmap..."):
            try:
                # Send a POST request to the Colab server
                response = requests.post(f"{NGROK_URL}/generate", json={"dream_career": dream_career})

                if response.status_code == 200:
                    # Load and display the image
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=f"Roadmap for: {dream_career}", use_column_width=True)

                    # Save the image locally (optional)
                    image.save("generated_roadmap.png")
                    st.success("Image saved as generated_roadmap.png")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a dream career!")
