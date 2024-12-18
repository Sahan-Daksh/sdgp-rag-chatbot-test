import streamlit as st
from groq import Groq
import tempfile

# API Key
GROQ_API_KEY = "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"  # Replace with your Groq API key
client = Groq(api_key=GROQ_API_KEY)

# Function to query Groq LLM for D3.js code
def generate_d3_code(dream_career):
    prompt = f"""
    Create a D3.js code snippet that generates a clear and professional roadmap graph for someone pursuing a career in {dream_career}.
    The roadmap should include multiple steps, milestones, and arrows showing dependencies. The code should work in a standard D3.js environment.
    """
    messages = [
        {"role": "system", "content": "You are a professional assistant skilled in D3.js code generation."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="llama3-8b-8192", messages=messages, temperature=0.5, max_tokens=1000
    )
    return response.choices[0].message.content

# Function to save D3.js code to an HTML file
def save_d3_html(d3_code):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>D3.js Roadmap</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
    </head>
    <body>
        <div id="roadmap"></div>
        <script>
        {d3_code}
        </script>
    </body>
    </html>
    """
    # Use a named temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file.write(html_template.encode("utf-8"))
    temp_file.flush()
    return temp_file.name

# Streamlit App
st.title("Career Roadmap Generator")

# User Input
dream_career = st.text_input("Enter your dream career:", placeholder="e.g., Data Scientist, AI Researcher")
submit_button = st.button("Generate Roadmap")

if submit_button:
    if dream_career.strip():
        with st.spinner("Generating roadmap..."):
            try:
                # Generate D3.js code
                d3_code = generate_d3_code(dream_career)

                # Save D3.js code to an HTML file
                html_path = save_d3_html(d3_code)

                # Display HTML file in Streamlit
                st.subheader("Career Roadmap")
                st.markdown("Below is the roadmap generated for your dream career:")
                st.components.v1.html(open(html_path).read(), height=600, scrolling=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please enter a valid dream career.")
