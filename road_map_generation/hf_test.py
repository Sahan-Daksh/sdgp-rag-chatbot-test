from huggingface_hub import InferenceClient

# API Key for Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

# output is a PIL.Image object
image = client.text_to_image(
	"Astronaut riding a horse",
	model="black-forest-labs/FLUX.1-dev"
)

# Optionally save or display the image
image.save("output.png")