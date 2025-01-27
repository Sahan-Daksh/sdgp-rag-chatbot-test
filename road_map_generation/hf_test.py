from huggingface_hub import InferenceClient

client = InferenceClient(
	api_key="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"
)

# output is a PIL.Image object
image = client.text_to_image(
	"Astronaut riding a horse",
	model="black-forest-labs/FLUX.1-dev"
)

# Optionally save or display the image
image.save("output.png")