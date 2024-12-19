import requests

# Replace <ngrok_url> with the Ngrok URL
url = "https://4779-34-16-168-159.ngrok-free.app/generate"
prompt = {"prompt": "A futuristic city skyline with flying cars"}

response = requests.post(url, json=prompt)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Image saved as output.png")
else:
    print("Error:", response.json())