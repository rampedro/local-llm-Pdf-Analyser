import ollama
import base64

def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

def analyze_text(text):
    response = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": f"Summarize and categorize:\n{text}"}
    ])
    return response["message"]["content"]

def analyze_image(path):
    image_b64 = encode_image(path)
    response = ollama.chat(model="llava:latest", messages=[
        {"role": "user", "content": "Describe this image."},
        {"role": "user", "content": {"type": "image", "image": image_b64}}
    ])
    return response["message"]["content"]
