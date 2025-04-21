import os
import json
import requests
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# API Keys (Stored securely in .env file)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize Flask
app = Flask(__name__)

# Load Pretrained Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Affirmation Data (Can be replaced with a database)
with open("affirmations.json", "r") as f:
    affirmations_data = json.load(f)

# Debugging: Check the data structure
print(f"Affirmations Data Loaded: {len(affirmations_data)} entries")

# Prepare FAISS Index for Efficient Retrieval
dimension = 384  # Embedding size of MiniLM
index = faiss.IndexFlatL2(dimension)

# Ensure 'Affirmation' key exists to prevent KeyErrors
affirmation_texts = [item.get("Affirmation", "No text found") for item in affirmations_data]
affirmation_embeddings = embedding_model.encode(affirmation_texts)

# Add embeddings to FAISS index
index.add(np.array(affirmation_embeddings, dtype=np.float32))


# Function: Retrieve Closest Affirmation (Semantic Search)
def retrieve_affirmation(goal):
    print(f"Searching for affirmations related to: {goal}")  # Debugging info

    # Convert user goal into embedding
    goal_embedding = embedding_model.encode([goal])[0]

    # Search in FAISS index
    _, best_match_index = index.search(np.array([goal_embedding], dtype=np.float32), 1)
    best_affirmation = affirmation_texts[best_match_index[0][0]]

    print(f"Best Match Found: {best_affirmation}")  # Debugging info
    return best_affirmation


# Function: Generate Affirmation using AI (if no match found)
def generate_affirmation(goal):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
        json={"model": "gpt-4", "messages": [{"role": "system", "content": f"Generate a positive affirmation for: {goal}"}]}
    )

    if response.status_code == 200:
        affirmation = response.json()["choices"][0]["message"]["content"]
        print(f"Generated Affirmation: {affirmation}")
        return affirmation
    else:
        return "Affirmation generation failed."


# Function: Generate Image using Stable Diffusion
def generate_image(goal):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": f"Inspirational image about {goal}."})

    if response.status_code == 200:
        image_path = "static/output.png"
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path
    return "Image generation failed"


# Function: Generate Audio using ElevenLabs
def generate_audio(affirmation):
    if not ELEVENLABS_API_KEY or not affirmation:
        return None

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": affirmation,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        },
        "voice_id": "Bella"
    }

    response = requests.post("https://api.elevenlabs.io/v1/text-to-speech", headers=headers, json=payload)

    if response.status_code == 200:
        audio_data = response.content
        audio_path = "static/output.mp3"
        os.makedirs("static", exist_ok=True)

        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        return audio_path

    return None


# Route: Serve HTML Frontend
@app.route("/")
def home():
    return render_template("index.html")


# Route: Generate Affirmation + Image + Audio
@app.route("/generate", methods=["POST"])
def generate_content():
    data = request.json
    goal = data.get("goal", "")

    if not goal:
        return jsonify({"error": "Goal is required"}), 400

    affirmation = retrieve_affirmation(goal)
    
    # If FAISS does not return a valid result, generate a new one
    if affirmation == "No text found":
        affirmation = generate_affirmation(goal)

    image_url = generate_image(goal)
    audio_url = generate_audio(affirmation)
    
    return jsonify({
        "affirmation": affirmation,
        "image_url": image_url,
        "audio_url": audio_url
    })


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
