from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import random
import numpy as np
import re
import nltk
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# Load data
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.load(open("Chatbot\Backend\static\intents.json"))
model = tf.keras.models.load_model("chatbot_model.keras")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.lower()

def chatbot_response(text):
    cleaned_text = clean_text(text)
    input_bag = [0] * len(words)
    input_words = nltk.word_tokenize(cleaned_text)
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
    for word in input_words:
        if word in words:
            input_bag[words.index(word)] = 1

    prediction = model.predict(np.array([input_bag]))[0]
    tag = classes[np.argmax(prediction)]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand..."

@app.route("/")
def index():
    return render_template("index.html")  # Serve frontend

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json("Chatbot\Backend\static\intents.json")
    user_message = data.get("message")
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
