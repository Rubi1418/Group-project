from flask import Flask, request, jsonify, send_from_directory
from difflib import get_close_matches
import json
import os

app = Flask(__name__)

with open("data.json", "r") as file:
    data = json.load(file)

def get_response(user_input):
    user_input = user_input.lower().strip()

    if user_input == "":
        return "Please type a question so I can help."

    all_questions = []

    for item in data:
        for question in item["questions"]:
            all_questions.append(question)

    match = get_close_matches(user_input, all_questions, n=1, cutoff=0.45)

    if match:
        matched_question = match[0]

        for item in data:
            if matched_question in item["questions"]:
                return item["answer"]

    return "I’m not sure I understood that. Try asking about Aaron, Rubi, or Ezza."

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/styles.css")
def css():
    return send_from_directory(".", "styles.css")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)