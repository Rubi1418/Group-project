from flask import Flask, request, jsonify, render_template
from difflib import get_close_matches
import json

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

    matches = get_close_matches(user_input, all_questions, n=3, cutoff=0.45)

    if matches:
        for item in data:
            for q in item["questions"]:
                if q in matches:
                    return item["answer"]

    return "I’m not sure I understood that. Try asking something else."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        response = get_response(user_message)
        return jsonify({"response": response})
    except:
        return jsonify({"response": "Something went wrong."})

if __name__ == "__main__":
    app.run(debug=True)
