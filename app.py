from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------- DATASET ----------------
GROUPDATASET = {
    # Aaron Flores
    "tell me about aaron": "Aaron Flores is a Desktop Support Specialist with a background in industrial mechanics.",
    "who is aaron flores": "Aaron Flores is a Desktop Support Specialist at Color Communications, LLC.",
    "what does aaron flores do": "He provides IT support and troubleshooting services.",
    "where does aaron flores work": "He works at Color Communications, LLC.",
    "aaron flores education": "Aaron Flores is pursuing a Bachelor's degree in Computer Science at Northeastern Illinois University.",
    "aaron flores skills": "IT support, troubleshooting, and hardware installation.",

    # Rubi Shrestha
    "tell me about rubi": "Rubi Shrestha is a Computer Science student at Northeastern Illinois University.",
    "who is rubi shrestha": "Rubi Shrestha is a CS student and Shift Manager at McDonald's.",
    "what does rubi do": "She works as a Shift Manager at McDonald's.",
    "rubi education": "She is pursuing a Bachelor's degree in Computer Science.",
    "rubi skills": "Leadership, teamwork, and customer service.",

    # Ezza May De La Cruz
    "tell me about ezza": "Ezza May De La Cruz is a Freight Forwarding Analyst at UPS Supply Chain Solutions.",
    "who is ezza": "Ezza works in supply chain analytics and operations.",
    "ezza skills": "Python, SQL, automation, and data analysis.",
    "ezza experience": "She has experience in logistics, freight forwarding, and aviation.",

    # Rasheed Johnson
    "tell me about rasheed": "Rasheed Johnson is a Journeyman Electrician at Amtrak.",
    "who is rasheed": "Rasheed is an electrician and serves in the Army National Guard.",
    "rasheed skills": "Electrical systems, troubleshooting, and logistics.",
    "rasheed experience": "Over 10 years in technical operations and leadership.",

    # Albion Kita
    "tell me about albion": "Albion Kita is a Computer Science student at Northeastern Illinois University.",
    "who is albion": "Albion is an aspiring software developer.",
    "albion skills": "Python, Java, C++, JavaScript, and web development.",
    "albion interests": "Coding and building software projects."
}

# ---------------- RETRIEVAL FUNCTION ----------------
def get_best_response(user_input):
    clean = user_input.lower().strip()

    # Edge case: empty input
    if clean == "":
        return "A9 here 👋 Please type a question so I can help!"

    scores = []

    # simple keyword scoring (RAG-style retrieval)
    for key, value in GROUPDATASET.items():
        match_score = sum(word in key for word in clean.split())
        if match_score > 0:
            scores.append((match_score, value))

    # return best match
    if scores:
        scores.sort(reverse=True)
        return scores[0][1]

    # fallback
    return "A9 here 👋 I don’t have info on that yet. Try asking about Aaron, Rubi, Ezza, Rasheed, or Albion!"

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    bot_reply = get_best_response(user_message)
    return jsonify({"response": bot_reply})

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
