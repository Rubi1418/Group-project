from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------- DATASET ----------------
GROUPDATASET = {
    "tell me about aaron": "Aaron is a Desktop Support Specialist with IT experience.",
    "tell me about rubi": "Rubi is a Computer Science student and shift manager at McDonald's.",
    "tell me about ezza": "Ezza works as an Analyst in Freight Forwarding at UPS.",
    "tell me about rasheed": "Rasheed is an electrician and Army National Guard member.",
    "tell me about albion": "Albion is a Computer Science student and aspiring developer."
}

# ---------------- CHAT LOGIC ----------------
def get_response(msg):
    msg = msg.lower().strip()

    if msg == "":
        return "Please type something."

    # smarter matching (keyword-based)
    if "aaron" in msg:
        return "Aaron Flores works as a Desktop Support Specialist at Color Communications."

    if "rubi" in msg:
        return "Rubi works as a shift manager at McDonald's."

    if "ezza" in msg:
        return "Ezza works in Freight Forwarding at UPS Supply Chain Solutions."

    if "rasheed" in msg:
        return "Rasheed is an electrician at Amtrak and serves in the Army National Guard."

    if "albion" in msg:
        return "Albion is a Computer Science student at NEIU."

    return "I don’t have that information yet. Try asking about Aaron, Rubi, Ezza, Rasheed, or Albion."
# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    response = get_response(user_message)
    return jsonify({"response": response})

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
