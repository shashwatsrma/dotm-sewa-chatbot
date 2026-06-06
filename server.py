from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import get_answer

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "").strip()
    
    if not question:
        return jsonify({"reply": "Please ask a question."})
    
    answer = get_answer(question)
    return jsonify({"reply": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)