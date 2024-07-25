import os

import torch
from flask import Flask, jsonify, request
from transformers import BertForSequenceClassification, BertTokenizer

app = Flask(__name__)

# Load the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=28
)
model_save_path = "model"  # Update with your model path
classifier_weights = torch.load(
    os.path.join(model_save_path, "deployed_classifier_weights.pt"),
    map_location=torch.device("cpu"),
    weights_only=True,
)
model.classifier.load_state_dict(classifier_weights)
model.eval()  # Set the model to evaluation mode

# Define emotions dictionary
emotions_dict = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}


@app.route("/predict", methods=["POST"])
def predict_emotion():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess and tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to emotion
    predicted_emotion = emotions_dict[predicted_class]
    return jsonify({"text": text, "emotion": predicted_emotion})


if __name__ == "__main__":
    app.run(debug=True)
