from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # call this API

# Load your trained model and scaler
model = joblib.load("models/improved_rf_country.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

@app.route('/predict-region', methods=['POST'])
def predict_region():
    data = request.json
    scores = data.get("scores")

    input_order = ['O score', 'C score', 'E score', 'A score', 'N score']
    input_vector = np.array([
        scores['OPN'],
        scores['CSN'],
        scores['EXT'],
        scores['AGR'],
        scores['EST']
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_vector)
    prediction = model.predict(input_scaled)[0]
    region = label_encoder.inverse_transform([prediction])[0]

    return jsonify({"region": region})

if __name__ == '__main__':
    app.run(debug=True)
