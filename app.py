from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

model = joblib.load("models/random_forest_model.joblib")
scaler = joblib.load("models/scaler.joblib")

@app.route("/predict-cluster", methods=["POST"])
def predict_cluster():
    print("Received POST /predict-cluster")
    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        cluster = model.predict(features_scaled)[0]

        print("Predicted cluster:", cluster)
        return jsonify({"cluster": int(cluster)})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Flask is working"

if __name__ == "__main__":
    print("Flask is starting...")
    app.run(port=5000)
