from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from flasgger import Swagger
import logging
try:
    from LocalModel import HybridNewsDetector
    print("HybridNewsDetector imported Successfully!")
except ImportError as e:
    print(f"Error: gp.py nahi mili ya usme error hai toh error show hoga : {e}")

app = Flask(__name__)
CORS(app)
swagger=Swagger(app)


MODEL_PATH = "./fake_news_model_20251124_091015/final_model"
OPENAI_KEY = ""
detector = None

def init_model():
    global detector
    if os.path.exists(MODEL_PATH):
        print(" Loading AI Models (DistilBERT + GPT)...")
        # Ensure gpt-4o-mini is set inside HybridNewsDetector for efficiency
        detector = HybridNewsDetector(distilbert_model_path=MODEL_PATH, openai_api_key=OPENAI_KEY)
        print(" Spot the Lie Engine: READY for Prediction")
    else:
        print(f"Alert: {MODEL_PATH}")

@app.route('/')
def home():
    return {"message": " API is running"}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Fake News Detection Endpoint
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "This is a sample news headline to check."
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input
      500:
        description: Model not initialized or Server Error
    """
    if detector is None:
        return jsonify({"error": "It is not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Please send 'text' in JSON"}), 400
        
        
        result = detector.predict_news(data['text'])
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    
    app.run(debug=True, port=5000,use_reloader=False)