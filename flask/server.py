from flask import Flask, render_template, request
import xgboost as xgb
import joblib
import os
import trafilatura as trf
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load your XGBoost model
model = joblib.load('/model/xgboost_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')
    prediction_result = "Phishing" # or it will be "Legitimate"
    if 'htmlFile' not in request.files:
        return "No file part"
    
    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location
    file_path = 'test/' + file.filename
    file.save(file_path)

    # START of the business logic here

    # Perform prediction using the file_path with your XGBoost model

    # END of the business logic here
    try:
        # Parse HTML content with trafilatura
        html_content = trf.extract(file_path)

        # Get embeddings of the parsed HTML content
        embeddings = model.encode(html_content)


        prediction = model.predict(embeddings)
        if prediction == 0:
            prediction_result = "Legitimate"
        elif prediction == 1:
            prediction_result = "Phishing"

    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during prediction"

    return f"{file_path} is {prediction_result}"

if __name__ == '__main__':
    app.run(debug=True, port=5050)
