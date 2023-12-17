from flask import Flask, render_template, request
import xgboost as xgb
import joblib
import os

app = Flask(__name__)

# Load your XGBoost model
model = joblib.load('model/your_xgboost_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    # Replace the following line with your actual prediction logic

    # END of the business logic here
   

    return f"{file_path} is {prediction_result}"

if __name__ == '__main__':
    app.run(debug=True, port=5050)