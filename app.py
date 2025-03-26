from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load all models
try:
    slr_model = joblib.load('models/slr_model.joblib')
    mlr_model = joblib.load('models/mlr_model.joblib')
    poly_model, poly_reg = joblib.load('models/poly_model.joblib')
    logistic_model = joblib.load('models/logistic_model.joblib')
    knn_model = joblib.load('models/knn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
except:
    print("Please run train_models.py first to train and save the models!")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = float(data['age'])
        experience = float(data['experience'])
        education_level = float(data['education_level'])
        model_type = data['model_type']

        # Prepare input data
        input_data = np.array([[age, experience, education_level]])
        input_scaled = scaler.transform(input_data)

        if model_type == 'slr':
            # Simple Linear Regression (using only experience)
            prediction = slr_model.predict([[experience]])[0]
        elif model_type == 'mlr':
            # Multiple Linear Regression
            prediction = mlr_model.predict(input_scaled)[0]
        elif model_type == 'poly':
            # Polynomial Regression
            input_poly = poly_model.transform(input_scaled)
            prediction = poly_reg.predict(input_poly)[0]
        elif model_type == 'logistic':
            # Logistic Regression
            prediction = logistic_model.predict(input_scaled)[0]
        elif model_type == 'knn':
            # KNN
            prediction = knn_model.predict(input_scaled)[0]
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        return jsonify({
            'prediction': float(prediction),
            'model_type': model_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 