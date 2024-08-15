from flask import Flask, request, jsonify
from flask import Flask
from flask_cors import CORS
import pickle
import joblib

model = joblib.load('model.joblib')

# Load the model
# with open('model.pkl', 'rb') as file:
#    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from POST request
    prediction = model.predict([data['features']])  # Predict using the model

    result = 'Fully Paid' if prediction == 1 else 'Charged Off'
    return jsonify({'prediction': result}) # Return the prediction

if __name__ == '__main__':
    app.run(debug=True)