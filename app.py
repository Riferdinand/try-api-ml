from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import numpy as np
import pickle
import logging
import os
import requests

app = Flask(__name__)
CORS(app)

# Define the paths for the model and tokenizer
model_url = 'https://storage.googleapis.com/ml-model-try/model.h5'
tokenizer_url = 'https://storage.googleapis.com/ml-model-try/tokenizer.pkl'

# Download the model and tokenizer files
model_path = 'model.h5'
tokenizer_path = 'tokenizer.pkl'

if not os.path.exists(model_path):
    model_content = requests.get(model_url).content
    with open(model_path, 'wb') as model_file:
        model_file.write(model_content)

if not os.path.exists(tokenizer_path):
    tokenizer_content = requests.get(tokenizer_url).content
    with open(tokenizer_path, 'wb') as tokenizer_file:
        tokenizer_file.write(tokenizer_content)

# Load model at the beginning of the application
model = load_model(model_path)

# Add softmax layer
num_classes = 3
model.add(Dense(num_classes, activation='softmax', name='output'))

# Load the saved tokenizer
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def response():
    return 'Response Success!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validation input
        data = request.get_json(force=True)
        if 'text' not in data or not isinstance(data['text'], str):
            return jsonify({'error': "'text' key not found or not a valid string in data"})
        user_text = data['text']

        # Pre-process the input data
        sequences = tokenizer.texts_to_sequences([user_text])
        padded_data = pad_sequences(sequences, maxlen=250, padding='post', truncating='post')

        # Make predictions using the loaded model
        predictions = model.predict(padded_data)

        # Define labels here
        labels = ["Anxiety", "Depression", "Stress"]

        # Process predictions and return the results
        results = []
        result = {
            'text': user_text,
            'predictions': [
                {'class': labels[j], 'probability': float(predictions[0][j])}
                for j in range(len(predictions[0]))
            ],
            'predicted_class': labels[np.argmax(predictions[0])]
        }
        results.append(result)

        # Log request and response
        logging.info('Received request: %s', data)
        logging.info('Sending response: %s', jsonify(results).json)

        return jsonify(results)

    except Exception as e:
        logging.error('Error: %s', str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
