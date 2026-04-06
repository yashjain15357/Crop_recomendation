import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, render_template, jsonify

# Load the trained model and label encoder
xgboost_model = pickle.load(open("model_file/xgboost_crop_model.pkl", 'rb'))
label_encoder = pickle.load(open("model_file/label_encoder.pkl", 'rb'))

app = Flask(__name__)



def predict_crop(input_data):
    """Predict crop name from ordered input dict."""
    new_data = np.array([[ 
        input_data['N'],
        input_data['P'],
        input_data['K'],
        input_data['temperature'],
        input_data['humidity'],
        input_data['ph'],
        input_data['rainfall']
    ]])

    prediction = xgboost_model.predict(new_data)
    crop_name = label_encoder.inverse_transform(prediction)[0]
    return crop_name


@application.route("/", methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        # Get input values from the form (7 features for crop recommendation)
        N = float(request.form.get('N'))
        P = float(request.form.get('P'))
        K = float(request.form.get('K'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('ph'))
        rainfall = float(request.form.get('rainfall'))

        input_data = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }

        crop_name = predict_crop(input_data)
        
        return render_template('home.html', results=crop_name)
    else:
        return render_template('home.html')


@application.route("/api/predict", methods=['GET', 'POST'])
def predict_api():
    """JSON API: takes input features and returns input + predicted crop."""
    
    # Handle POST request - make prediction
    try:
        data = request.get_json(force=True)

        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f"Missing fields: {', '.join(missing_fields)}"
            }), 400

        input_data = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }

        crop_name = predict_crop(input_data)

        return jsonify({
            'status': 'success',
            'input': input_data,
            'prediction': {
                'recommended_crop': crop_name
            }
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        

if __name__ == "__main__":
    application.run(debug=True)