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


@app.route('/ping', methods=['GET'])
def ping():
    """Lightweight endpoint for keep-alive cron jobs / uptime monitors."""
    return jsonify({'status': 'alive'}), 200


@app.route("/", methods=['GET', 'POST'])
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

        crop_name = predict_crop(input_data).upper()
        
        return render_template('home.html', results=crop_name)
    else:
        return render_template('home.html')


@app.route("/api/predict", methods=['GET', 'POST'])
def predict_api():
    """JSON/API endpoint: accepts JSON/form/query input and returns recommended crop."""
    try:
        data = request.get_json(silent=True)

        if data is None:
            data = request.form.to_dict()
        if not data:
            data = request.args.to_dict()

        if not isinstance(data, dict):
            return jsonify({
                'status': 'error',
                'message': 'Request body must be a JSON object, form data, or query parameters'
            }), 400

        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data or data[field] in [None, '']]

        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f"Missing or empty fields: {', '.join(missing_fields)}"
            }), 400

        def to_float(value, field_name):
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Field '{field_name}' must be a valid number")

        input_data = {
            'N': to_float(data['N'], 'N'),
            'P': to_float(data['P'], 'P'),
            'K': to_float(data['K'], 'K'),
            'temperature': to_float(data['temperature'], 'temperature'),
            'humidity': to_float(data['humidity'], 'humidity'),
            'ph': to_float(data['ph'], 'ph'),
            'rainfall': to_float(data['rainfall'], 'rainfall')
        }

        crop_name = predict_crop(input_data)

        return jsonify({
            'status': 'success',
            'input': input_data,
            'prediction': {
                'recommended_crop': crop_name
            }
        }), 200

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)