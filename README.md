# Crop Recommendation System

A Flask-based machine learning web application that recommends the most suitable crop based on soil nutrients and weather conditions.

## Features

- Web form UI for crop prediction
- REST API endpoint for programmatic predictions
- Uses a trained XGBoost model
- Input validation for required API fields

## Tech Stack

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- XGBoost

## Project Structure

```text
Crop_recomendation/
├── application.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── Crop_recommendation.csv
├── model_file/
│   ├── xgboost_crop_model.pkl
│   └── label_encoder.pkl
├── model_notebook/
│   └── model.ipynb
└── templates/
    └── home.html
```

## Model Inputs

The model expects the following 7 features:

1. `N` (Nitrogen)
2. `P` (Phosphorus)
3. `K` (Potassium)
4. `temperature` (°C)
5. `humidity` (%)
6. `ph`
7. `rainfall` (mm)

## Setup and Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python application.py
```

The app starts in debug mode and is typically available at:

- http://127.0.0.1:5000/

## API Usage

### Endpoint

- `POST /api/predict`

### Request Body (JSON)

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### Success Response

```json
{
  "status": "success",
  "input": {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
  },
  "prediction": {
    "recommended_crop": "rice"
  }
}
```

## Notes

- Ensure trained model files exist in `model_file/`:
  - `xgboost_crop_model.pkl`
  - `label_encoder.pkl`
- If model files are missing, the app will fail at startup while loading them.

## License

This project is for educational and learning purposes.
