import os
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
MODEL_PATH = 'diabetes_dt.pkl'
META_PATH = 'model_meta.pkl'  # contains feature order and medians

FEATURES = [
    'Glucose',
    'BMI',
    'Age',
    'DiabetesPedigreeFunction',
    'Insulin',
    'Pregnancies',
    'BloodPressure',
    'SkinThickness'
]


def train_and_save_model(force_retrain=False):
    """Train the DecisionTree model on local diabetes.csv if model file is missing.
    Saves model and metadata (feature order, medians) as pickles.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH) and not force_retrain:
        print('Model already exists. Skipping training.')
        return

    csv_path = 'diabetes.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Place the dataset in the same folder as app.py")

    df = pd.read_csv(csv_path)

    # Replace zeroes with NaN for physiologically invalid zeros
    cols_with_invalid_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_invalid_zero:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Fill NaN with median
    medians = {}
    for col in FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            medians[col] = float(med)
        else:
            raise KeyError(f"Expected column '{col}' not found in CSV")

    # Ensure outcome column is present
    if 'Outcome' not in df.columns:
        raise KeyError("Expected 'Outcome' column not found in CSV")

    X = df[FEATURES]
    y = df['Outcome'].astype(int)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Save model and metadata
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(META_PATH, 'wb') as f:
        pickle.dump({'features': FEATURES, 'medians': medians}, f)

    print('Model and metadata saved.')


def load_model_and_meta():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        train_and_save_model()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    return model, meta


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint that accepts JSON or form data and returns prediction JSON."""
    model, meta = load_model_and_meta()
    medians = meta['medians']
    features = meta['features']

    # Accept JSON body or form-encoded POST
    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()

    # Build input vector in the training feature order
    inp = []
    missing = []
    for feat in features:
        raw = data.get(feat)
        if raw is None or raw == '':
            # fill with median if not provided
            val = medians.get(feat, 0.0)
        else:
            try:
                val = float(raw)
            except ValueError:
                return jsonify({'error': f'Invalid value for {feat}: {raw}'}), 400
        inp.append(val)

    arr = np.array(inp).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(arr)[0][pred])

    label = 'Diabetic' if pred == 1 else 'Non Diabetic'

    return jsonify({'prediction': pred, 'label': label, 'probability': prob})


@app.route('/result', methods=['POST'])
def result_page():
    """Form submit endpoint: renders a result page with prediction."""
    model, meta = load_model_and_meta()
    medians = meta['medians']
    features = meta['features']

    data = request.form.to_dict()
    inp = []
    for feat in features:
        raw = data.get(feat)
        if raw is None or raw == '':
            val = medians.get(feat, 0.0)
        else:
            try:
                val = float(raw)
            except ValueError:
                val = medians.get(feat, 0.0)
        inp.append(val)

    arr = np.array(inp).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(arr)[0][pred])

    label = 'Diabetic' if pred == 1 else 'Non Diabetic'

    # pass back original inputs for display
    display_inputs = {feat: inp[i] for i, feat in enumerate(features)}
    return render_template('result.html', prediction=pred, label=label, probability=prob, inputs=display_inputs)


if __name__ == '__main__':
    # Train if model not present
    try:
        train_and_save_model()
    except Exception as e:
        print('Warning: could not train model automatically:', e)
    app.run(debug=True)
