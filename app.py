import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model & scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Fixed feature order (must match training order exactly)
feature_order = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        # Extract features in correct order
        input_array = np.array([data[feat] for feat in feature_order]).reshape(1, -1)
        new_data = scalar.transform(input_array)
        output = regmodel.predict(new_data)[0]
        return jsonify({'prediction': float(output)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data in correct order
        data = [float(request.form.get(feat)) for feat in feature_order]
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template("home.html", prediction_text=f"The House price prediction is {output}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
