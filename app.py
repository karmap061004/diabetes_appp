from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

# Cache the diabetes data median values for preprocessing
df_ref = pd.read_csv('diabetes.csv')
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
medians = {}
for col in zero_cols:
    col_data = df_ref[col].replace(0, np.nan)
    medians[col] = col_data.median()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create features array in the correct order
        features = np.array([[
            data['pregnancies'],
            data['glucose'],
            data['bp'],
            data['skin'],
            data['insulin'],
            data['bmi'],
            data['dpf'],
            data['age']
        ]])
        
        # Predict probability
        probability = model.predict_proba(features)[0][1]
        
        # Custom threshold (reduce false negatives)
        threshold = 0.2
        prediction = int(probability >= threshold)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "Moderate Risk"
            risk_color = "yellow"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        return jsonify({
            'probability': round(float(probability), 4),
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_color': risk_color
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)