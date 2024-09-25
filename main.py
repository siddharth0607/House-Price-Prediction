from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load the model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the columns used during training
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    
    # Initialize input features array
    input_features = np.zeros(len(X_columns))
    
    # Basic feature assignments
    input_features[0] = float(data.get('sqft', 0))
    input_features[1] = float(data.get('bath', 0))
    input_features[2] = float(data.get('balcony', 0))
    input_features[3] = float(data.get('bhk', 0))

    # Handle categorical features
    for feature in ['location', 'area', 'availability']:
        if feature in data and data[feature] in X_columns:
            index = np.where(np.array(X_columns) == data[feature])[0][0]
            input_features[index] = 1

    # Ensure input shape is correct
    input_features = input_features.reshape(1, -1)

    print("Input Features:", input_features)
    # Making prediction
    prediction = model.predict(input_features)[0]
    formatted_prediction = "₹{:.2f}".format(prediction * 100000)
    
    return jsonify(formatted_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    input_features = np.zeros(len(X_columns))
    input_features[0] = float(request.form.get('sqft', 0))
    input_features[1] = float(request.form.get('bath', 0))
    input_features[2] = float(request.form.get('balcony', 0))
    input_features[3] = float(request.form.get('bhk', 0))
    
    for feature in ['location', 'area', 'availability']:
        if request.form.get(feature) in X_columns:
            index = np.where(np.array(X_columns) == request.form.get(feature))[0][0]
            input_features[index] = 1
    
    # Ensure input shape is correct
    input_features = input_features.reshape(1, -1)
    
    prediction = model.predict(input_features)[0]
    formatted_prediction = "₹{:.2f}".format(prediction * 100000)
    
    return render_template('home.html', prediction_text=f"The estimated house price is {formatted_prediction}.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)