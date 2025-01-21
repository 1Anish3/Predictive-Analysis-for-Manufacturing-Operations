# from flask import Flask, request, jsonify
# import joblib
# import os
#
# app = Flask(__name__)
# model = None
# # Initialize the global model variable
#
#
# @app.route('/load_model', methods=['POST'])
# def load_model():
#     global model
#     try:
#         # Verify if the file exists before loading
#         model_path = 'machine_downtime_model.pkl'
#         if not os.path.exists(model_path):
#             return jsonify({"error": f"Model file '{model_path}' not found."}), 404
#
#         # Load the model
#         model = joblib.load(model_path)
#         return jsonify({"message": "Model loaded successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     global model
#     model_path = 'machine_downtime_model.pkl'
#     if not os.path.exists(model_path):
#         return jsonify({"error": f"Model file '{model_path}' not found."}), 404
#
#     model = joblib.load(model_path)
#
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 400
#
#     # Get input data
#     try:
#         input_data = request.json
#         temperature = input_data.get('Temperature')
#         run_time = input_data.get('Run_Time')
#
#         # Ensure the required inputs are provided
#         if temperature is None or run_time is None:
#             return jsonify({"error": "Missing input data. 'Temperature' and 'Run_Time' are required."}), 400
#
#         # Make prediction
#         prediction = model.predict([[temperature, run_time]])
#         downtime = "Yes" if prediction[0] == 1 else "No"
#         return jsonify({"Downtime": downtime}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/train', methods=['POST'])
# def train_model():
#     global model
#     if data is None:
#         return jsonify({"error": "No data uploaded"}), 400
#
#     # Load the pre-trained model from the .pkl file
#     try:
#         model = joblib.load('machine_downtime_model.pkl')  # Update the path if necessary
#     except Exception as e:
#         return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
#
#     # Preprocess the data
#     # Ensure the target variable is binary (1 for Yes, 0 for No)
#     data['Downtime_Flag'] = data['Downtime_Flag'].map({'Yes': 1, 'No': 0})
#
#     # Define features and target variable
#     X = data[['Temperature', 'Run_Time']]
#     y = data['Downtime_Flag']
#
#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     # Train the model
#     model.fit(X_train, y_train)
#
#     # Make predictions on the validation set
#     y_val_pred = model.predict(X_val)
#
#     # Calculate performance metrics
#     accuracy = accuracy_score(y_val, y_val_pred)
#     f1 = f1_score(y_val, y_val_pred)
#
#     # Return performance metrics
#     return jsonify({
#         "message": "Model trained successfully!",
#         "accuracy": accuracy,
#         "f1_score": f1,
#         "classification_report": classification_report(y_val, y_val_pred, output_dict=True)
#     }), 200
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


# new
# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# import joblib
#
# app = Flask(__name__)
#
# # Global variables for the model and data
# model = None
# data = None
#
#
# @app.route('/')
# def home():
#     return "Welcome to the Machine Downtime Prediction API!"
#
#
# @app.route('/upload', methods=['POST'])
# def upload_data():
#     global data
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     if file:
#         try:
#             data = pd.read_csv(file)
#             return jsonify({"message": "Data uploaded successfully!"}), 200
#         except Exception as e:
#             return jsonify({"error": f"Failed to read the file: {str(e)}"}), 500
#
#
# @app.route('/load_model', methods=['POST'])
# def load_model():
#     global model
#
#     try:
#         model = joblib.load('machine_downtime_model.pkl')  # Update the path if necessary
#         return jsonify({"message": "Model loaded successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
#
#
# @app.route('/train', methods=['POST'])
# def train_model():
#     global model
#     model_path = 'machine_downtime_model.pkl'
#     model = joblib.load(model_path)
#
#     if data is None:
#         return jsonify({"error": "No data uploaded"}), 400
#
#     # Preprocess the data
#     data['Downtime_Flag'] = data['Downtime_Flag'].map({'Yes': 1, 'No': 0})
#
#     # Define features and target variable
#     X = data[['Temperature', 'Run_Time']]
#     y = data['Downtime_Flag']
#
#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     # Train the model
#     model.fit(X_train, y_train)
#
#     # Make predictions on the validation set
#     y_val_pred = model.predict(X_val)
#
#     # Calculate performance metrics
#     accuracy = accuracy_score(y_val, y_val_pred)
#     f1 = f1_score(y_val, y_val_pred)
#
#     # Return performance metrics
#     return jsonify({
#         "message": "Model trained successfully!",
#         "accuracy": accuracy,
#         "f1_score": f1,
#         "classification_report": classification_report(y_val, y_val_pred, output_dict=True)
#     }), 200
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     global model
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 400
#
#     # Get input data
#     input_data = request.json
#     temperature = input_data.get('Temperature')
#     run_time = input_data.get('Run_Time')
#
#     # Make prediction
#     try:
#         prediction = model.predict([[temperature, run_time]])
#         downtime = "Yes" if prediction[0] == 1 else "No"
#         return jsonify({"Downtime": downtime}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

app = Flask(__name__)

# Global variables for the model and data
model = None
data = None


@app.route('/')
def home():
    return "Welcome to the Machine Downtime Prediction API!"


@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            data = pd.read_csv(file)
            return jsonify({"message": "Data uploaded successfully!"}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to read the file: {str(e)}"}), 500


@app.route('/load_model', methods=['POST'])
def load_model():
    global model
    try:
        model = joblib.load('machine_downtime_model.pkl')  # Update the path if necessary
        return jsonify({"message": "Model loaded successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train_model():
    global model
    if data is None:
        return jsonify({"error": "No data uploaded"}), 400

    # Preprocess the data
    data['Downtime_Flag'] = data['Downtime_Flag'].map({'Yes': 1, 'No': 0})

    # Define features and target variable
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)

    # Calculate performance metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    # Return performance metrics
    return jsonify({
        "message": "Model trained successfully!",
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": classification_report(y_val, y_val_pred, output_dict=True)
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 400

    # Get input data
    input_data = request.json
    temperature = input_data.get('Temperature')
    run_time = input_data.get('Run_Time')

    # Make prediction
    try:
        prediction = model.predict([[temperature, run_time]])
        downtime = "Yes" if prediction[0] == 1 else "No"
        return jsonify({"Downtime": downtime}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)