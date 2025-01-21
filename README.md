# Predictive-Analysis-for-Manufacturing-Operations

This repository contains a Flask RESTful API for predicting machine downtime based on temperature and run time. The API allows users to upload manufacturing data, load a pre-trained model, and make predictions.

## Technologies and Skills Used

- **Python**: The primary programming language used for developing the API and machine learning model.
- **Flask**: A lightweight web framework for building the RESTful API.
- **pandas**: A data manipulation and analysis library used for handling CSV data.
- **scikit-learn**: A machine learning library used for building and evaluating the predictive model.
- **joblib**: A library used for saving and loading the machine learning model in `.pkl` format.
- **Postman**: A tool used for testing API endpoints and making HTTP requests.
- **Git**: Version control system used for managing the project repository.
- **GitHub**: Platform for hosting the project repository and collaborating with others.

## Project Structure

The project consists of the following Python files:

1. **Machine Learning Notebook**: 
   - This is a Jupyter Notebook where the machine learning model was developed and trained. It includes data preprocessing, model training, and evaluation steps.

2. **Flask API**: 
   - This Python file (`app.py`) contains the Flask API implementation. It provides endpoints for uploading data, loading the pre-trained model, training the model, and making predictions.

## Requirements

- Python 3.x
- Flask
- pandas
- scikit-learn
- joblib

## Setup Instructions

1. **Clone the Repository**:
   - Clone this repository to your local machine using the following command:
     - `git clone https://github.com/yourusername/MachineDowntimeAPI.git`

2. **Navigate to the Project Directory**:
   - Change to the project directory:
     - `cd MachineDowntimeAPI`

3. **Create a Virtual Environment (Optional)**:
   - It is recommended to create a virtual environment to manage dependencies:
     - For Windows: `python -m venv venv` and then activate it using `venv\Scripts\activate`
     - For macOS/Linux: `python -m venv venv` and then activate it using `source venv/bin/activate`

4. **Install Required Libraries**:
   - Install the required libraries using pip:
     - `pip install Flask pandas scikit-learn joblib`

5. **Add the Model File**:
   - Ensure that you have your pre-trained model file (`machine_downtime_model.pkl`) in the project directory.

6. **Run the Flask Application**:
   - Start the Flask server by running the following command:
     - `python app.py`

7. **Access the API**:
   - The API will be running at `http://127.0.0.1:5000/`.
  

![image](https://github.com/user-attachments/assets/aa09ebe7-01e0-40c3-a92d-814add9d8752)
![image](https://github.com/user-attachments/assets/87fa57f2-ef96-4814-801d-ed75e781c40f)
![image](https://github.com/user-attachments/assets/81bc41ca-cecf-4371-8c22-f72c99b98822)


