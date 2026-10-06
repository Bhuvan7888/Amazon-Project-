# Amazon Supply Chain Intelligence - Delivery Delay Risk Prediction

This project builds an end-to-end machine learning application to predict the risk of delivery delay for Amazon orders using a Flask backend API and a Streamlit frontend UI.

## Project Structure

- `Amazon_SupplyChain_Intelligence_Final.ipynb`: Original Jupyter Notebook with data exploration, preprocessing, and model training.
- `create_dummy_model.py`: Script to generate a trained XGBoost dummy model (`delivery_risk_model.pkl`) so the API can run without requiring the entire Kaggle dataset to be downloaded locally.
- `app.py`: Flask backend application serving the ML model via a `/predict` REST API.
- `frontend.py`: Streamlit frontend application providing an interactive UI to test the model.
- `olist_customers_dataset.csv`: Example dataset containing customer information.

## Requirements

Ensure you have Python 3.8+ installed.

You can install the required packages using:

```bash
pip install flask streamlit pandas numpy xgboost scikit-learn joblib requests
```

## Setup & Running the Application

### 1. Generate the Model (Optional if already exists)

If the model files (`delivery_risk_model.pkl` and `model_features.pkl`) are missing, run the following script to generate them:

```bash
python create_dummy_model.py
```
*Note: This creates a dummy model with the same feature space as the original notebook for testing the end-to-end integration without needing the full dataset.*

### 2. Start the Flask Backend

Open a terminal and run the Flask application:

```bash
python app.py
```

The Flask API will start running on `http://localhost:5001`.

### 3. Start the Streamlit Frontend

Open a new terminal (leave the Flask backend running) and start the Streamlit application:

```bash
streamlit run frontend.py
```

Streamlit will launch a browser window (usually `http://localhost:8501`) with the UI.

## Usage

1. In the Streamlit UI, enter the order details.
2. Click the **Predict Delivery Risk** button.
3. The frontend will send a request to the Flask backend, which runs the input through the XGBoost model and returns the predicted risk level.
4. The result (Low Risk, Medium Risk, or High Risk) will be displayed on the screen.
