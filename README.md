# 📦 Amazon Supply Chain Intelligence: Delivery Delay Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Flask-API-green.svg)
![UI](https://img.shields.io/badge/Streamlit-Frontend-red.svg)
![Model](https://img.shields.io/badge/Model-XGBoost-orange.svg)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline and web application designed to predict the risk of an e-commerce order facing delivery delays. It is trained on the **Brazilian E-Commerce Public Dataset by Olist** and features a sophisticated **XGBoost Classifier**. 

The system exposes the predictive model via a **Flask REST API**, which is then consumed by a highly interactive **Streamlit Web Dashboard**.

## 🧠 The Machine Learning Model (Why XGBoost?)
Supply chain data typically features a severe class imbalance—most deliveries arrive on time, while delays are rare but critical. 

While algorithms like standard Gradient Boosting might show high raw accuracy by simply guessing "On Time" for every order, this project implements **XGBoost (Extreme Gradient Boosting)** because:
1. **Handling Imbalance:** It natively supports scaling positive weights (`scale_pos_weight`) to penalize missing rare delayed deliveries.
2. **Speed & Scalability:** It uses parallel tree construction, crucial for handling massive e-commerce datasets.
3. **Regularization:** It provides advanced hyperparameters (`gamma`, `colsample_bytree`) to prevent overfitting.

### The Prediction Classes:
- `0`: **Low Risk** (Expected to arrive on time or early)
- `1`: **Medium Risk** (Slight delay of ≤ 3 days)
- `2`: **High Risk** (Significant delay of > 3 days)

---

## 📂 Project Structure
```text
Amazon-Project/
├── Amazon_SupplyChain_Intelligence_Final.ipynb  # Original data analysis, EDA, and model training
├── create_dummy_model.py                        # Script to generate a synthetic model for quick UI testing
├── app.py                                       # Flask Backend REST API
├── frontend.py                                  # Streamlit UI Dashboard
├── delivery_risk_model.pkl                      # The serialized XGBoost Model
├── model_features.pkl                           # Saved feature columns ensuring exact alignment
├── olist_customers_dataset.csv                  # Sample raw dataset
└── README.md                                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Bhuvan7888/Amazon-Project-.git
cd Amazon-Project-
pip install flask streamlit pandas numpy xgboost scikit-learn joblib requests
```

### 2. Prepare the Model
Since the full multi-gigabyte Kaggle dataset isn't stored in this repo, a script is provided to generate a logical synthetic model. This ensures the app works out-of-the-box!
```bash
python create_dummy_model.py
```
*(This script generates `delivery_risk_model.pkl` and maps high processing days and low review scores to higher delay risks.)*

### 3. Run the Flask API Backend
Open a terminal and start the Flask server. It will load the model into memory and listen for predictions on port `5001`.
```bash
python app.py
```
*(Note: We use port 5001 to prevent conflicts with the macOS AirPlay Receiver).*

### 4. Run the Streamlit Frontend UI
Open a **second terminal** and launch the UI:
```bash
streamlit run frontend.py
```
This will open `http://localhost:8501` in your web browser. 

---

## 💻 Usage
1. Open the Streamlit Dashboard.
2. Tweak the logistical parameters (e.g., increase **Processing Days** or decrease the **Review Score**).
3. Click **Predict Delivery Risk**.
4. The UI will hit the Flask `/predict` endpoint and instantly return a color-coded risk assessment!

## 🔧 Future Improvements
- **Live Database Integration:** Connect the Flask app directly to a live PostgreSQL database holding real-time order states.
- **Weather API Integration:** Add external historical weather data mapped by Zip Code to further refine delay prediction accuracy.
