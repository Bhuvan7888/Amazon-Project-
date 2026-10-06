import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# Features inferred from the notebook
features = [
    'price', 'freight_value', 'payment_value', 'review_score', 
    'processing_days', 'purchase_month', 'purchase_weekday', 
    'num_sellers', 'total_items', 'payment_installments'
]

# Create synthetic data with logical patterns so the model learns something
n_samples = 1000
np.random.seed(42)

# Generate basic features
X_dict = {
    'price': np.random.uniform(10, 1000, n_samples),
    'freight_value': np.random.uniform(5, 100, n_samples),
    'payment_value': np.random.uniform(15, 1100, n_samples),
    'review_score': np.random.uniform(1, 5, n_samples),
    'processing_days': np.random.randint(0, 15, n_samples),
    'purchase_month': np.random.randint(1, 13, n_samples),
    'purchase_weekday': np.random.randint(0, 7, n_samples),
    'num_sellers': np.random.randint(1, 5, n_samples),
    'total_items': np.random.randint(1, 10, n_samples),
    'payment_installments': np.random.randint(1, 12, n_samples)
}
X_dummy = pd.DataFrame(X_dict)

# Create logic for 'delivery_risk' (y):
# 0 = On Time (Low Risk)
# 1 = Slight Delay (Medium Risk)
# 2 = Significant Delay (High Risk)

y_dummy = []
for i in range(n_samples):
    risk_score = 0
    # High processing days -> higher risk
    if X_dict['processing_days'][i] > 5:
        risk_score += 2
    elif X_dict['processing_days'][i] > 2:
        risk_score += 1
        
    # Low review score correlates with delays in historical data
    if X_dict['review_score'][i] < 3.0:
        risk_score += 1
        
    # Many items or many sellers -> higher risk of logistical issues
    if X_dict['total_items'][i] > 4:
        risk_score += 1
    if X_dict['num_sellers'][i] > 1:
        risk_score += 1
        
    # Assign class based on accumulated risk score
    if risk_score <= 1:
        y_dummy.append(0)
    elif risk_score <= 3:
        y_dummy.append(1)
    else:
        y_dummy.append(2)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    max_depth=3,
    n_estimators=100
)
model.fit(X_dummy, np.array(y_dummy))

joblib.dump(model, 'delivery_risk_model.pkl')
joblib.dump(features, 'model_features.pkl')
print("Logical dummy model created successfully.")
