#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Environment ready.")


# In[2]:


get_ipython().system('mkdir -p /root/.kaggle')
get_ipython().system('cp /content/kaggle.json /root/.kaggle/')
get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')

print("Kaggle configured.")


# In[3]:


get_ipython().system('kaggle datasets download -d olistbr/brazilian-ecommerce')

print("Download complete.")


# In[4]:


import zipfile

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:

    zip_ref.extractall("dataset")

print("Extraction complete.")


# In[5]:


from google.colab import drive
drive.mount('/content/drive')


# In[6]:


orders = pd.read_csv("dataset/olist_orders_dataset.csv")
customers = pd.read_csv("dataset/olist_customers_dataset.csv")
order_items = pd.read_csv("dataset/olist_order_items_dataset.csv")
payments = pd.read_csv("dataset/olist_order_payments_dataset.csv")
reviews = pd.read_csv("dataset/olist_order_reviews_dataset.csv")
sellers = pd.read_csv("dataset/olist_sellers_dataset.csv")
products = pd.read_csv("dataset/olist_products_dataset.csv")

print("Tables loaded successfully.")


# In[7]:


print("Orders:", orders.shape)
print("Customers:", customers.shape)
print("Order Items:", order_items.shape)
print("Payments:", payments.shape)
print("Reviews:", reviews.shape)
print("Sellers:", sellers.shape)
print("Products:", products.shape)


# In[8]:


df = orders.merge(customers, on="customer_id", how="left")

order_items_agg = order_items.groupby("order_id").agg({
    "price": "sum",
    "freight_value": "sum"
}).reset_index()

df = df.merge(order_items_agg, on="order_id", how="left")

payments_agg = payments.groupby("order_id").agg({
    "payment_value": "sum"
}).reset_index()

df = df.merge(payments_agg, on="order_id", how="left")

reviews_agg = reviews.groupby("order_id").agg({
    "review_score": "mean"
}).reset_index()

df = df.merge(reviews_agg, on="order_id", how="left")

print("Master dataset created.")
print("Shape:", df.shape)


# In[9]:


date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

print("Date conversion complete.")


# In[10]:


df = df[df["order_status"] == "delivered"]

df = df.dropna(subset=["order_delivered_customer_date"])

df["actual_days"] = (
    df["order_delivered_customer_date"] -
    df["order_purchase_timestamp"]
).dt.days

df["estimated_days"] = (
    df["order_estimated_delivery_date"] -
    df["order_purchase_timestamp"]
).dt.days

df["delay_days"] = df["actual_days"] - df["estimated_days"]

print("Time features created.")
print("Shape after cleaning:", df.shape)


# In[11]:


def classify_delay(days):
    if days <= 0:
        return 0
    elif days <= 3:
        return 1
    else:
        return 2

df["delivery_risk"] = df["delay_days"].apply(classify_delay)

print("Class distribution:")
print(df["delivery_risk"].value_counts())


# In[12]:


# ---------- Advanced Time Features ----------

df["processing_days"] = (
    df["order_approved_at"] - df["order_purchase_timestamp"]
).dt.days

df["shipping_days"] = (
    df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
).dt.days

df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday

# ---------- Ratio Features (Very Powerful) ----------

df["delivery_ratio"] = df["actual_days"] / (df["estimated_days"] + 1)
df["processing_ratio"] = df["processing_days"] / (df["actual_days"] + 1)

# ---------- Seller Complexity ----------

seller_per_order = order_items.groupby("order_id").agg(
    num_sellers=("seller_id", "nunique"),
    num_items=("order_item_id", "count")
).reset_index()

df = df.merge(seller_per_order, on="order_id", how="left")

# ---------- Payment Features ----------

payment_agg = payments.groupby("order_id").agg(
    total_payment=("payment_value", "sum"),
    payment_installments=("payment_installments", "max")
).reset_index()

df = df.merge(payment_agg, on="order_id", how="left")

print("Advanced features added.")
print("New shape:", df.shape)


# In[13]:


drop_cols = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",

    # Remove leakage features
    "actual_days",
    "estimated_days",
    "delay_days",
    "delivery_ratio",
    "processing_ratio",
    "shipping_days"
]

df_model = df.drop(columns=drop_cols)

print("Model dataset shape:", df_model.shape)



# In[14]:


# Separate features and target

X = df_model.drop("delivery_risk", axis=1)
y = df_model["delivery_risk"]

print("Feature shape:", X.shape)
print("Target distribution:")
print(y.value_counts(normalize=True))


# In[15]:


# ---------- Keep Only Numeric Columns ----------

X = X.select_dtypes(include=["int64", "float64"])

print("Numeric feature shape:", X.shape)
print("Remaining columns:")
print(X.columns)

# ---------- Handle Missing Values ----------

print("\nMissing values before cleaning:")
print(X.isnull().sum().sort_values(ascending=False).head(10))

# Fill numeric columns with median
X = X.fillna(X.median())

print("\nMissing values after cleaning:")
print(X.isnull().sum().sum())


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# In[17]:


# Keep only numeric columns

X = X.select_dtypes(include=["int64", "float64"])

print("Numeric feature shape:", X.shape)
print("Remaining columns:")
print(X.columns)


# In[18]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class Weights:", class_weights)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        multi_class="multinomial"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weights,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight=class_weights,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
}

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Macro)": precision_score(y_test, y_pred, average="macro"),
        "Recall (Macro)": recall_score(y_test, y_pred, average="macro"),
        "F1 Score (Macro)": f1_score(y_test, y_pred, average="macro"),
        "Training Time (s)": round(end - start, 2)
    })

print("All models trained.")


# In[20]:


results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro")
    })

results_df = pd.DataFrame(results)
results_df



# In[24]:


import pandas as pd

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1", ascending=False)


print("Model Comparison:")
display(results_df)


# In[26]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.bar(results_df["Model"], results_df["F1"])

plt.xticks(rotation=45)
plt.title("Model Comparison (Macro F1 Score)")
plt.ylabel("Macro F1 Score")
plt.show()


# In[27]:


item_agg = order_items.groupby("order_id").agg(
    total_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum")
).reset_index()

df = df.merge(item_agg, on="order_id", how="left")

print("Order item features added.")


# In[28]:


payment_agg = payments.groupby("order_id").agg(
    payment_installments=("payment_installments", "max"),
    payment_value=("payment_value", "sum")
).reset_index()

df = df.merge(payment_agg, on="order_id", how="left")

print("Payment features added.")


# In[29]:


review_agg = reviews.groupby("order_id").agg(
    review_score=("review_score", "mean")
).reset_index()

df = df.merge(review_agg, on="order_id", how="left")

print("Review features added.")


# In[30]:


seller_agg = order_items.groupby("seller_id").agg(
    seller_total_orders=("order_id", "count")
).reset_index()


seller_per_order = order_items.groupby("order_id").agg(
    num_sellers=("seller_id", "nunique")
).reset_index()

df = df.merge(seller_per_order, on="order_id", how="left")

print("Correct seller features added.")

print("Final dataset shape:", df.shape)



# In[31]:


print("Final dataset shape:", df.shape)
df.head()


# In[32]:


from sklearn.model_selection import cross_val_score

print("Cross Validation (5-fold) — Macro F1")

for name, model in models.items():

    if name == "Logistic Regression":
        scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    else:
        scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")

    print(f"{name}:")
    print("Mean F1:", round(scores.mean(), 4))
    print("Std Dev :", round(scores.std(), 4))
    print("-" * 40)


# In[33]:


from imblearn.over_sampling import SMOTE

print("Before SMOTE:", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", y_train_sm.value_counts())


# In[34]:


from xgboost import XGBClassifier

best_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    learning_rate=0.03,
    max_depth=7,
    n_estimators=800,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=3,
    scale_pos_weight=8,   # Increase for minority sensitivity
    random_state=42
)
best_model.fit(X_train_sm, y_train_sm)




print("Final tuned XGBoost trained successfully.")



# In[35]:


import shap

print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)


# In[36]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# In[37]:


get_ipython().system('pip install imbalanced-learn')


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Macro ROC AUC
y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

print("\nMacro ROC-AUC:", round(roc_macro, 4))


# In[39]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

y_test_bin = label_binarize(y_test, classes=[0,1,2])

plt.figure(figsize=(8,6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve")
plt.legend()
plt.show()


# In[40]:


import joblib

joblib.dump(best_model, "delivery_risk_model.pkl")

print("Model saved successfully.")


# In[41]:


import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1"]

for model_name in results_df["Model"]:
    model_data = results_df[results_df["Model"] == model_name]

    plt.figure(figsize=(6,4))
    plt.bar(metrics, model_data[metrics].values[0])
    plt.title(f"{model_name} Performance Metrics")
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.show()


# In[42]:


results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Model Performance Comparison")
plt.ylim(0,1)
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()


# In[43]:


import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Final Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[44]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_proba = best_model.predict_proba(X_test)

plt.figure(figsize=(8,6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'--')
plt.title("Multi-Class ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# In[45]:


import xgboost as xgb

plt.figure(figsize=(8,6))
xgb.plot_importance(best_model, max_num_features=10)
plt.title("Top 10 Important Features")
plt.show()


# In[46]:


import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)


# In[47]:


y.value_counts().plot(kind="bar")
plt.title("Delivery Risk Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.boxplot(x=y_test, y=X_test["processing_days"])
plt.title("Processing Days vs Risk Level")
plt.xlabel("Risk Class")
plt.ylabel("Processing Days")
plt.show()


