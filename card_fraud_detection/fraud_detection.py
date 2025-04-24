import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
from flask import Flask, request, jsonify

# Load the dataset (replace with your actual dataset file path)
df = pd.read_csv('credit_card_transactions.csv')

# Preprocessing
def preprocess_data(df):
    df.fillna(0, inplace=True)  # Handle missing values
    X = df.drop(['fraudulent'], axis=1)
    y = df['fraudulent']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return X_resampled, y_resampled, scaler

# Preprocess and split the data
X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

# Save the model and scaler
with open('fraud_detection_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    transaction_data = np.array(data['transaction']).reshape(1, -1)
    transaction_data_scaled = scaler.transform(transaction_data)
    prediction = model.predict(transaction_data_scaled)
    fraud_probability = model.predict_proba(transaction_data_scaled)[:, 1]
    result = {
        'prediction': int(prediction[0]),
        'fraud_probability': fraud_probability[0]
    }
    return jsonify(result)

# Manual test for a single transaction (run only if script is executed directly)
if __name__ == '__main__':
    # Manual input section
    print("\n=== Manual Transaction Prediction ===")
    try:
        # Take input from user
        amount = float(input("Enter transaction amount: "))
        time_delta = int(input("Enter time since last transaction (in seconds): "))
        merchant_id = int(input("Enter merchant ID (e.g., 1000-1999): "))
        user_id = int(input("Enter user ID (e.g., 1-99): "))
        device_type = int(input("Enter device type (0: mobile, 1: desktop, 2: tablet): "))

        sample_transaction = [amount, time_delta, merchant_id, user_id, device_type]
        input_data = np.array(sample_transaction).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)[0]
        fraud_prob = model.predict_proba(input_data_scaled)[0][1]

        # Output
        print(f"\nTransaction: {sample_transaction}")
        print(f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
        print(f"Fraud Probability: {fraud_prob:.4f}")

    except Exception as e:
        print("❌ Error:", e)

    # Start Flask server after manual test
    print("\nStarting Flask server...")
    app.run(debug=True)
