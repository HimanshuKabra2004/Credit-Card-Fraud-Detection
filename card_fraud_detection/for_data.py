import pandas as pd
import numpy as np

# Create a dummy credit card transactions dataset
np.random.seed(42)
n_samples = 1000

# Generate features
amount = np.random.exponential(scale=100, size=n_samples)
time_delta = np.random.randint(0, 86400, n_samples)  # seconds in a day
merchant_id = np.random.randint(1000, 2000, n_samples)
user_id = np.random.randint(1, 100, n_samples)
device_type = np.random.choice([0, 1, 2], n_samples)  # 0: mobile, 1: desktop, 2: tablet

# Generate target variable (0: legitimate, 1: fraudulent)
fraudulent = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # imbalanced

# Create DataFrame
df = pd.DataFrame({
    'amount': amount,
    'time_delta': time_delta,
    'merchant_id': merchant_id,
    'user_id': user_id,
    'device_type': device_type,
    'fraudulent': fraudulent
})

# Save to CSV
csv_path = "credit_card_transactions.csv"
df.to_csv(csv_path, index=False)
csv_path
