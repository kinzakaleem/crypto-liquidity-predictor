import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned data
df = pd.read_csv("cleaned_crypto_data.csv")

# Define features and target
X = df[['price', '1h', '24h', '7d', 'mkt_cap']]  # Features
y = df['24h_volume']                            # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(" Mean Squared Error:", mean_squared_error(y_test, y_pred))
print(" R² Score:", r2_score(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'liquidity_model.pkl')
print("Model saved as liquidity_model.pkl")