import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("crypto_data.csv")

# Drop rows with any missing values (optional: use fillna instead)
df.dropna(inplace=True)

# Select only the numeric columns for scaling
numeric_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Add back the non-numeric columns
df_scaled['coin'] = df['coin'].values
df_scaled['symbol'] = df['symbol'].values
df_scaled['date'] = df['date'].values

# Save cleaned data
df_scaled.to_csv("cleaned_crypto_data.csv", index=False)

print("Preprocessing completed and saved as 'cleaned_crypto_data.csv'")