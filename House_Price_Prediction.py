# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Make sure your dataset has columns like 'Area', 'Bedrooms', 'Location', and 'Price'
df = pd.read_csv('house_prices.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

df['date'] = pd.to_datetime(df['date'])

# Extract year, month, and day from 'Date'
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day

df = df.drop(['date', 'street','statezip','country'], axis=1)

# Preprocessing the data
# Encoding the 'Location' column
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Define features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plotting the results
plt.figure(figsize=(12, 6))

# Scatter plot of Actual vs Predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.grid(True)

# Residual plot
plt.subplot(1, 2, 2)
sns.residplot(x=y_test, y=y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)

plt.tight_layout()
plt.show()
