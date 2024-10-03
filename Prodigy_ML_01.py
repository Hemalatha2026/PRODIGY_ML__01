import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\housing.csv")


for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)


features = ['sqft_living', 'sqft_lot', 'sqft_above', 'yr_built', 'sqft_living15']
X = data[features]
y = data['price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)

# Model evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print evaluation results
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot Actual vs Predicted Sale Price
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# Plot Residuals
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Plot distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Pairplot of features and price
plt.figure(figsize=(12, 8))
sns.pairplot(data[features + ['price']])
plt.show()

# Example prediction with realistic values
example = pd.DataFrame({
    'sqft_living': [2000],
    'sqft_lot': [5000],  # Adjusted to a realistic lot size
    'sqft_above': [1500],  # Adjusted to a typical above-ground area
    'yr_built': [1990],  # Adjusted to a more realistic year
    'sqft_living15': [1800]  # Adjusted for neighboring homes' size
})
example_prediction = model.predict(example)
print(f'Example Prediction: ${example_prediction[0]:,.2f}')

# Prepare the entire dataset and make predictions
X_test = data[features]
test_predictions = model.predict(X_test)


submission = pd.DataFrame({'id': data['id'], 'price': test_predictions})
submission.to_csv('submission.csv', index=False)
