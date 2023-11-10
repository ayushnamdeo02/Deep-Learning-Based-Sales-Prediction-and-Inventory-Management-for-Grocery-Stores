import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('./dataset/test3.csv')

# Keeping the Item_ID for later use
item_ids = df['Item_ID'].values

# One-hot encoding
categorical_cols = ['Item_Name', 'Category', 'Weather']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# The target variable is the actual sales values
y = df['Predicted_sales'].values
df.drop(columns=['Item_ID', 'Date', 'Predicted_sales'], inplace=True)

X = df.values

# Scaling the features
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

# Scaling the target
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
gbr.fit(X_train, y_train.ravel())

def predict_sales_for_item_gbr(item_no):
    if item_no not in item_ids:
        return f"Item number {item_no} does not exist."
    item_idx = list(item_ids).index(item_no)
    latest_data = X[item_idx].reshape(1, -1)
    predicted_sales_scaled = gbr.predict(latest_data)
    predicted_sales = scaler_y.inverse_transform(predicted_sales_scaled.reshape(-1, 1))[0][0]
    return f"Predicted future sales for item {item_no} (GBR): {predicted_sales:.2f}"

# Predict on validation set for Gradient Boosting Regressor
y_val_pred_gbr = gbr.predict(X_val)

# Calculate evaluation metrics for Gradient Boosting Regressor
mae_gbr = mean_absolute_error(y_val, y_val_pred_gbr)
mse_gbr = mean_squared_error(y_val, y_val_pred_gbr)
r2_gbr = r2_score(y_val, y_val_pred_gbr)

print("Gradient Boosting Regressor Metrics:")
print("Mean Absolute Error (MAE):", mae_gbr)
print("Mean Squared Error (MSE):", mse_gbr)
print("R^2 Score:", r2_gbr)
# Define a threshold (e.g., 5%)
threshold = 0.05

# Calculate the number of predictions within the threshold
within_threshold = np.sum(np.abs(y_val - y_val_pred_gbr.reshape(-1, 1)) <= threshold * y_val)

# Calculate the accuracy-like metric
accuracy_like_metric = within_threshold / len(y_val)

print("Accuracy-like Metric (within 5% threshold):", accuracy_like_metric)


# Example usage
print(predict_sales_for_item_gbr(5))
