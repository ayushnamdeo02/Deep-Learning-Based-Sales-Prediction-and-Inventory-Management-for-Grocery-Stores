import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('./dataset/test3.csv')
df_original = df.copy()

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

joblib.dump(gbr, "gbr_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

def predict_sales_for_item_gbr(item_no):
    # Load the saved model and scalers
    loaded_gbr = joblib.load("gbr_model.pkl")
    loaded_scaler_X = joblib.load("scaler_X.pkl")
    loaded_scaler_y = joblib.load("scaler_y.pkl")
    
    if item_no not in item_ids:
        return f"Item number {item_no} does not exist."
    
    item_name = df_original[df_original['Item_ID'] == item_no]['Item_Name'].iloc[0]
    item_idx = list(item_ids).index(item_no)
    latest_data = X[item_idx].reshape(1, -1)
    predicted_sales_scaled = loaded_gbr.predict(latest_data)
    predicted_sales = loaded_scaler_y.inverse_transform(predicted_sales_scaled.reshape(-1, 1))[0][0]
    
    return f"Predicted future sales for item {item_no} (Item Name: {item_name}): {predicted_sales:.2f}"

def plot_sales_vs_date_with_endpoints(item_no):
    # Filter the dataframe for the given item ID
    item_data = df_original[df_original['Item_ID'] == item_no]
    
    # Extract Date and Predicted_sales columns
    dates = item_data['Date'].values
    sales = item_data['Predicted_sales'].values
    
    # Plot the data with line
    plt.figure(figsize=(12, 6))
    plt.plot(dates, sales, marker='', color='b', label='Sales Trend')
    
    # Overlaying with scatter to emphasize data points
    plt.scatter(dates, sales, color='r', marker='o', label='Sales Data Points')
    
    plt.xticks(rotation=45)
    plt.title(f"Sales vs Date for Item ID {item_no}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test the functions
item_id_input = int(input("Enter the item ID: "))
print(predict_sales_for_item_gbr(item_id_input))
plot_sales_vs_date_with_endpoints(item_id_input)
