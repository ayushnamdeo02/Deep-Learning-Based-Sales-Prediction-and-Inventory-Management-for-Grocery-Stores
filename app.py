from flask import Flask, render_template, request
import joblib
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)

# Load the dataset
df_original = pd.read_csv('./dataset/test3.csv')

# One-hot encoding
categorical_cols = ['Item_Name', 'Category', 'Weather']
df = pd.get_dummies(df_original, columns=categorical_cols, drop_first=True)
df.drop(columns=['Item_ID', 'Date', 'Predicted_sales'], inplace=True)

# Keeping the Item_ID for later use
item_ids = df_original['Item_ID'].values

# Load the saved model and scalers
loaded_gbr = joblib.load("./models/gbr_model.pkl")
loaded_scaler_X = joblib.load("./models/scaler_X.pkl")
loaded_scaler_y = joblib.load("./models/scaler_y.pkl")

def predict_sales_for_item_gbr(item_no,avail):
    if item_no not in item_ids:
        return f"Item number {item_no} does not exist."
    
    item_name = df_original[df_original['Item_ID'] == item_no]['Item_Name'].iloc[0]
    item_idx = list(item_ids).index(item_no)
    latest_data = df.iloc[item_idx].values.reshape(1, -1)
    
    # Scaling the input data
    scaled_data = loaded_scaler_X.transform(latest_data)

    # Predict using the loaded model
    predicted_sales_scaled = loaded_gbr.predict(scaled_data)
    predicted_sales = loaded_scaler_y.inverse_transform(predicted_sales_scaled.reshape(-1, 1))[0][0]
    sales = predicted_sales - float(avail)

    return f"Predicted future sales for item {item_no} (Item Name: {item_name}): {predicted_sales:.2f} | The Quantity to Increase/Decrease is {sales}"

def plot_sales_vs_date_with_endpoints(item_no):
    import matplotlib.pyplot as plt

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
    
    # Convert the plot to PNG image in binary format
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    # Encode the binary data to base64 string
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = ""
    avail=""
    graph = None
    if request.method == 'POST':
        item_id = int(request.form.get('item_id'))
        avail = (request.form.get('avail'))
        prediction = predict_sales_for_item_gbr(item_id,avail)
        graph = plot_sales_vs_date_with_endpoints(item_id)
    return render_template('index.html', prediction=prediction, graph=graph)

if __name__ == "__main__":
    app.run(debug=True)
