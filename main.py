import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Step 1: Load your dataset (update the path accordingly)
data = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Project/stock_data_model.csv")

# Step 2: Handle missing values
data = data.dropna()  # Option to fill missing values: data.fillna(data.mean(), inplace=True)

# Step 3: Remove duplicates
data = data.drop_duplicates()

# Step 4: Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 5: Normalize/scale features
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Step 6: Create new features
data['MA_7'] = data['Close'].rolling(window=7).mean()  # 7-day moving average
data['Lag_Close'] = data['Close'].shift(1)  # Previous day's close

# Step 7: Drop NaN values created by rolling/lag features
data = data.dropna()

# Step 8: Define the features and target variable
X = data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'Lag_Close', '20D_MA', '50D_MA']]  # Add these if using in model

y = data['Close']  # Target variable

# Step 9: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Make predictions on the testing set
y_pred = model.predict(X_test)

# Step 12: Calculate MAE and R-squared
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 13: Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R²): {r2}')

# Step 14: Save the trained model to a file named 'stock_model.pkl'
joblib.dump(model, 'stock_model.pkl')
print("Model saved as 'stock_model.pkl'.")

# Function to load the model
def load_model():
    return joblib.load('stock_model.pkl')

# Function to make predictions
def make_prediction(input_data):
    # Load the saved model
    model = load_model()
    
    # Make a prediction using the input data
    prediction = model.predict(input_data)
    return prediction

# Step 15: Example new data (ensure it matches the training feature columns)
new_data = pd.DataFrame(
    [[0.7, 0.8, 0.6, 0.45, 0.4, 0.5]],  # Replace with actual values
    columns=['Open', 'High', 'Low', 'Volume', 'MA_7', 'Lag_Close']
)

# Step 16: Get the prediction
predicted_price = make_prediction(new_data)
print(f'Predicted Close Price: {predicted_price[0]}')
