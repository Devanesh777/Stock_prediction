import pandas as pd
import joblib

def load_model():
    # Load the trained model from the file
    return joblib.load('stock_model.pkl')

def make_prediction(input_data):
    # Load the saved model
    model = load_model()
    # Make a prediction using the input data
    prediction = model.predict(input_data)
    return prediction

def get_user_input():
    # Prompt the user to input the required features
    open_price = float(input("Enter the Open price: "))
    high_price = float(input("Enter the High price: "))
    low_price = float(input("Enter the Low price: "))
    volume = float(input("Enter the Volume: "))
    ma_7 = float(input("Enter the 7-day Moving Average: "))
    lag_close = float(input("Enter the Previous day's Close: "))
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[open_price, high_price, low_price, volume, ma_7, lag_close]], 
                               columns=['Open', 'High', 'Low', 'Volume', 'MA_7', 'Lag_Close'])
    return input_data

def make_terminal_prediction():
    input_data = get_user_input()
    predicted_price = make_prediction(input_data)
    print(f"Predicted Close Price: {predicted_price[0]}")

if __name__ == "__main__":
    make_terminal_prediction()
