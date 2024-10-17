from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('stock_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_price = float(request.form['Open'])
        high = float(request.form['High'])
        low = float(request.form['Low'])
        volume = float(request.form['Volume'])
        ma_7 = float(request.form['MA_7'])  # Ensure this matches your model features
        lag_close = float(request.form['Lag_Close'])

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[open_price, high, low, volume, ma_7, lag_close]],
                                  columns=['Open', 'High', 'Low', 'Volume', 'MA_7', 'Lag_Close'])

        # Make a prediction
        prediction = model.predict(input_data)
        predicted_price = prediction[0]

        return f'The predicted close price is: {predicted_price}'
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)  # Change to app.run(host='0.0.0.0', port=5000) for production
