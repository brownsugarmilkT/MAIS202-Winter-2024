from flask import Flask, render_template, request, jsonify
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

app = Flask(__name__)

# Define your machine learning model


def create_model():
    set_seed(455)
    np.random.seed(455)
    model = Sequential([
        GRU(units=125, return_sequences=True, input_shape=(60, 1), kernel_regularizer=l1(0.01)),
        Dropout(0.3),  
        GRU(units=125, kernel_regularizer=l1(0.01)),  
        Dropout(0.3),  
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# Load your trained model
model = create_model()

from datetime import datetime, timedelta

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    elif request.method == 'POST':
        # Get user input for stock symbol
        stock_symbol = request.form['stock_symbol']
        
        # Fetch stock price data for the specified stock symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=9*365)
        dataset = yf.download(stock_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        training_set = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))
        window_size = 60
        X_train, y_train = split_sequence(training_set, window_size)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1)
        # Make predictions using your model
        predicted_stock_price = model.predict(X_train)
        

        # Predict the next 30 days
        future_days = 30

        # Predict the next 30 days
        for _ in range(future_days):
            x_input = training_set[-window_size:].reshape(1, window_size, 1)
            future_prediction = model.predict(x_input)
            training_set = np.append(training_set, future_prediction[0])
            predicted_stock_price = np.append(predicted_stock_price, future_prediction, axis=0)

        predicted_stock_price_rescaled = scaler.inverse_transform(predicted_stock_price).flatten()
        actual_stock_price_rescaled = scaler.inverse_transform(y_train).flatten()

        common_days = min(len(actual_stock_price_rescaled), len(predicted_stock_price_rescaled))
        rmse = np.sqrt(mean_squared_error(actual_stock_price_rescaled[-common_days:], predicted_stock_price_rescaled[-common_days:]))

        # Generate plots with updated predictions included 
        train_test_plot_path = generate_train_test_plot(actual_prices=actual_stock_price_rescaled,
                                                        predicted_prices=predicted_stock_price_rescaled)

        # Generate a separate plot for the predictions
        predictions_plot_path = generate_predictions_plot(predicted_prices=predicted_stock_price_rescaled[-future_days:])

        return render_template('results.html', 
                                stock_symbol=stock_symbol,
                                train_test_plot_path=train_test_plot_path,
                                predictions_plot_path=predictions_plot_path,
                                rmse=rmse)

def generate_predictions_plot(predicted_prices):
    plt.figure(figsize=(16, 8), facecolor='lightblue')
    ax = plt.subplot(111)
    ax.set_facecolor('lightseagreen')
    ax.plot(predicted_prices, color='red', linewidth=2, label="Predicted")
    leg = plt.legend(facecolor='lightgrey')
    for text in leg.get_texts():
        text.set_color('black')
    plt.title("Future Stock Price Predictions", color='black', fontsize=14)
    plt.xlabel("Days into the Future", color='black', fontsize=10)
    plt.ylabel("Predicted Stock Price", color='black', fontsize=10)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5)
    plot_path = 'static/predictions_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def split_sequence(sequence, window):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + window
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def generate_train_test_plot(actual_prices, predicted_prices):
    plt.figure(figsize=(16, 8), facecolor='lightblue')
    ax = plt.subplot(111)
    ax.set_facecolor('lightseagreen')
    ax.plot(actual_prices, color='black', linewidth=2, label="Actual")
    ax.plot(range(len(actual_prices)-len(predicted_prices), len(actual_prices)), predicted_prices, color='red', linewidth=2, label="Predicted")
    leg = plt.legend(facecolor='lightgrey')
    for text in leg.get_texts():
        text.set_color('black')
    plt.title("Stock Price Prediction", color='black', fontsize=14)
    plt.xlabel("Days from Start of Training Data (9 years today)", color='black', fontsize=10)
    plt.ylabel("Stock Price", color='black', fontsize=10)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5)
    if not os.path.exists('static'):
        os.makedirs('static')
    plot_path = 'static/train_test_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

if __name__ == '__main__':
    app.run(debug=True)
