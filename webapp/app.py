from flask import Flask, render_template, request, jsonify
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Define your machine learning model
def create_model():
    set_seed(455)
    np.random.seed(455)
    model = Sequential([
        GRU(units=125, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),  
        GRU(units=125),  
        Dropout(0.2),  
        Dense(units=1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Load your trained model
model = create_model()

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('form.html')
    elif request.method == 'POST':
        # Get user input for stock symbol
        stock_symbol = request.form['stock_symbol']
        # Fetch stock price data for the specified stock symbol
        dataset = yf.download(stock_symbol, start='2015-01-01', end='2022-01-01')
        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        training_set = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))
        window_size = 60
        X_test, y_test = split_sequence(training_set, window_size)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        # Make predictions using your model
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price).flatten()
        actual_stock_price = scaler.inverse_transform(y_test).flatten()
        # Generate plots
        train_test_plot_path = generate_train_test_plot(actual_stock_price, predicted_stock_price)
        # Return path to the saved image as JSON response
        return render_template('results.html', stock_symbol=stock_symbol, train_test_plot_path=train_test_plot_path)

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
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, color='black', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plot_path = 'static/train_test_plot.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path

if __name__ == '__main__':
    app.run(debug=True)
