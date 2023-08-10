import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import v20
from fx_signals.common import config

# Function to create LSTM dataset
def create_lstm_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Main function to forecast FX prices using LSTM with TensorFlow
def forecast_fx_prices(df):
    # Normalize the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_fx_data = scaler.fit_transform(df[['close']])

    # Define the look-back window for creating LSTM dataset
    look_back = 100

    # Create LSTM dataset
    X, y = create_lstm_dataset(normalized_fx_data, look_back)

    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input data to [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print('Building the model...')
    # Build the LSTM model 
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])

    print('Compiling the model...')
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.0001))

    print('Training the model...')
    # Train the model
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

    print('Testing the model...')
    # Make predictions on the test data
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Prepare the test data index for plotting
    test_index = df.index[train_size + look_back:]

    # Plot the actual and predicted FX prices
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'].loc[test_index], label='Actual Prices')
    plt.plot(test_index, test_predictions, label='Predicted Prices')
    plt.legend()
    plt.title('FX Price Forecast using LSTM (TensorFlow)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
def get_df(api: v20.Context, instrument: str, granularity: str) -> pd.DataFrame:
    res = api.instrument.candles(
        instrument=instrument,
        granularity=granularity,
        count=500
    )
    
    headers = [
        'time',
        'open',
        'high',
        'low',
        'close',
    ]
    
    data = []

    candle: v20.instrument.Candlestick
    for candle in res.get("candles", 200):
        data.append(
            [
                candle.time,
                candle.mid.o,
                candle.mid.h,
                candle.mid.l,
                candle.mid.c,
            ]
        )
    
    df = pd.DataFrame(data, columns=headers)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()

    config.add_argument(parser)

    args = parser.parse_args()
    api = args.config.create_context()
    
    df = get_df(api, 'EUR_USD', 'H1')
    forecast_fx_prices(df)

if __name__ == "__main__":
    main()