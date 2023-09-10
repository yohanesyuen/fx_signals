import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import os
import pickle
import v20
from fx_signals.common import config

lookback = 200

def train_model(data, target_col, extra_cols=[], epochs=100):
    # Prepare the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[target_col] + extra_cols].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=32)
    
    return model, scaler

def test_model(model, scaler, data, target_col, extra_cols=[]):
    scaled_data = scaler.transform(data[[target_col] + extra_cols].values)
    
    X_test, y_test = [], []
    for i in range(lookback, len(scaled_data)):
        X_test.append(scaled_data[i-lookback:i])
        y_test.append(scaled_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, predictions)
    
    return mse

def predict_next_day(model, scaler, data, target_col, extra_cols=[]):
    last_values = data[[target_col] + extra_cols].values[-lookback:]
    last_values_scaled = scaler.transform(last_values)
    
    # Reshape for LSTM input
    last_values_scaled = np.reshape(last_values_scaled, (1, lookback, len(extra_cols) + 1))
    
    next_day_prediction_scaled = model.predict(last_values_scaled)
    
    # Prepare the shape for inverse_transform
    dummy_array = np.zeros((1, len(extra_cols) + 1))
    dummy_array[0, 0] = next_day_prediction_scaled
    
    next_day_prediction = scaler.inverse_transform(dummy_array)
    
    return next_day_prediction[0, 0]
    
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
    
    df = get_df(api, 'EUR_USD', 'D')
    df[f'{lookback}MA'] = df['close'].rolling(lookback).mean()
    df.dropna(inplace=True)
    model, scaler = train_model(df, 'close', extra_cols=[f'{lookback}MA'], epochs=100)
    mse = test_model(model, scaler, df, 'close', extra_cols=[f'{lookback}MA'])
    print(f'MSE: {mse}')
    next_day_prediction = predict_next_day(model, scaler, df, 'close', extra_cols=[f'{lookback}MA'])
    print(f'Next day prediction: {next_day_prediction}')

if __name__ == "__main__":
    main()