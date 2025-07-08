import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_utils import prepare_sequences, recursive_forecast
from config import SHORT_TERM_STEPS, LONG_TERM_STEPS, TARGET_COL, FEATURE_COLS

def create_lstm_model(input_shape):
    """创建LSTM模型"""
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(X_train, y_train, initial_sequence, test_features, scaler, target_idx, forecast_steps):
    """训练和评估LSTM模型"""
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    predictions = recursive_forecast(
        model, initial_sequence, 
        n_steps=forecast_steps,
        scaler=scaler,
        target_idx=target_idx,
        future_features=test_features[:forecast_steps]
    )
    
    return model, predictions