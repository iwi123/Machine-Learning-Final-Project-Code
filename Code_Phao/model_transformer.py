import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from data_utils import prepare_sequences, recursive_forecast
from config import SHORT_TERM_STEPS, LONG_TERM_STEPS, TARGET_COL, FEATURE_COLS

def create_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128]):
    """创建Transformer模型"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(x)
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([x, attn_output])
        
        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(x)
        ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(input_shape[-1]),
        ])
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
    
    # Final layers
    x = GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(0.2)(x)
    
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(X_train, y_train, initial_sequence, test_features, scaler, target_idx, forecast_steps):
    """训练和评估Transformer模型"""
    model = create_transformer_model((X_train.shape[1], X_train.shape[2]))
    
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