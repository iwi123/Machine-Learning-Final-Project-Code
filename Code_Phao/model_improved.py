import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Conv1D
from tensorflow.keras.layers import SeparableConv1D, Add, Activation, Multiply
from tensorflow.keras.callbacks import EarlyStopping
from data_utils import prepare_sequences, recursive_forecast
from tensorflow.keras.optimizers import Adam

def create_improved_model(input_shape):
    """改进的创新模型：膨胀卷积 + 门控机制 + BiGRU + 自注意力"""
    inputs = Input(shape=input_shape)
    
    # 第一层：膨胀卷积提取多尺度特征
    conv1 = SeparableConv1D(filters=64, kernel_size=3, padding='causal', 
                           dilation_rate=1, activation='relu')(inputs)
    conv1 = LayerNormalization()(conv1)
    
    conv2 = SeparableConv1D(filters=64, kernel_size=3, padding='causal', 
                           dilation_rate=2, activation='relu')(conv1)
    conv2 = LayerNormalization()(conv2)
    
    # 残差连接 + 门控机制
    shortcut = Conv1D(filters=64, kernel_size=1, padding='same')(inputs)
    gate = Conv1D(filters=64, kernel_size=1, padding='same', activation='sigmoid')(conv2)
    gated_conv = Multiply()([conv2, gate])
    res_out = Add()([shortcut, gated_conv])
    res_out = Activation('relu')(res_out)
    
    # 第二层：双向GRU捕获序列依赖
    gru_out = Bidirectional(GRU(128, return_sequences=True))(res_out)
    gru_out = Dropout(0.3)(gru_out)
    gru_out = LayerNormalization()(gru_out)
    
    # 第三层：门控自注意力机制
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(gru_out, gru_out)
    # 注意力门控
    attn_gate = Conv1D(filters=attn_output.shape[-1], kernel_size=1, 
                      activation='sigmoid')(gru_out)
    gated_attn = Multiply()([attn_output, attn_gate])
    # 残差连接
    attn_res = Add()([gru_out, gated_attn])
    attn_res = LayerNormalization()(attn_res)
    
    # 第四层：时间维度特征压缩
    compressed = GRU(64, activation='relu')(attn_res)
    compressed = Dropout(0.2)(compressed)
    
    # 输出层
    outputs = Dense(1)(compressed)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse',
                  metrics=['mae'])
    return model

def train_and_evaluate(X_train, y_train, initial_sequence, test_features, scaler, target_idx, forecast_steps):
    # 创建改进的创新模型
    model = create_improved_model((X_train.shape[1], X_train.shape[2]))
    
    # 训练流程保持不变
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 递归预测保持不变
    predictions = recursive_forecast(
        model, initial_sequence, 
        n_steps=forecast_steps,
        scaler=scaler,
        target_idx=target_idx,
        future_features=test_features[:forecast_steps]
    )
    
    return model, predictions