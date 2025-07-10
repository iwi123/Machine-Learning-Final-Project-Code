import os
import numpy as np
import pandas as pd
from data_utils import load_data, engineer_features, get_scaler, prepare_sequences
from model_lstm import train_and_evaluate as train_lstm
from model_transformer import train_and_evaluate as train_transformer
from model_improved import train_and_evaluate as train_improved
from eval import evaluate_predictions
from utils import save_results_and_visualize
from config import *
import tensorflow as tf
# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

def main():
    
    # 加载数据

    train = load_data('train.csv')
    test = load_data('test.csv', is_test=True)
    
    print(f"训练数据天数: {len(train)}")
    print(f"测试数据天数: {len(test)}")
    
    train = engineer_features(train)
    test = engineer_features(test)
    
    # 数据归一化
    scaler = get_scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # 准备序列数据
    target_idx = FEATURE_COLS.index(TARGET_COL)
    X_train, y_train = prepare_sequences(train_scaled, target_idx, SHORT_TERM_STEPS)
    initial_sequence = train_scaled[-SHORT_TERM_STEPS:].reshape(1, SHORT_TERM_STEPS, len(FEATURE_COLS))
    
    # 5. 训练和评估模型
    model_types = ['LSTM','Transformer','Ours']
    forecast_types = ['short', 'long']

    results = {}
    predictions_dict = {}  # 存储每轮实验的预测结果
    
    for model_name in model_types:
        for forecast_type in forecast_types:
            key = f"{model_name}_{forecast_type}"
            results[key] = {'mse': [], 'mae': []}
            predictions_dict[key] = []  # 存储每轮实验的预测结果
            
            forecast_steps = SHORT_TERM_STEPS if forecast_type == 'short' else LONG_TERM_STEPS
            
            for exp in range(NUM_EXPERIMENTS):
                print(f"\n=== {model_name} {forecast_type}预测 - 实验 #{exp+1}/{NUM_EXPERIMENTS} ===")
                
                if model_name == 'LSTM':
                    _, pred = train_lstm(
                        X_train, y_train, initial_sequence,
                        test_scaled, scaler, target_idx, forecast_steps
                    )
                elif model_name == 'Transformer':
                    _, pred = train_transformer(
                        X_train, y_train, initial_sequence,
                        test_scaled, scaler, target_idx, forecast_steps
                    )
                else:  
                    _, pred = train_improved(
                        X_train, y_train, initial_sequence,
                        test_scaled, scaler, target_idx, forecast_steps
                    )
                
                # 存储预测结果
                predictions_dict[key].append(pred)
                
                # 评估预测结果
                true_target = test[TARGET_COL].values
                mse, mae = evaluate_predictions(true_target, pred, forecast_steps)
                
                results[key]['mse'].append(mse)
                results[key]['mae'].append(mae)
                
                print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # 6. 保存结果并生成可视化
    save_results_and_visualize(results,train, test, predictions_dict)
    
    print("done")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()