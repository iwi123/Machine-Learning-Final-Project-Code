import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import COLUMN_NAMES, FEATURE_COLS, TARGET_COL

def load_data(file, is_test=False):
    """加载并处理数据集"""
    try:
        na_values = ['?', 'NaN', 'nan', 'na', 'NA', 'N/A', '', ' ']
        if is_test:
            df = pd.read_csv(file, header=None, names=COLUMN_NAMES, na_values=na_values)
        else:
            df = pd.read_csv(file, header=0, names=COLUMN_NAMES, na_values=na_values)
    except Exception as e:
        raise ValueError(f"读取文件错误: {e}")

    try:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    except Exception as e:
        raise ValueError(f"日期时间转换错误: {e}")
    
    # 处理缺失值
    for col in COLUMN_NAMES[1:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                df[col] = 0
            else:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    df = df.dropna(subset=['DateTime'])
    df['Date'] = df['DateTime'].dt.date
    
    # 按天聚合数据
    daily = df.groupby('Date').agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    })
    
    return daily

def engineer_features(df):
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # 单位转换：RR列转换为毫米
    df['RR'] = df['RR'] / 10.0
    
    # 计算剩余能耗
    df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - \
                                  (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    
    # 添加时间特征
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
    except Exception as e:
        print(f"添加时间特征错误: {e}")
    
    # 确保所有列都是数值类型
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # 处理缺失值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df[FEATURE_COLS]

def prepare_sequences(data, target_idx, n_steps):
    """准备时间序列数据"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps, target_idx])
    return np.array(X), np.array(y)

def get_scaler():
    return MinMaxScaler(feature_range=(0, 1))


def recursive_forecast(model, initial_sequence, n_steps, scaler, target_idx, future_features=None):

    current_sequence = initial_sequence.copy()
    predictions = []
    n_features = initial_sequence.shape[2]  # 特征数量

    for step in range(n_steps):
        # 预测下一步的有功功率
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # 创建新特征向量
        new_features = np.zeros((1, n_features))
        
        # 设置预测的有功功率
        new_features[0, target_idx] = next_pred
        
        # 如果有提供未来特征数据，则使用
        if future_features is not None and step < future_features.shape[0]:
            # 复制所有特征，除了目标变量
            for i in range(n_features):
                if i != target_idx:
                    new_features[0, i] = future_features[step, i]
        
        # 更新序列
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1] = new_features[0]
    
    # 反归一化预测结果
    predictions = np.array(predictions).reshape(-1, 1)
    # 创建虚拟数组用于反归一化
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_idx] = predictions[:, 0]
    predictions = scaler.inverse_transform(dummy)[:, target_idx]
    
    return predictions