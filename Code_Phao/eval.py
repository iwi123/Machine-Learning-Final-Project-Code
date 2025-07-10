import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
def evaluate_predictions(true, pred, horizon):
    """评估预测结果"""
    if len(true) < horizon:
        # 填充不足的部分
        padded_true = np.pad(true, (0, horizon - len(true)), 'constant', constant_values=np.nan)
        valid_mask = ~np.isnan(padded_true)
        mse = mean_squared_error(padded_true[valid_mask], pred[valid_mask])
        mae = mean_absolute_error(padded_true[valid_mask], pred[valid_mask])
        return mse, mae
    return mean_squared_error(true[:horizon], pred[:horizon]), mean_absolute_error(true[:horizon], pred[:horizon])