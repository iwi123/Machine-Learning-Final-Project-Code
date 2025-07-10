import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from config import SHORT_TERM_STEPS, LONG_TERM_STEPS, TARGET_COL, FEATURE_COLS,NUM_EXPERIMENTS 
def save_results_and_visualize(results, train, test, predictions_dict):
    """保存结果并生成可视化图表"""
    
    # 1. 创建结果汇总表
    summary = pd.DataFrame()
    models = ['LSTM','Transformer','Ours']
    
    for model in models:
        for forecast_type in ['short', 'long']:
            key = f"{model}_{forecast_type}"
            metrics = results[key]
            
            # 计算均值和标准差
            mse_mean = np.mean(metrics['mse'])
            mse_std = np.std(metrics['mse'])
            mae_mean = np.mean(metrics['mae'])
            mae_std = np.std(metrics['mae'])
            
            # 添加到汇总表
            new_row = pd.DataFrame({
                'Model': [model],
                'Forecast Type': ['Short Term (90 days)' if forecast_type == 'short' else 'Long Term (365 days)'],
                'MSE Mean': [mse_mean],
                'MSE Std': [mse_std],
                'MAE Mean': [mae_mean],
                'MAE Std': [mae_std]
            })
            summary = pd.concat([summary, new_row], ignore_index=True)
    
    # 保存结果到CSV
    summary.to_csv('prediction_results.csv', index=False)
    print("Prediction results saved to prediction_results.csv")
    
    # 2. 生成可视化图表
    plt.figure(figsize=(18, 24))
    plt.suptitle('Power Consumption Forecast Comparison', fontsize=16, y=0.98)
    
    # 获取第一轮实验的预测结果用于可视化
    exp_idx = 0
    
    # ========== 短期预测对比图 (90天) ==========
    ax1 = plt.subplot(4, 1, 1)
    days = np.arange(1, 91)
    true_short = test[TARGET_COL].values[:90]
    
    # 真实值 - 点折线图
    ax1.plot(days, true_short, 'b-o', label='Actual Value', markersize=4, linewidth=1.5)
    
    # 预测值 - 三种模型对比
    colors = ['g', 'r', 'm']
    markers = ['s', '^', 'D']
    for i, model in enumerate(models):
        key = f"{model}_short"
        if key in predictions_dict:
            pred = predictions_dict[key][exp_idx][:90]
            ax1.plot(days, pred, f'{colors[i]}-{markers[i]}', 
                    label=f'{model} Prediction', markersize=4, linewidth=1.5)
    
    ax1.set_title('Short-Term Forecast (90 Days) - Active Power Comparison', fontsize=14)
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Active Power (kW)', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(np.arange(0, 91, 10))
    
    # ========== 完整长期预测对比图 (365天) ==========
    ax2 = plt.subplot(4, 1, 2)  # 原长期前90天位置替换为完整长期图
    days_long = np.arange(1, 366)  # 365天
    
    # 真实值 - 线图
    true_long = test[TARGET_COL].values[:365]  # 完整365天真实值
    ax2.plot(days_long, true_long, 'b-', label='Actual Value', linewidth=1.5)
    
    # 预测值 - 三种模型对比
    for i, model in enumerate(models):
        key = f"{model}_long"
        if key in predictions_dict:
            pred = predictions_dict[key][exp_idx][:365]  # 完整365天预测
            ax2.plot(days_long, pred, f'{colors[i]}-', 
                    label=f'{model} Prediction', linewidth=1.5)
    
    ax2.set_title('Long-Term Forecast (365 Days) - Active Power Comparison', fontsize=14)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_ylabel('Active Power (kW)', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(np.arange(0, 366, 30))  # 每30天一个刻度
    
    # ========== 误差对比图 (MSE) ==========
    ax3 = plt.subplot(4, 1, 3)
    bar_width = 0.25
    index = np.arange(len(models))
    
    # 提取短期和长期的MSE数据
    mse_short = [np.mean(results[f"{model}_short"]['mse']) for model in models]
    mse_short_err = [np.std(results[f"{model}_short"]['mse']) for model in models]
    
    mse_long = [np.mean(results[f"{model}_long"]['mse']) for model in models]
    mse_long_err = [np.std(results[f"{model}_long"]['mse']) for model in models]
    
    # 绘制柱状图
    rects1 = ax3.bar(index - bar_width/2, mse_short, bar_width, 
                    yerr=mse_short_err, capsize=5, 
                    label='Short-Term Forecast', color='skyblue')
    rects2 = ax3.bar(index + bar_width/2, mse_long, bar_width, 
                    yerr=mse_long_err, capsize=5, 
                    label='Long-Term Forecast', color='lightcoral')
    
    # 添加数据标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_title('Model Prediction Error Comparison (MSE)', fontsize=14)
    ax3.set_xticks(index)
    ax3.set_xticklabels(models, fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # ========== 误差对比图 (MAE) ==========
    ax4 = plt.subplot(4, 1, 4)
    # 提取短期和长期的MAE数据
    mae_short = [np.mean(results[f"{model}_short"]['mae']) for model in models]
    mae_short_err = [np.std(results[f"{model}_short"]['mae']) for model in models]
    
    mae_long = [np.mean(results[f"{model}_long"]['mae']) for model in models]
    mae_long_err = [np.std(results[f"{model}_long"]['mae']) for model in models]
    
    # 绘制柱状图
    rects3 = ax4.bar(index - bar_width/2, mae_short, bar_width, 
                    yerr=mae_short_err, capsize=5, 
                    label='Short-Term Forecast', color='skyblue')
    rects4 = ax4.bar(index + bar_width/2, mae_long, bar_width, 
                    yerr=mae_long_err, capsize=5, 
                    label='Long-Term Forecast', color='lightcoral')
    
    # 添加数据标签
    autolabel(rects3)
    autolabel(rects4)
    
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('MAE', fontsize=12)
    ax4.set_title('Model Prediction Error Comparison (MAE)', fontsize=14)
    ax4.set_xticks(index)
    ax4.set_xticklabels(models, fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Prediction comparison chart saved as prediction_comparison.png")
    
    # 3. 生成实验报告
    report = f"""
    ================================
       电力消耗预测实验结果报告
    ================================
    
    实验设置:
    - 历史时间步长: {SHORT_TERM_STEPS} 天
    - 特征数量: {len(FEATURE_COLS)}
    - 实验次数: {NUM_EXPERIMENTS}
    - 训练数据天数: {len(train)}
    - 测试数据天数: {len(test)}
    
    模型架构:
    - LSTM: 3层LSTM(128,64,32) + Dropout
    - Transformer: 4个Transformer块 + 多头注意力机制
    - Custom: 膨胀卷积 + 门控机制 + BiGRU + 自注意力 混合架构
    
    结果摘要:
    {summary.to_string(index=False)}
    
    可视化:
    - 预测对比图已保存为 prediction_comparison.png
    
    ================================
    报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================
    """
    
    print(report)
    with open('experiment_report.txt', 'w') as f:
        f.write(report)
    print("实验报告已保存为 experiment_report.txt")