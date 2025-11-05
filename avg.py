import json

# 请将 'data.json' 替换为您的JSON文件名
file_name = './experimental_outviews/unnamed_experiment_20250908-194840/run_logs.json'

# 从文件中加载JSON数据
with open(file_name, 'r') as f:
    data = json.load(f)

# 初始化用于存储指标总和与计数的字典
metrics_sum = {
    "loss": 0,
    "lpips": 0,
    "ssim": 0,
    "psnr": 0
}
count = 0

# 遍历结果并计算总和与计数
for result in data['results']:
    metrics = result['per_view_details'][0]['metrics']
    for key in metrics_sum:
        if key in metrics:
            metrics_sum[key] += metrics[key]
    count += 1

# 计算每个指标的平均值
metrics_avg = {key: value / count for key, value in metrics_sum.items()}

# 打印结果
print("所有指标的均值 (Average of all metrics):")
for key, value in metrics_avg.items():
    print(f"{key}: {value}")