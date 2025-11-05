import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from typing import Dict, Any

# ===================================================================
#                      绘图配置 (请在这里修改)
# ===================================================================

# 定义两个实验的配置文件
# 您只需修改这里的 "filepath" 和 "label" 即可轻松对比不同的实验结果
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "Experiment_1": {
        "filepath": "analysis_results/analysis_cache_attention_q_8-8_frame_0.csv", # 假设这是1-out的数据
        "label": "1-View Generation (1-out)",
        "color": "dodgerblue",
        "linestyle": "--"
    },
    "Experiment_2": {
        "filepath": "analysis_results/analysis_2out_q_8-8_frame_0.csv", # 假设这是2-out的数据
        "label": "2-View Generation (2-out)",
        "color": "orangered",
        "linestyle": "-"
    }
}

# 图表和保存相关的设置
FIGURE_TITLE = "Comparison of Attention Dynamics for Token (8, 8) in Frame 0"
OUTPUT_FILENAME = "attention_dynamics_comparison.png"

# ===================================================================
#                      数据加载与绘图函数
# ===================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """从CSV文件加载并准备数据。"""
    try:
        df = pd.read_csv(filepath)
        # 确保Timestep是数值类型并按降序排列
        df['Timestep'] = pd.to_numeric(df['Timestep'])
        df = df.sort_values(by='Timestep', ascending=False)
        return df
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到! 请检查路径: '{filepath}'")
        return None
    except Exception as e:
        print(f"❌ 读取文件时出错 '{filepath}': {e}")
        return None

def plot_dynamics_comparison(experiments_config: Dict, title: str, output_filename: str):
    """加载多个实验的数据并绘制对比图。"""
    
    # --- 加载数据 ---
    all_data = {}
    for name, config in experiments_config.items():
        df = load_data(config["filepath"])
        if df is not None:
            all_data[name] = df
    
    if len(all_data) < 2:
        print("未能加载足够的数据进行对比，程序退出。")
        return

    # --- 设置绘图风格 ---
    style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    ax1, ax2 = axes

    # --- 绘制子图1: 注意力熵 ---
    for name, df in all_data.items():
        config = experiments_config[name]
        ax1.plot(df['Timestep'], df['Attention_Entropy'], 
                 label=config['label'], 
                 color=config['color'],
                 linestyle=config['linestyle'],
                 linewidth=2.5)
    
    ax1.set_ylabel("Attention Entropy", fontsize=14)
    ax1.set_title("Attention Certainty (Entropy)", fontsize=16, pad=10)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    # --- 绘制子图2: 注意力稳定性 (变化) ---
    for name, df in all_data.items():
        config = experiments_config[name]
        # 忽略第一个时间步的空值
        df_stability = df.dropna(subset=['Attention_Stability'])
        ax2.plot(df_stability['Timestep'], df_stability['Attention_Stability'], 
                 label=config['label'], 
                 color=config['color'],
                 linestyle=config['linestyle'],
                 linewidth=2.5)

    ax2.set_ylabel("Attention Stability (Change)", fontsize=14)
    ax2.set_title("Attention Strategy Stability (Change between Steps)", fontsize=16, pad=10)
    ax2.set_xlabel("Denoising Timestep (t)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    # --- 全局设置 ---
    # 关键：反转X轴，使其从高时间步到低时间步，符合去噪逻辑
    ax1.invert_xaxis()
    
    fig.suptitle(title, fontsize=20, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # 为主标题留出空间
    
    # --- 保存和显示 ---
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ 图表已成功保存至: {output_filename}")
    except Exception as e:
        print(f"❌ 保存图表时出错: {e}")

    plt.show()


# ===================================================================
#                      主程序入口
# ===================================================================

if __name__ == '__main__':
    plot_dynamics_comparison(EXPERIMENTS, FIGURE_TITLE, OUTPUT_FILENAME)