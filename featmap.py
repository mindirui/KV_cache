import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. 配置 (请根据你的实际情况修改这里的路径) ---

# 包含你的三组结果的父文件夹路径
# 假设你的文件夹结构是:
# /path/to/results/
#   ├── baseline/
#   │   ├── _step_000_pixel.png
#   │   └── ...
#   ├── high_quality/
#   │   ├── _step_000_pixel.png
#   │   └── ...
#   └── my_improvement/
#       ├── _step_000_pixel.png
#       └── ...

# 基线版本结果的文件夹
BASELINE_FOLDER = './1peranalyze/baseline/cache_interlatent'
# 高质量版本结果的文件夹
HQ_FOLDER = './1peranalyze/2out_with24/2out_interlatent'
# 你的改良版本结果的文件夹
MY_METHOD_FOLDER = './1peranalyze/cache_with24/cache_interlatent'

# 保存生成的热力图的输出文件夹
OUTPUT_FOLDER = './heatmaps_output'

# --- 代码正文 (通常无需修改以下部分) ---


def find_step_number(filename):
    """从文件名中提取步骤编号 (例如从 '_step_007_pixel_0.png' 提取 '007')"""
    match = re.search(r'_step_(\d+)_', filename)
    if match:
        return match.group(1)
    return None

def find_file_for_step(folder, step_number):
    """在指定文件夹中查找对应步骤的文件"""
    # 使用更鲁棒的查找，避免因文件名后缀不同导致匹配失败
    pattern = f"step_{step_number}_"
    for f in os.listdir(folder):
        if pattern in f and f.endswith('.png'):
            return os.path.join(folder, f)
    return None

def generate_heatmap(image_path1, image_path2, title, output_path):
    """
    计算两张图片的差异并生成一张热力图。
    """
    try:
        # 以灰度模式加载图片并转为numpy数组
        img1 = Image.open(image_path1).convert('L')
        img2 = Image.open(image_path2).convert('L')

        # 确保两张图片尺寸一致
        if img1.size != img2.size:
            print(f"  [警告] 图像尺寸不匹配，跳过: {os.path.basename(image_path1)} 和 {os.path.basename(image_path2)}")
            return

        # 转换为浮点数以进行计算
        img1_np = np.array(img1, dtype=np.float32)
        img2_np = np.array(img2, dtype=np.float32)

        # 计算绝对差异
        difference = np.abs(img1_np - img2_np)

        # 绘图
        fig, ax = plt.subplots()
        im = ax.imshow(difference, cmap='inferno')
        fig.colorbar(im, ax=ax, label='Pixel Value Absolute Difference')
        ax.set_title(title)
        ax.axis('off')

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    except FileNotFoundError:
        print(f"  [错误] 无法找到文件进行比较: {image_path1} 或 {image_path2}")
    except Exception as e:
        print(f"  [错误] 生成热力图 '{title}' 时发生未知错误: {e}")


def main():
    """
    主执行函数
    """
    # 为三组热力图创建输出文件夹
    output_folder_A = os.path.join(OUTPUT_FOLDER, 'A_HQ_vs_Baseline')
    output_folder_B = os.path.join(OUTPUT_FOLDER, 'B_MyMethod_vs_Baseline')
    output_folder_C = os.path.join(OUTPUT_FOLDER, 'C_MyMethod_vs_HQ (Error)')
    os.makedirs(output_folder_A, exist_ok=True)
    os.makedirs(output_folder_B, exist_ok=True)
    os.makedirs(output_folder_C, exist_ok=True)

    # 检查输入文件夹是否存在
    for folder in [BASELINE_FOLDER, HQ_FOLDER, MY_METHOD_FOLDER]:
        if not os.path.isdir(folder):
            print(f"[致命错误] 文件夹不存在: {folder}")
            return

    print("开始处理三组热力图生成...")
    # 以高质量文件夹为基准，遍历所有文件
    hq_files = sorted(os.listdir(HQ_FOLDER))
    processed_steps = set()

    for hq_filename in hq_files:
        if not hq_filename.endswith('.png'):
            continue

        step_num = find_step_number(hq_filename)
        if step_num is None or step_num in processed_steps:
            continue

        print(f"处理步骤: {step_num}")
        processed_steps.add(step_num)

        # 查找所有三个文件夹中的对应文件
        hq_path = os.path.join(HQ_FOLDER, hq_filename)
        baseline_path = find_file_for_step(BASELINE_FOLDER, step_num)
        my_method_path = find_file_for_step(MY_METHOD_FOLDER, step_num)

        # 确保三个文件都找到了
        if not all([baseline_path, hq_path, my_method_path]):
            print(f"  [跳过] 步骤 {step_num} 的文件不完整，请检查文件夹。")
            continue

        # 1. 生成 "引导目标" 热力图 (HQ vs Baseline)
        output_path_A = os.path.join(output_folder_A, f"step_{step_num}_hq_vs_baseline.png")
        generate_heatmap(hq_path, baseline_path, f'Guidance Target (HQ vs Baseline)\nStep {step_num}', output_path_A)

        # 2. 生成 "实际改动" 热力图 (MyMethod vs Baseline)
        output_path_B = os.path.join(output_folder_B, f"step_{step_num}_mymethod_vs_baseline.png")
        generate_heatmap(my_method_path, baseline_path, f'Actual Change (My Method vs Baseline)\nStep {step_num}', output_path_B)

        # 3. 生成 "最终误差" 热力图 (MyMethod vs HQ)
        output_path_C = os.path.join(output_folder_C, f"step_{step_num}_mymethod_vs_hq.png")
        generate_heatmap(my_method_path, hq_path, f'Final Error (My Method vs HQ)\nStep {step_num}', output_path_C)

    print("\n处理完成！")
    print(f"三组热力图已保存到以下文件夹中:")
    print(f"1. 引导目标: {output_folder_A}")
    print(f"2. 实际改动: {output_folder_B}")
    print(f"3. 最终误差: {output_folder_C}")


if __name__ == '__main__':
    main()