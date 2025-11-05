import os
import re
import numpy as np
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# --- 1. 配置 (请根据你的实际情况修改) ---

# 高质量版本结果的文件夹
HQ_FOLDER = './1peranalyze/2out_with24/2out_interlatent'
# 你的改良版本结果的文件夹
MY_METHOD_FOLDER = './1peranalyze/cache_with24/cache_interlatent'

# --- 代码正文 ---

def find_step_from_filename(filename):
    """从文件名中提取步骤编号"""
    match = re.search(r'_step_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    # --- 阶段1: 缓存“我的方法”的latent特征 ---
    print("\n--- 阶段1: 正在缓存 'My Method' 的 latent 特征... ---")
    my_method_features = {}
    my_method_files = [f for f in os.listdir(MY_METHOD_FOLDER) if f.endswith('_latent.pt')]
    my_method_files.sort(key=lambda x: find_step_from_filename(x) or -1, reverse=True)

    for filename in my_method_files:
        step = find_step_from_filename(filename)
        if step is None: continue
        
        filepath = os.path.join(MY_METHOD_FOLDER, filename)
        try:
            # 加载Tensor，其形状应为 [1, C, H, W]
            latent_tensor = torch.load(filepath, map_location=device)
            # 提取第0个元素，得到 [C, H, W] 的目标视图latent
            target_latent = latent_tensor[0]
            # 展平为特征向量
            features = target_latent.flatten().cpu().numpy()
            my_method_features[step] = features
        except Exception as e:
            print(f"  [错误] 加载或处理文件 {filename} 时出错: {e}")
    
    print(f"缓存完成！共缓存了 {len(my_method_features)} 个步骤的特征。")

    # --- 阶段2: 遍历HQ步骤，寻找最佳匹配 ---
    print("\n--- 阶段2: 正在为 'HQ' 的每一步寻找最佳匹配... ---")
    results = []
    hq_files = [f for f in os.listdir(HQ_FOLDER) if f.endswith('_latent.pt')]
    hq_files.sort(key=lambda x: find_step_from_filename(x) or -1, reverse=True)

    for filename in hq_files:
        hq_step = find_step_from_filename(filename)
        if hq_step is None: continue

        filepath = os.path.join(HQ_FOLDER, filename)
        try:
            # 加载Tensor，其形状应为 [2, C, H, W]
            latent_tensor = torch.load(filepath, map_location=device)
            # 提取第0个元素，得到 [C, H, W] 的目标视图latent
            target_latent = latent_tensor[0]
            
            hq_features = target_latent.flatten().cpu().numpy()

            best_match_step = -1
            min_distance = float('inf')

            for my_step, my_features in my_method_features.items():
                distance = cosine(hq_features, my_features)
                if distance < min_distance:
                    min_distance = distance
                    best_match_step = my_step
            
            print(f"HQ 步骤: {hq_step:2d} -> 最匹配的 My Method 步骤: {best_match_step:2d} (Latent 距离: {min_distance:.4f})")
            results.append((hq_step, best_match_step))

        except Exception as e:
            print(f"  [错误] 加载或处理HQ文件 {filename} 时出错: {e}")

    # --- 阶段3: 可视化结果 ---
    if results:
        print("\n--- 阶段3: 正在生成进度对齐可视化图... ---")
        hq_steps = [r[0] for r in results]
        my_method_steps = [r[1] for r in results]

        plt.figure(figsize=(10, 8))
        plt.plot(hq_steps, my_method_steps, 'o-', label='实际进度匹配 (基于Latent)')
        plt.plot(hq_steps, hq_steps, 'r--', label='理想对齐线 (y=x)')
        
        plt.title('去噪进度对齐验证 (直接比较Latent)')
        plt.xlabel('HQ 方法的时间步 (t_hq)')
        plt.ylabel('与之Latent最匹配的 My Method 时间步 (t_my)')
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        
        output_filename = 'denoising_progress_alignment_final_shape_adjusted.png'
        plt.savefig(output_filename)
        print(f"可视化图已保存为: {output_filename}")


if __name__ == '__main__':
    main()