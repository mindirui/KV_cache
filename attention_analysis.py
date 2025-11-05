import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import collections
import os
import csv
# ===================================================================
#                      辅助函数 (用于计算指标)
# ===================================================================

def calculate_attention_entropy(attention_vector: torch.Tensor) -> float:
    """
    计算一个注意力分布向量的香农熵。
    熵越高，注意力越分散，不确定性越大。
    """
    # 使用softmax确保权重是概率分布（总和为1）
    vector_float32 = attention_vector.to(torch.float32)
    
    # 使用更高精度的张量进行softmax计算
    p = vector_float32
    
    # 计算熵: H(p) = - sum(p_i * log2(p_i))
    log_p = torch.log2(p + 1e-12)
    entropy = -torch.sum(p * log_p)
    
    return entropy.item()

def calculate_attention_stability_l2(vec_t: torch.Tensor, vec_t_minus_1: torch.Tensor) -> float:
    """
    计算两个注意力向量之间的L2距离，衡量不稳定性（变化/漂移）。
    值越大，表示从上一步到这一步，注意力策略变化越大。
    """
    p = vec_t.flatten()
    q = vec_t_minus_1.flatten()
    
    drift = torch.linalg.norm(p - q)
    return drift.item()

# ===================================================================
#                      核心分析函数
# ===================================================================

def analyze_token_dynamics(
    attention_data: Dict,
    module_name: str,
    query_point: Tuple[int, int],
    query_frame_index: int,
    latent_res_q: int,
    num_frames: int
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    分析指定token在所有去噪步骤中的注意力动态，并返回数值结果。

    Args:
        attention_data (Dict): 已加载的、包含所有注意力图的字典。
        module_name (str): 要分析的注意力模块名称。
        query_point (Tuple[int, int]): token在latent空间中的(x, y)坐标。
        query_frame_index (int): token所在的视图/帧的索引。
        latent_res_q (int): latent空间的分辨率。
        num_frames (int): 总的视图/帧数。

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: 
        一个包含(时间步: 熵)的字典，和一个包含(时间步: 稳定性)的字典。
    """
    
    entropy_results = collections.OrderedDict()
    stability_results = collections.OrderedDict()
    
    sorted_steps = sorted(attention_data.keys(), reverse=True)
    print(f"开始分析... 共找到 {len(sorted_steps)} 个时间步。")

    prev_attention_vector = None
    tokens_per_frame = latent_res_q * latent_res_q
    
    for timestep in sorted_steps:
        # ---- 完全按照您提供的逻辑提取注意力向量 ----
        
        full_attn_map = attention_data[timestep][module_name]
        
        # 1. 选择批次中的一项 (索引为1)
        self_attn_map_item = full_attn_map[1]
        
        # 2. 计算全局索引
        qx, qy = query_point
        query_idx_in_frame = qy * latent_res_q + qx
        global_query_idx = query_frame_index * tokens_per_frame + query_idx_in_frame
        
        # 3. 提取该查询点对所有像素的注意力向量
        attention_vector = self_attn_map_item[global_query_idx, :]
        
        # ---- 计算并存储指标 ----
        
        # 计算注意力熵
        entropy = calculate_attention_entropy(attention_vector)
        entropy_results[timestep] = entropy
        
        # 计算注意力稳定性 (从第二个时间步开始)
        if prev_attention_vector is not None:
            stability = calculate_attention_stability_l2(attention_vector, prev_attention_vector)
            stability_results[timestep] = stability
        
        # 更新上一步的向量，为下一次循环做准备
        prev_attention_vector = attention_vector.clone()

    return entropy_results, stability_results

# ===================================================================
#                      主程序入口 (如何使用)
# ===================================================================

if __name__ == '__main__':
    
    # --- 1. 请在这里配置您的“材料” ---
    FILE_PATH = "attention_map/cache_attention_2out.pt"
    MODULE_TO_ANALYZE = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.self"
    
    # Token和视图的参数
    QUERY_POINT = (10, 6) 
    QUERY_FRAME_INDEX = 0
    
    # 模型和数据相关的参数
    LATENT_RESOLUTION_QUERY = 16
    # 根据文件名或您的实验设置，我们知道这是2个视角的输出
    # 如果您分析的是 cache_attention.pt (1-out)，请将此值改为 1
    num_self_attention_frames = 2 

    # --- 2. 加载数据 ---
    print(f"正在从 '{FILE_PATH}' 加载数据...")
    try:
        attention_data = torch.load(FILE_PATH, map_location='cpu')
    except FileNotFoundError:
        print(f"错误: 文件未找到! 请确保 '{FILE_PATH}' 路径正确。")
        exit()
    print("数据加载成功！")

    # --- 3. 调用核心函数进行分析 ---
    entropy_data, stability_data = analyze_token_dynamics(
        attention_data=attention_data,
        module_name=MODULE_TO_ANALYZE,
        query_point=QUERY_POINT,
        query_frame_index=QUERY_FRAME_INDEX,
        latent_res_q=LATENT_RESOLUTION_QUERY,
        num_frames=num_self_attention_frames
    )
    
    # --- 4. 打印并输出您需要的“一串参数” ---
    print("\n" + "="*50)
    print("分析结果:")
    print(f"模块: {MODULE_TO_ANALYZE}")
    print(f"查询Token: {QUERY_POINT} @ 视图索引: {QUERY_FRAME_INDEX}")
    print("-"*50)
    
    print(f"{'Timestep':<10} | {'Attention Entropy':<20} | {'Attention Stability (Change)':<30}")
    print(f"{'-'*8:<10} | {'-'*18:<20} | {'-'*28:<30}")
    
    for timestep in entropy_data.keys():
        entropy_val = f"{entropy_data[timestep]:.4f}"
        # 稳定性数据比熵数据少一个（因为第一步无法计算）
        stability_val = f"{stability_data.get(timestep, 'N/A'):.4f}" if timestep in stability_data else "N/A (第一步)"
        print(f"{timestep:<10} | {entropy_val:<20} | {stability_val:<30}")

    print("="*50)
    
     # --- 5. (新增) 将结果保存到 CSV 文件 ---
    print("\n正在将结果保存到 CSV 文件...")
    
    # 创建一个动态、信息丰富的文件名
    # 例如: analysis_2out_q_8-6_frame_0.csv
    
    base_name = os.path.basename(FILE_PATH).replace('cache_attention_', '').replace('.pt', '')
    query_point_str = f"{QUERY_POINT[0]}-{QUERY_POINT[1]}"
    output_filename = f"analysis_{base_name}_q_{query_point_str}_frame_{QUERY_FRAME_INDEX}.csv"

    # 确保输出目录存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, output_filename)

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 创建 CSV写入器
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow(['Timestep', 'Attention_Entropy', 'Attention_Stability'])
            
            # 逐行写入数据
            for timestep in entropy_data.keys():
                entropy_val = entropy_data[timestep]
                # 使用 .get() 安全地获取稳定性数据，如果不存在则留空
                stability_val = stability_data.get(timestep, '')
                writer.writerow([timestep, f"{entropy_val:.6f}", f"{stability_val:.6f}" if stability_val != '' else ''])
                
        print(f"✅ 分析结果已成功保存至: {output_csv_path}")

    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")