# attention_visualization_overlay.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List
import cv2
import matplotlib.patches as patches

# =================================================================================
# 新增辅助函数：创建注意力叠加图
# =================================================================================

from PIL import Image

def extract_images_from_grid(
    combined_image_path: str, 
    num_rows: int,
    num_cols: int = 2
):
    """
    从一个网格状的拼接图中分割出所有独立的图像。

    此函数专门为以下布局设计：
    - 图像是一个包含 R 行 C 列的网格。
    - 每行代表一个独立的样本（例如，预测图和GT图）。
    - 不同的样本在垂直方向上堆叠。

    Args:
        combined_image_path (str): 网格拼接图的文件路径。
        num_rows (int): 网格中的行数（即垂直方向上有多少组图像）。
        num_cols (int): 网格中的列数（对于“预测+GT”的场景，此值通常为2）。

    Returns:
        一个嵌套的列表，结构为 [[row1_img1, row1_img2], [row2_img1, row2_img2], ...]。
        如果文件未找到，则返回 None。
    """
    try:
        combined_image = Image.open(combined_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 错误：拼接图文件未找到 {combined_image_path}。")
        return None

    total_width, total_height = combined_image.size
    
    # --- 检查尺寸是否能被整除 ---
    if total_width % num_cols != 0:
        print(f"⚠️ 警告：图像总宽度 {total_width} 无法被列数 {num_cols} 整除。分割可能不准确。")
    if total_height % num_rows != 0:
        print(f"⚠️ 警告：图像总高度 {total_height} 无法被行数 {num_rows} 整除。分割可能不准确。")
        
    # --- 计算每个小图的尺寸 ---
    panel_width = total_width // num_cols
    panel_height = total_height // num_rows
    
    all_images = []
    # --- 使用嵌套循环遍历网格 ---
    for r in range(num_rows):  # 遍历每一行
        row_images = []
        for c in range(num_cols):  # 遍历当前行中的每一列
            
            # --- 计算当前小图的裁剪坐标 ---
            left = c * panel_width
            top = r * panel_height
            right = (c + 1) * panel_width
            bottom = (r + 1) * panel_height
            
            panel = combined_image.crop((left, top, right, bottom))
            row_images.append(panel)
            
        all_images.append(row_images)
        
    return all_images

def create_overlay(image, heatmap_tensor, colormap='viridis', alpha=0.6):
    """
    创建一个将热力图叠加在原始图像上的图像。

    Args:
        image (PIL.Image.Image): 原始的高分辨率图像。
        heatmap_tensor (torch.Tensor): 单通道的、低分辨率的注意力热力图。
        colormap (str): 用于给热力图着色的 matplotlib 色谱。
        alpha (float): 热力图的透明度。

    Returns:
        PIL.Image.Image: 融合后的图像。
    """
    if not isinstance(image, Image.Image):
        raise TypeError("输入图像必须是 PIL.Image 对象。")
    if not isinstance(heatmap_tensor, torch.Tensor):
        raise TypeError("热力图必须是 PyTorch Tensor。")

    # 1. 归一化热力图到 [0, 1] 区间
    heatmap = heatmap_tensor.numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

    # 2. 应用色谱，将 (H, W) 的热力图转换为 (H, W, 3) 的彩色图
    cmap = plt.get_cmap(colormap)
    colored_heatmap = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_heatmap_pil = Image.fromarray(colored_heatmap)

    # 3. 将低分辨率的彩色热力图升采样到与原始图像相同的尺寸
    upsampled_heatmap = colored_heatmap_pil.resize(image.size, Image.BICUBIC)

    # 4. 阿尔法融合
    overlay = Image.blend(image, upsampled_heatmap, alpha=alpha)
    
    return overlay

# =================================================================================
# 更新版的可视化函数
# =================================================================================

def visualize_cross_attention_overlay(cross_attn_map, output_image, condition_images,
                                      t_in, latent_res_q, latent_res_k, query_point,
                                      timestep, module_name):
    """
    (最终版-带叠加) 可视化交叉注意力图。
    """
    if cross_attn_map is None: return
    cross_attn_map_item = cross_attn_map[1] # 选择有条件的batch
    qx, qy = query_point
    query_idx = qy * latent_res_q + qx
    attention_vector = cross_attn_map_item[query_idx, :]
    heatmaps = attention_vector.reshape(t_in, latent_res_k, latent_res_k)

    # 绘图
    fig, axes = plt.subplots(1, t_in + 1, figsize=(20, 5))
    fig.suptitle(f"Cross-Attention Overlay @ t={timestep}\nModule: {module_name}", fontsize=14)

    # 1. 在输出图像上标记查询点
    output_with_marker = output_image.copy()
    # 在图像上画一个点 (这里用一个简单的黑白方块代替)
    marker_size = output_image.size[0] // 32
    for x_i in range(marker_size):
        for y_i in range(marker_size):
            draw_x = (qx * output_image.size[0] // latent_res_q) + x_i
            draw_y = (qy * output_image.size[1] // latent_res_q) + y_i
            if 0 <= draw_x < output_image.size[0] and 0 <= draw_y < output_image.size[1]:
                output_with_marker.putpixel((draw_x, draw_y), (255, 0, 0)) # 红色标记
    axes[0].imshow(output_with_marker)
    axes[0].set_title(f"Query Point ({qx},{qy})\nin Output View")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # 2. 循环创建并绘制每个输入视角的叠加图
    for i in range(t_in):
        overlay_image = create_overlay(condition_images[i], heatmaps[i])
        axes[i+1].imshow(overlay_image)
        axes[i+1].set_title(f"Attention on Input View {i+1}")
        axes[i+1].set_xticks([]); axes[i+1].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_self_attention_overlay(self_attn_map, output_image,
                                     latent_res_q, query_point,
                                     timestep, module_name):
    """
    (最终版-带叠加) 可视化自注意力图。
    """
    if self_attn_map is None: return
    self_attn_map_item = self_attn_map[1] # 选择有条件的batch
    qx, qy = query_point
    query_idx = qy * latent_res_q + qx
    attention_vector = self_attn_map_item[query_idx, :]
    heatmap = attention_vector.reshape(latent_res_q, latent_res_q)
    
    # 在输出图像上叠加自注意力热力图
    overlay_image = create_overlay(output_image, heatmap)

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_image)
    plt.title(f"Self-Attention Overlay for Query Point ({qx},{qy})\nModule: {module_name} @ t={timestep}", fontsize=14)
    plt.xticks([]); plt.yticks([])
    plt.show()
   

def create_overlay(image, heatmap, alpha=0.6):
    """
    (最终防弹版) 创建热力图叠加，能处理值为常数的边缘情况。
    """
    if heatmap is None or heatmap.numel() == 0:
        print("⚠️ 警告: create_overlay 接收到的 heatmap 为空，返回原图。")
        return image

    heatmap_np = heatmap.detach().cpu().numpy().astype(np.float32)
    heatmap_resized = cv2.resize(heatmap_np, (image.width, image.height))
    
    # --- 关键修复：处理分母为零的情况 ---
    min_val = np.min(heatmap_resized)
    max_val = np.max(heatmap_resized)
    
    if max_val == min_val:
        # 如果热力图所有值都相同，则创建一个全零的标准化图，
        # 这意味着它不会对原图产生任何可见的叠加效果。
        heatmap_normalized = np.zeros_like(heatmap_resized)
    else:
        # 只有在分母不为零时，才执行标准化
        heatmap_normalized = (heatmap_resized - min_val) / (max_val - min_val)
    # --- 修复结束 ---

    heatmap_colored = plt.cm.viridis(heatmap_normalized)[:, :, :3] * 255
    
    # 确保两个图像都是 uint8 类型，并且 channel 匹配
    image_np = np.array(image).astype(np.uint8)
    heatmap_colored_np = heatmap_colored.astype(np.uint8)
    
    # 确保原图是3通道的
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored_np, alpha, 0)
    
    return Image.fromarray(overlay)


def visualize_cross_frame_self_attention(
    self_attn_map: torch.Tensor,
    all_pred_images: List[Image.Image],
    latent_res_q: int,
    query_point: tuple,
    query_frame_index: int,
    timestep: int,
    module_name: str,
    output_dir: str = "attention_visualizations"
    
):
    """
    (跨帧版) 可视化自注意力图，将一个查询点的注意力分布到所有帧上。

    Args:
        self_attn_map (torch.Tensor): 完整的自注意力图。
        all_pred_images (List[Image.Image]): 包含所有预测图的列表。
        latent_res_q (int): 单个 latent 特征图的空间分辨率 (H 或 W)。
        query_point (tuple): 查询点在帧内的 (x, y) 坐标。
        query_frame_index (int): 查询点所在的帧的索引 (例如, 0, 1, 2...)。
        timestep (int): 当前的时间步。
        module_name (str): 注意力模块的名称。
    """
    if self_attn_map is None:
        return

    num_frames = len(all_pred_images)
    tokens_per_frame = latent_res_q * latent_res_q
    
    # 1. 选择批次中的一项 (这里沿用你之前的逻辑，选择索引为1的)
    self_attn_map_item = self_attn_map[1]  # Shape: [N*H*W, N*H*W]
    
    # 2. 计算查询点在整个拼接特征图中的全局索引
    qx, qy = query_point
    query_idx_in_frame = qy * latent_res_q + qx
    global_query_idx = query_frame_index * tokens_per_frame + query_idx_in_frame

    # 3. 提取该查询点对所有像素的注意力向量
    # 这个向量的长度是 N * H * W
    attention_vector = self_attn_map_item[global_query_idx, :]
    
    # 4. 将长向量分割成 N 块，每一块对应一帧图像
    attention_maps_for_frames = torch.chunk(attention_vector, num_frames, dim=0)

    # 5. 可视化：为每一帧创建一个带叠加的子图
    fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 6))
    if num_frames == 1: # 如果只有一张图，axes不是列表，将其转为列表
        axes = [axes]

    fig.suptitle(f"Cross-Frame Self-Attention Overlay for Query Point ({qx},{qy}) in Frame {query_frame_index+1}\n"
                 f"Module: {module_name} @ t={timestep}", fontsize=16)

    for i in range(num_frames):
        # 将对应帧的注意力向量块 reshape 成 2D 热力图
        heatmap = attention_maps_for_frames[i].reshape(latent_res_q, latent_res_q)
        
        # 在对应帧的图像上创建叠加
        overlay_image = create_overlay(all_pred_images[i], heatmap)
        
        # 绘制子图
        axes[i].imshow(overlay_image)
        axes[i].set_title(f"Attention on Frame {i+1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # 高亮标记查询点所在的帧
        if i == query_frame_index:
            # --- 从这里开始是修改的部分 ---
            
            # 1. 计算每个 token 在图像空间对应的 patch (方块) 大小
            current_image = all_pred_images[i]
            patch_width = current_image.width / latent_res_q
            patch_height = current_image.height / latent_res_q
            
            # 2. 计算方块的左上角坐标
            rect_x = qx * patch_width
            rect_y = qy * patch_height
            
            # 3. 创建一个半透明的红色矩形 Patch
            # alpha 控制透明度, 0.4 表示 40% 的不透明度
            rect = patches.Rectangle(
                (rect_x, rect_y), 
                patch_width, 
                patch_height, 
                linewidth=2, 
                edgecolor='r', 
                facecolor='r', 
                alpha=0.4
            )
            
            # 4. 将方块添加到图上
            axes[i].add_patch(rect)
            
            # 之前的红色边框仍然保留，以高亮整个查询帧
            axes[i].spines['top'].set_color('red'); axes[i].spines['bottom'].set_color('red')
            axes[i].spines['left'].set_color('red'); axes[i].spines['right'].set_color('red')
            axes[i].spines['top'].set_linewidth(4); axes[i].spines['bottom'].set_linewidth(4)
            axes[i].spines['left'].set_linewidth(4); axes[i].spines['right'].set_linewidth(4)
            # --- 修改结束 ---


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    safe_module_name = module_name.replace('.', '_').replace('/', '_')
    safe_query_point = f"{query_point[0]}_{query_point[1]}" # 例如 (10, 6) -> "10_6"
    
    # 2. 构建层次化的目录结构
    # 结构: output_dir / 模块名 / 查询点坐标 / 查询帧索引 /
    target_dir = os.path.join(output_dir, safe_module_name, safe_query_point, f"frame_{query_frame_index}")
    
    # 3. 确保这个嵌套的目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 4. 将时间步作为文件名 (补零以方便排序)
    filename = f"{str(timestep).zfill(4)}.png"
    
    # 5. 拼接完整路径并保存图像
    output_path = os.path.join(target_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    
    # 6. 关闭图像，释放内存
    plt.close(fig)
   
    
def visualize_latent_grid(image, latent_resolution: int):
    """
    在图像上叠加一个latent空间的网格和坐标，以帮助手动定位区域。

    Args:
        image_path (str): 带有瑕疵的图像文件路径。
        latent_resolution (int): 图像对应的latent空间分辨率 (例如 16, 32)。
    """
    image = image[0]
    width, height = image.size
    patch_w = width // latent_resolution
    patch_h = height // latent_resolution

    # 创建一个可以在图像上绘制的对象
    draw = ImageDraw.Draw(image)

    # 尝试加载一个字体，如果失败则不写文字
    try:
        font = ImageFont.truetype("arial.ttf", size=max(10, int(patch_w / 4)))
    except IOError:
        print("⚠️ 警告：未找到 'arial.ttf' 字体，将不显示坐标文本。")
        font = None

    # 绘制网格和坐标
    for i in range(latent_resolution): # x 坐标
        for j in range(latent_resolution): # y 坐标
            # 绘制格子边框
            left = i * patch_w
            top = j * patch_h
            right = (i + 1) * patch_w
            bottom = (j + 1) * patch_h
            draw.rectangle([left, top, right, bottom], outline="red", width=1)
            
            # 在格子中央写上坐标
            if font:
                text = f"({i},{j})"
                # text_bbox = draw.textbbox((0,0), text, font=font) # for newer Pillow
                # text_width = text_bbox[2] - text_bbox[0]
                # text_height = text_bbox[3] - text_bbox[1]
                # 用一个简单的方式居中
                text_x = left + patch_w / 4
                text_y = top + patch_h / 4
                draw.text((text_x, text_y), text, fill="yellow", font=font)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Latent Space Grid Overlay")
    plt.axis('off')
    plt.show()


# =================================================================================
# 主逻辑
# =================================================================================

if __name__ == "__main__":
    
    # --- 1. 加载注意力数据 (与之前相同) ---
    print("--- 步骤 1: 正在加载注意力图数据 ---")
    # FILE_PATH = "attention_map/cache_attention_2out.pt"
    FILE_PATH = "attention_map/cache_attention.pt"

    attention_data = torch.load(FILE_PATH, map_location='cpu')
    print("✅ 注意力数据加载成功。")
    # ... (打印可用时间步和模块名的代码) ...
    
    # --- 2. 新的图像加载方式 ---
    print("\n--- 步骤 2: 正在从拼接图中加载并分割图像 ---")

    # !! 请将这里的路径替换为您的拼接图路径 !!
    # COMBINED_IMAGE_PATH = "experimental_results_024+/unnamed_experiment_20250825-204109/comparison_images/compare_0000.png"
    COMBINED_IMAGE_PATH = "experimental_results_024+/unnamed_experiment_20250825-221413/comparison_images/compare_0000.png"
    extracted_images = extract_images_from_grid(COMBINED_IMAGE_PATH, num_rows = 1)
    # extracted_images = extract_images_from_grid(COMBINED_IMAGE_PATH, num_rows = 2)


    if extracted_images:
        # 根据我们之前的假设：左边是预测，右边是GT
        output_image = [image_pair[0] for image_pair in extracted_images]
        
        # 加载作为条件的输入图像 (这部分逻辑保持不变)
        CONDITION_IMAGE_PATHS = [
            "D:/GSO_datasets/eschernet_data/3D_Dollhouse_Sofa/model/006.png",
            "D:/GSO_datasets/eschernet_data/3D_Dollhouse_Sofa/model/024.png",
            "D:/GSO_datasets/eschernet_data/3D_Dollhouse_Sofa/model/002.png",
        ]
        condition_images = [Image.open(p).convert("RGB") for p in CONDITION_IMAGE_PATHS]
         
        print("✅ 拼接图分割成功，所有图像已加载。")
    else:
        exit() # 如果图像加载失败则退出
    print("-" * 50)

    # --- 3. 设置参数 (与之前相同) ---
    print("\n--- 步骤 3: 请在下方配置您的可视化参数 ---")
    T_IN = 3
    LATENT_RESOLUTION_QUERY = 16 
    LATENT_RESOLUTION_KEY = 7
    available_steps = sorted(attention_data.keys())
    TIMESTEP_TO_ANALYZE = available_steps[len(available_steps) // 2]
    # MODULE_TO_ANALYZE = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.cross"
    MODULE_TO_ANALYZE = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.self"
    QUERY_POINT = (8, 8) 
    # QUERY_POINT = (8, 6) 
    QUERY_FRAME_INDEX = 0
    num_self_attention_frames = len(output_image)
    
    # visualize_latent_grid(output_image, 16) 
    

    # --- 4. 调用新的可视化函数 ---
    print("\n--- 步骤 4: 正在生成带叠加的可视化图像 ---")
    for i in range(len(available_steps) - 1 , 0, -1):
        TIMESTEP_TO_ANALYZE = available_steps[i]
        maps_at_selected_step = attention_data.get(TIMESTEP_TO_ANALYZE, {})
        attn_map_tensor = maps_at_selected_step.get(MODULE_TO_ANALYZE)
        # del attention_data  
        
        # for name in maps_at_selected_step.keys():
        #     print(f"- {name} (形状: {maps_at_selected_step[name].shape})")
        
        if "cross" in MODULE_TO_ANALYZE:
            visualize_cross_attention_overlay(
                attn_map_tensor, output_image, condition_images, T_IN,
                LATENT_RESOLUTION_QUERY, LATENT_RESOLUTION_KEY, QUERY_POINT,
                TIMESTEP_TO_ANALYZE, MODULE_TO_ANALYZE
            )
        elif "self" in MODULE_TO_ANALYZE:
            visualize_cross_frame_self_attention(
                self_attn_map=attn_map_tensor,
                all_pred_images=output_image,
                latent_res_q=LATENT_RESOLUTION_QUERY,
                query_point=QUERY_POINT,
                query_frame_index=QUERY_FRAME_INDEX,
                timestep=TIMESTEP_TO_ANALYZE,
                module_name=MODULE_TO_ANALYZE
            )
        
        print("\n✅ 可视化完成。")