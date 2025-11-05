import cv2
import lpips
LPIPS = lpips.LPIPS(net='alex', version='0.1')
import torch.nn.functional as F
import torch
from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import einops
import torchvision.transforms.functional as TF
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import os
import traceback
from torchvision.transforms.functional import to_pil_image # 需要导入这个转换函数
# from new_processor import AttentionCache, ProviderProcessor


def evaluate_and_save_single(
    generated_np: np.ndarray,
    gt_np: np.ndarray,
    save_dir: str = "./outputs", # 提供一个默认的保存目录
    item_id: str = None
):
    """
    一个自包含的函数，用于评估、保存并打印单张生成图像的指标。
    它会在内部自行初始化LPIPS模型。

    Args:
        generated_np (np.ndarray): pipeline输出的, [H, W, 3], 范围[0, 255], uint8格式的图像。
        gt_np (np.ndarray):        对应的Ground Truth图像，格式与上面相同。
        save_dir (str):              保存生成图像的目录。
        item_id (str, optional):     用于命名的ID，如果为None则使用时间戳。
    """
    # --- 1. 统一数据格式 ---
    print("\n--- [DEBUG] 进入评估函数，正在统一数据格式 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_np = gt_np.squeeze(0)
    try:
        # a. 处理生成图 (PIL -> Numpy)
        # 确保输入是 PIL Image 对象
        if not isinstance(generated_np, Image.Image):
            generated_np = Image.fromarray((generated_np[0] * 255).astype(np.uint8))
        if not isinstance(generated_np, Image.Image):
            raise TypeError(f"generated_pil 需要是 PIL.Image, 但收到的是 {type(generated_np)}")
        generated_np = np.array(generated_np)

        # b. 处理Ground Truth (Tensor -> PIL -> Numpy)
        if not isinstance(gt_np, torch.Tensor):
            raise TypeError(f"gt_tensor 需要是 torch.Tensor, 但收到的是 {type(gt_np)}")
        
        if gt_np.shape[2] == 512:
            gt_np = F.interpolate(gt_np, size=(256, 256), mode='bilinear', align_corners=False)
            
        # 从 [1, 3, H, W] 中去掉批次维度 -> [3, H, W]
        gt_tensor_3d = gt_np.squeeze(0) 
        # 将Tensor转换为PIL Image以便保存和转换为Numpy
        # 注意: to_pil_image 期望输入在CPU上
        gt_pil = to_pil_image(gt_tensor_3d.cpu())
        gt_np = np.array(gt_pil)

        print(f"[DEBUG] 数据格式统一完成。 Gen shape: {generated_np.shape}, GT shape: {gt_np.shape}")

    except Exception as e:
        print(f"❌ 数据格式统一时出错: {e}")
        traceback.print_exc()
        return

    # --- 2. 保存图像 ---
    try:
        # 如果没有提供item_id，则使用时间戳作为文件名
        if item_id is None:
            item_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
        output_path = os.path.join(save_dir, f"{item_id}_generated.png")
        Image.fromarray(generated_np).save(output_path)
    except Exception as e:
        print(f"❌ 保存图像时出错: {e}")
        return

    # --- 3. 计算指标 ---
    # a. PSNR 和 SSIM (在Numpy上计算)
    psnr_score = psnr(gt_np, generated_np, data_range=255)
    ssim_score = ssim(gt_np, generated_np, data_range=255, channel_axis=-1, win_size=7)

    # b. LPIPS (在Tensor上计算)
    # 【关键修改】在函数内部初始化LPIPS模型
    # 注意：这会使函数每次调用时都加载一次LPIPS模型，速度稍慢，但对于单次测试完全没问题。
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    gt_tensor = lpips.im2tensor(gt_np).to(device)
    gen_tensor = lpips.im2tensor(generated_np).to(device)
    
    with torch.no_grad():
        lpips_score = lpips_model(gt_tensor, gen_tensor).item()

    # --- 4. 打印结果 ---
    print("\n--- 图像质量评估结果 ---")
    print(f"图像已保存至: {output_path}")
    print(f"PSNR (越高越好)   : {psnr_score:.4f}")
    print(f"SSIM (越高越好)   : {ssim_score:.4f}")
    print(f"LPIPS (越低越好)  : {lpips_score:.4f}")
    print("--------------------------\n")

def calc_2D_metrics(pred_np, gt_np):
    # pred_np: [H, W, 3], [0, 255], np.uint8
    pred_image = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)
    gt_image = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)
    # [0-255] -> [-1, 1]
    pred_image = pred_image.float() / 127.5 - 1
    gt_image = gt_image.float() / 127.5 - 1
    # for 1 image
    # pixel loss

    loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
    # LPIPS
    lpips = LPIPS(pred_image[0], gt_image[0].cpu()).item()  # [-1, 1] torch tensor
    # SSIM
    ssim = calculate_ssim(pred_np, gt_np, channel_axis=2)
    # PSNR
    psnr = cv2.PSNR(gt_np, pred_np)

    return loss, lpips, ssim, psnr

def preprocess_and_eval(output, gt_batch):
    """
    Args:
        output: np.ndarray, shape [V, H, W, 3] or [B, V, H, W, 3], values in [0,1]
        gt_batch: torch.Tensor, shape [B, V, 3, H_gt, W_gt], values in [0,1]

    Returns:
        平均loss, lpips, ssim, psnr
    """
    if output.ndim == 4:
        # output [V, H, W, 3] → add batch dim
        output = np.expand_dims(output, axis=0) # [1, V, H, W, 3]

    B, V, H_out, W_out, _ = output.shape
    losses, lpips_vals, ssims, psnrs = [], [], [], []

    for b in range(B):
        for v in range(V):
            pred_np = output[b, v]
            gt_i = gt_batch[b, v]  # [3, H_gt, W_gt]

            # resize gt 到 output 大小
            gt_resized = F.interpolate(
                gt_i.unsqueeze(0), size=(H_out, W_out), mode='bilinear', align_corners=False
            )[0]  # [3, H_out, W_out]

            # 转成 np.uint8
            pred_uint8 = (pred_np * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt_resized.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(pred_np)
            # plt.title("Predicted")
            # plt.axis("off")

            # plt.subplot(1, 2, 2)
            # plt.imshow(gt_np)
            # plt.title("Ground Truth")
            # plt.axis("off")

            # plt.tight_layout()
            # plt.savefig("debug_compare.png")

            loss, lpips, ssim, psnr = calc_2D_metrics(pred_uint8, gt_np)
            losses.append(loss)
            lpips_vals.append(lpips)
            ssims.append(ssim)
            psnrs.append(psnr)

    return losses, lpips_vals, ssims, psnrs

def apply_zoom_augmentation(batch, max_extra_inputs, radius_list=[1.4, 1.6, 1.8],  seed=42):
    import os
    import random

    rng = random.Random(seed)

    if max_extra_inputs == -1:
        max_extra_inputs = batch["input_images"].shape[1]  # 注意：假设 shape 为 (B, V, C, H, W)

    input_paths = batch["input_path"]

    existing = set(p[0] for p in input_paths)  # 兼容 (str,) tuple
    extra_paths = []
    extra_images = []
    extra_poses = []

    def parse_metadata(p):
        name = os.path.basename(p).replace(".png", "")
        elev_id, azim_str, r_str = name.split("_")
        elev = -20 if elev_id == "1" else 30
        azim = float(azim_str) * 30
        r = float(r_str)
        return elev, azim, r

    elev_azim_pairs = [(parse_metadata(p[0])[0], parse_metadata(p[0])[1]) for p in input_paths]

    m = rng.randint(1, min(max_extra_inputs, len(input_paths)))
    trials = 0
    max_trials = 100  # 防止死循环

    while len(extra_paths) < m and trials < max_trials:
        trials += 1
        idx = rng.randint(0, len(input_paths) - 1)
        elev, azim = elev_azim_pairs[idx]

        base_path = os.path.dirname(input_paths[idx][0])
        old_r = parse_metadata(input_paths[idx][0])[2]
        available_rs = [r for r in radius_list if r != old_r]

        if not available_rs:
            continue

        new_r = rng.choice(available_rs)
        elev_id = int(elev == -20)
        azim_id = f"{azim / 30:.1f}"
        new_name = f"{elev_id}_{azim_id}_{new_r}.png"
        new_path = os.path.join(base_path, new_name)

        if new_path in existing:
            continue
        existing.add(new_path)
        
        # === 图像读取逻辑复用 Dataset ===
        img = plt.imread(new_path)
        img[img[:, :, -1] == 0.] = [1., 1., 1., 1.]
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
        if hasattr(batch, "transform"):
            img_tensor = batch.transform(img)
        else:
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # 位姿加载
        pose_path = new_path.replace(".png", ".npy")
        pose = np.load(pose_path)
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :4] = pose
        pose_tensor = torch.tensor(pose_4x4, dtype=torch.float32)

        extra_paths.append(new_path)
        extra_images.append(img_tensor)
        extra_poses.append(pose_tensor)
        existing.add(new_path)

    # 添加到 batch
    if extra_paths:
        batch["input_path"].extend(extra_paths)
        batch["input_images"] = torch.cat([
            batch["input_images"],
            torch.stack(extra_images).unsqueeze(0)  # (1, V_new, C, H, W)
        ], dim=1)

        batch["input_poses"] = torch.cat([
            batch["input_poses"],
            torch.stack(extra_poses).unsqueeze(0)  # (1, V_new, 4, 4)
        ], dim=1)
    return batch



def get_bbox_area_ratio_batch(images: torch.Tensor) -> list[float]:
    """
    输入:
        images: torch.Tensor, shape = (n_view, 3, h, w), RGB格式，值域 0~1 或 0~255
    返回:
        list[float]，每张图的 中心对称bbox对角线 / 图像对角线
        若触及图像边界则返回1.0，若无前景返回0.0
    """
    images = images[0]
    ratios = []
    n_view, c, h, w = images.shape
    diag_full = np.sqrt(h ** 2 + w ** 2)
    cx, cy = w // 2, h // 2  # 图像中心

    # 转为 uint8 numpy
    if images.dtype != torch.uint8:
        imgs_np = (images * 255).byte().cpu().numpy()
    else:
        imgs_np = images.cpu().numpy()

    for i in range(n_view):
        img = imgs_np[i]  # (3, h, w)
        img = np.transpose(img, (1, 2, 0))  # -> (h, w, 3)

        # 前景掩码（非白）
        tol = 5
        mask = ~np.all(img >= 255 - tol, axis=2)

        if not np.any(mask):
            ratios.append(0.0)
            continue

        ys, xs = np.where(mask)

        # 中心对称 bbox（最大偏移量）
        dx = np.max(np.abs(xs - cx))
        dy = np.max(np.abs(ys - cy))

        # bbox边界（左、右、上、下）
        x_min = cx - dx
        x_max = cx + dx
        y_min = cy - dy
        y_max = cy + dy

        # 如果 bbox 触及图像边缘，视为不完整
        if x_min < 0 or x_max >= w or y_min < 0 or y_max >= h:
            ratios.append(1.0)
            continue

        diag_bbox = np.sqrt((2 * dx + 1) ** 2 + (2 * dy + 1) ** 2)
        ratios.append(diag_bbox / diag_full)
        
    return ratios

def save_concat_images(gt, image_mask, save_path, target_size=(256, 256)):
    """
    将 gt 和 image_mask 两组图像按行拼接，拼成两列图：
    左列为 image_mask，右列为 gt。
    两组图像均resize到 target_size。
    
    参数:
        gt: torch.Tensor，形状 (N, C, H, W)，通常是gt图像
        image_mask: numpy数组或torch.Tensor，形状 (N, H, W, C) 或 (N, C, H, W)，数值范围[0,1]
        save_path: 保存路径，包含文件名
        target_size: tuple，resize大小，默认 (256,256)
    """ 
    if gt.ndim == 5:
        gt = gt[0]
    n = gt.shape[0]
    width, height = target_size[0] * 2, target_size[1] * n
    combined_image = Image.new("RGB", (width, height))
    
    if gt.shape[-2:] != target_size:
        gt = F.interpolate(
        gt,                  # 输入张量
        size=target_size,    # 目标尺寸 (H, W)
        mode='bilinear',     # 插值模式，对于图像通常使用'bilinear'或'bicubic'
        align_corners=False  # 建议在大多数情况下设为False，以获得更一致的行为
    )
    
    for i in range(n):
        # 处理 image_mask[i]
        mask_img = image_mask[i]
        # 如果是Tensor，先转换成 numpy，且确认通道维度
        if isinstance(mask_img, torch.Tensor):
            # 假设 mask_img shape (C,H,W)
            mask_img = mask_img.permute(1, 2, 0).cpu().numpy()
        # mask_img shape 应该是 (H,W,C)，数值范围[0,1]
        mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(target_size, Image.BILINEAR)
        
        # 处理 gt[i]
        gt_img = gt[i] # (1,C,H,W)
        gt_pil = TF.to_pil_image(gt_img)
        
        y_offset = i * target_size[1]
        combined_image.paste(mask_pil, (0, y_offset))        # 左列 image_mask
        combined_image.paste(gt_pil, (target_size[0], y_offset))  # 右列 gt
    
    combined_image.save(save_path)
    
class FinalCallbackHandler:
    def __init__(self, pipe, output_dir, run_name):
        """
        初始化时，“接收房子的钥匙”。
        我们将整个 pipeline 实例 (pipe) 保存下来。
        """
        self.pipe = pipe
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, i: int, t: int, latents: torch.FloatTensor):
        """
        每次被调用时，使用我们保存好的钥匙（self.pipe）来开门（调用方法）。
        """
        print(f"--- Callback at step {i} (Timestep: {t}) ---")

        # a) 保存原始潜变量（这部分不变）
        latent_path = os.path.join(self.output_dir, f"{self.run_name}_step_{i:03d}_latent.pt")
        torch.save(latents.clone(), latent_path)
        print(f"  Saved latent to: {latent_path}")

        # b) 直接调用 pipe 上的方法来解码 latent
        #    这完全实现了您的想法！
        #    注意: 大多数 pipeline 没有直接的 decode_latents, 但我们可以模拟它
        #    它内部的逻辑就是我们之前讨论的 缩放 -> vae.decode
        with torch.no_grad():
            # Step 1: Call the same decode_latents method
            # This handles VAE scaling and decoding.
            decoded_image_np = self.pipe.decode_latents(latents.clone())
            
            # Step 2: Call the same numpy_to_pil method
            image_pil = self.pipe.numpy_to_pil(decoded_image_np)
            
        # --- c) Save the resulting PIL image ---
        if len(image_pil) == 1:
            image_path = os.path.join(self.output_dir, f"{self.run_name}_step_{i:03d}_pixel.png")
            image_pil[0].save(image_path)
            print(f"  Saved decoded image to: {image_path}")
        else:
            for j, img in enumerate(image_pil):
                image_path = os.path.join(self.output_dir, f"{self.run_name}_step_{i:03d}_pixel_{j}.png")
                img.save(image_path)
                print(f"  Saved {len(image_pil)} decoded image to: {image_path}")
                
def get_averaged_noise(
    base_tensor: torch.Tensor,
    generator: torch.Generator,
    num_avg: int = 5
) -> torch.Tensor:
    """
    使用同一个generator生成并平均多个噪声。

    Args:
        base_tensor (torch.Tensor): 用于获取形状、设备和数据类型的模板张量 (例如 z0)。
        generator (torch.Generator): 用于生成随机数的、有状态的生成器。
        num_avg (int, optional): 要平均的噪声样本数量。默认为 5。

    Returns:
        torch.Tensor: 平均后的噪声张量。
    """
    # 初始化一个空列表来存储生成的每个噪声张量
    noises = []

    # print(f"INFO: Generating and averaging {num_avg} noise samples...")
    # 循环N次，生成N个不同的噪声
    for _ in range(num_avg):
        # 每次调用torch.randn，generator的内部状态都会更新，从而产生新的随机数
        noise = torch.randn(
            base_tensor.shape,
            generator=generator,
            device=base_tensor.device,
            dtype=base_tensor.dtype
        )
        noises.append(noise)
    
    # 使用torch.stack将噪声列表变成一个新的张量，形状为 [num_avg, C, H, W]
    stacked_noises = torch.stack(noises)
    
    # 沿着新的维度(dim=0)取平均，得到最终的、更平滑的噪声
    avg_noise = torch.mean(stacked_noises, dim=0)
    
    # print(f"INFO: Averaged noise generated. Shape: {avg_noise.shape}")
    
    return avg_noise

# def display_cache_info(cache: AttentionCache):
#     """
#     以一种清晰、可读的方式打印AttentionCache的内容摘要。
#     """
#     if not isinstance(cache, AttentionCache) or not cache.cache:
#         print("❌ 缓存为空或不是一个有效的AttentionCache对象。")
#         return

#     print("\n--- 缓存内容摘要 ---")
    
#     # 1. 总体统计
#     timesteps_cached = sorted(cache.cache.keys())
#     total_steps = len(timesteps_cached)
#     print(f"✅ 共缓存了 {total_steps} 个时间步。")
#     print(f"   时间步范围: 从 {timesteps_cached[0]} 到 {timesteps_cached[-1]}")

#     # 2. 随机抽样一个时间步进行详细检查
#     sample_timestep = timesteps_cached[total_steps // 2] # 选择中间的时间步
#     modules_in_step = cache.cache[sample_timestep]
#     print(f"\n--- 以时间步 t={sample_timestep} 为例进行详细展示 ---")
#     print(f"   在该时间步，共缓存了 {len(modules_in_step)} 个模块的K/V对。")
    
#     # 3. 随机抽样一个模块进行最终确认
#     sample_module_name = list(modules_in_step.keys())[0]
#     cached_kv = modules_in_step[sample_module_name]
#     k_tensor = cached_kv['k']
#     v_tensor = cached_kv['v']
    
#     print(f"\n   以模块 '{sample_module_name}' 为例:")
#     print(f"     - 缓存的 Key (K) 张量:")
#     print(f"         - 形状 (Shape): {k_tensor.shape}")
#     print(f"         - 数据类型 (dtype): {k_tensor.dtype}")
#     print(f"         - 所在设备 (device): {k_tensor.device}") # 应该显示为 'cpu'
#     print(f"     - 缓存的 Value (V) 张量:")
#     print(f"         - 形状 (Shape): {v_tensor.shape}")
#     print(f"         - 数据类型 (dtype): {v_tensor.dtype}")
#     print(f"         - 所在设备 (device): {v_tensor.device}") # 应该显示为 'cpu'
#     print("-----------------------\n")
