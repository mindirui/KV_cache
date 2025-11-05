import torch
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from hook_generator import create_hook_manager
import einops
from models import util as util 
import torch.nn.functional as F
import os
from models.new_processor import (
    AttentionCache, 
    pipeline_in_caching_mode, 
    pipeline_in_injection_mode
)
from models.new_processor import backup_original_processors, restore_original_processors
from util import get_averaged_noise



# ======================================================================
# 1. 辅助函数 (Optimized Helper Functions)
# ======================================================================

def setup_output_directory(config: dict) -> Path:
    """根据配置创建一个独特的、带时间戳的输出目录，并返回其路径。"""
    experiment_name = config.get('name', 'unnamed_experiment').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.get('root_output_dir', 'outputs')) / f"{experiment_name}_{timestamp}"
    
    # 同时创建用于保存拼接对比图的目录
    (run_dir / "comparison_images").mkdir(parents=True, exist_ok=True)
    
    print(f"✔️ 结果将保存在: {run_dir}")
    return run_dir

def log_parameters(config: dict, run_dir: Path):
    """
    【优化点1: 自动记录参数】
    自动将配置字典中的关键信息序列化为 JSON，无需手动指定。
    """
    log_path = run_dir / "run_logs.json"
    
    # 准备要记录的配置信息，可以排除掉庞大的对象
    config_to_log = {}
    for key, value in config.items():
        if key not in ['pipeline', 'dataloader']: # 排除掉无法序列化的对象
            config_to_log[key] = value

    # 特别处理 hooks，只记录它们的描述信息
    config_to_log['hooks_applied'] = [h.description() for h in config.get('hooks_to_apply', [])]

    log_data = {
        "run_configuration": config_to_log,
        "results": [], # 初始化结果列表
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        # 使用 json.dump 来格式化输出，方便阅读
        json.dump(log_data, f, indent=4, default=str) # default=str 防止部分类型无法序列化
    print(f"✔️ 配置文件已自动记录到: {log_path}")

def evaluate_single_view(generated_view_np: np.ndarray, gt_view_tensor: torch.Tensor) -> dict:
    """
    【已填充】处理单个视角的预处理和评估。
    这个函数现在包含了您 `preprocess_and_eval` 内部循环的核心逻辑。
    """
    # 1. 预处理：将 PyTorch Tensor 格式的真值图转换为 NumPy uint8 格式
    #    输入 gt_view_tensor shape: (3, H, W)
    gt_np_uint8 = (gt_view_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # 2. 预处理：将 NumPy float 格式的生成图转换为 NumPy uint8 格式
    #    输入 generated_view_np shape: (H, W, 3)
    pred_np_uint8 = (generated_view_np * 255).clip(0, 255).astype(np.uint8)

    # 3. 调用底层的指标计算函数
    loss, lpips, ssim, psnr = util.calc_2D_metrics(pred_np_uint8, gt_np_uint8)

    # 4. 将结果打包成我们框架需要的字典格式
    return {
        "loss": loss,
        "lpips": lpips,
        "ssim": ssim,
        "psnr": psnr,
    }

def parse_path_info(full_path: str) -> tuple[str, str]:
    """
    【V2 版】从一个完整的文件路径中解析出【物体名】和【文件标识符】。

    Args:
        full_path (str): 完整的文件路径字符串。

    Returns:
        tuple[str, str]: 一个包含 (物体名, 文件标识符) 的元组。
                         例如：('Circo_Fish_Toothbrush_Holder_14995988', '1_5.0_1.8')
    """
    if not full_path or full_path == "N/A":
        return "N/A", "N/A"
    
    try:
        p = Path(full_path)
        # p.stem 会获取文件名但去除最后的扩展名（.png）
        identifier = p.stem
        object_name = p.parent.parent.name
        return object_name, identifier
    except Exception as e:
        print(f"警告: 解析路径 '{full_path}' 失败: {e}")
        return "parsing_error", Path(full_path).stem


def evaluate_and_log_batch(
    generated_images_np: np.ndarray, 
    ground_truth_tensor: torch.Tensor,
    batch_info: dict,
    batch_idx: int,
    run_dir: Path
) -> list[dict]:
    """
    【V6 版 - 最终版】生成包含【批次概览】和【视角详情】的层次化日志。
    """
    # 1. 预处理 (不变)
    num_views = generated_images_np.shape[0]
    gt_squeezed = ground_truth_tensor.squeeze(0)
    gt_resized = F.interpolate(
        gt_squeezed, 
        size=(generated_images_np.shape[1], generated_images_np.shape[2]), 
        mode='bilinear', 
        align_corners=False
    )

    # 2. 解析路径信息，用于顶层概览 (不变)
    input_paths = batch_info.get("input_path", [])
    output_paths = batch_info.get("output_path", [])
    object_name = parse_path_info(input_paths[0][0])[0] if input_paths else "N/A"
    input_identifiers = [parse_path_info(p[0])[1] for p in input_paths]
    output_identifiers = [parse_path_info(p[0])[1] for p in output_paths]
    
    # 3. 【保留】循环处理每个视角，构建包含完整细节的列表
    per_view_details_list = []
    for view_idx in range(num_views):
        view_metrics = evaluate_single_view(
            generated_view_np=generated_images_np[view_idx], 
            gt_view_tensor=gt_resized[view_idx]
        )
        
        raw_input_path = input_paths[view_idx][0] if view_idx < len(input_paths) else "N/A"
        raw_output_path = output_paths[view_idx][0] if view_idx < len(output_paths) else "N/A"
        
        # 构建每个视角的详细信息字典
        view_detail_entry = {
            "view_index": view_idx,
            "input_path_full": raw_input_path,
            "ground_truth_path_full": raw_output_path,
            "metrics": view_metrics
        }
        per_view_details_list.append(view_detail_entry)

    # 4. 计算批次平均指标 (不变)
    metrics_for_avg = [result['metrics'] for result in per_view_details_list]
    batch_average_metrics = calculate_average_metrics(metrics_for_avg)
    
    # 5. 保存对比图 (不变)
    save_path = run_dir / "comparison_images" / f"compare_{batch_idx:04d}.png"
    util.save_concat_images(gt_resized,generated_images_np , save_path)
    
    # 6. 【关键】构建最终的、层次化的日志条目
    log_entry = {
        # --- 顶层概览信息 ---
        "batch_index": batch_idx,
        "path": object_name,
        "input_index": ",".join(input_identifiers),
        "output_index": ",".join(output_identifiers),
        "num_views": num_views,
        "saved_comparison_image": str(save_path),
        "batch_average_metrics": batch_average_metrics,
        
        # --- 保留的、详细的每个视角的信息 ---
        "per_view_details": per_view_details_list 
    }
    
    update_log_file(run_dir, {"result_item": log_entry})
    
    # 7. 返回用于计算全局平均值的指标列表 (不变)
    return metrics_for_avg

def update_log_file(run_dir: Path, entry: dict):
    """【优化点3: 安全的日志更新】读取、更新、写回JSON，而非重写整个文件。"""
    log_path = run_dir / "run_logs.json"
    # 使用文件锁可以进一步增加并发安全性，但在这里暂时简化
    with open(log_path, 'r+', encoding='utf-8') as f:
        log_data = json.load(f)
        
        if "result_item" in entry:
            log_data["results"].append(entry["result_item"])
        elif "summary" in entry:
            log_data["summary"] = entry["summary"]
            
        f.seek(0)
        json.dump(log_data, f, indent=4, default=str)
        f.truncate()

# ... (calculate_average_metrics 函数保持不变) ...
def calculate_average_metrics(all_metrics: list[dict]) -> dict:
    if not all_metrics: return {}
    summary = {}
    metric_keys = all_metrics[0].keys()
    for key in metric_keys:
        # 使用 np.nanmean 可以安全地处理可能存在的 NaN 值
        summary[key] = float(np.nanmean([m[key] for m in all_metrics]))
    return summary


def prepare_batch_for_pipeline(batch: dict, device: torch.device, weight_dtype: torch.dtype) -> dict:
    """
    接收原始批次数据，准备好 pipeline 调用所需的所有参数。
    """
    # 假设 'resize_and_normalize_batch_images' 是一个您已经定义的函数
    # from utils import resize_and_normalize_batch_images
    
    # --- 开始移植您的代码 ---
    input_images = batch["input_images"]
    input_images = resize_and_normalize_batch_images(input_images)
    
    pose_in = batch["input_poses"]
    pose_out = batch["output_poses"]
    
    # 这里的 squeeze(0) 假设 batch_size=1，如果不是，逻辑可能需要调整
    pose_in_np = pose_in.squeeze(0).cpu().numpy()
    pose_out_np = pose_out.squeeze(0).cpu().numpy()
    
    pose_in_inv_np = np.linalg.inv(pose_in_np).transpose([0, 2, 1])
    pose_out_inv_np = np.linalg.inv(pose_out_np).transpose([0, 2, 1])
    
    pose_in_inv = torch.from_numpy(pose_in_inv_np).to(device=device, dtype=weight_dtype).unsqueeze(0)
    pose_out_inv = torch.from_numpy(pose_out_inv_np).to(device=device, dtype=weight_dtype).unsqueeze(0)
    pose_in = torch.from_numpy(pose_in_np).to(device=device, dtype=weight_dtype).unsqueeze(0)
    pose_out = torch.from_numpy(pose_out_np).to(device=device, dtype=weight_dtype).unsqueeze(0)
    
    input_images = einops.rearrange(input_images, "b t c h w -> (b t) c h w")
    
    T_in = pose_in.shape[1]
    T_out = pose_out.shape[1]
    
    # 从原始 batch 中获取其他所需数据
    gt_images = batch["output_images"]
    # h, w = batch.get("height"), batch.get("width") # 假设 h, w 也在这里
    # mask = batch.get("prompt_img_mask")
    # mask_config = batch.get("mask_config")
    
    # --- 将所有 pipeline 参数打包成一个字典 ---
    pipeline_args = {
        "input_imgs": input_images,
        "prompt_imgs": input_images, # 您的代码中 prompt_imgs 和 input_imgs 相同
        "poses": [[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
        "height": 256, # 使用固定值或从 batch 中获取
        "width": 256,
        "T_in": T_in,
        "T_out": T_out,
        # "prompt_img_mask": mask,
        # "mask_config": mask_config,
    }

    # --- 返回一个包含两部分的字典：pipeline参数和真值数据 ---
    return {
        "pipeline_args": pipeline_args,
        "ground_truth": gt_images
    }

def resize_and_normalize_batch_images(input_images, resolution=256):
    """
    input_images: torch.Tensor of shape (B, V, 3, H, W), values in range [0, 1] or [0, 255]
    Output: resized and normalized images of shape (B, V, 3, resolution, resolution)
    """
    B, V = input_images.shape[:2]

    # 如果输入是 [0, 255]，可选进行归一化到 [0, 1]（视数据而定）
    if input_images.max() > 1:
        input_images = input_images / 255.0

    # Resize
    resized_images = F.interpolate(
        input_images.view(-1, 3, input_images.shape[-2], input_images.shape[-1]),
        size=(resolution, resolution),
        mode='bilinear',
        align_corners=False
    )

    # Normalize: 归一化到 [-1, 1]（模拟 torchvision.transforms.Normalize([0.5], [0.5])）
    resized_images = resized_images * 2 - 1

    # 恢复形状
    output_images = resized_images.view(B, V, 3, resolution, resolution)
    return output_images

def prepare_data_for_cache(full_pipeline_args, reference_view_index):
        caching_input_imgs = full_pipeline_args["input_imgs"] # 条件是全部3张输入图
        caching_input_poses_list = full_pipeline_args["poses"][1]
        # reconstruction_target_img = full_pipeline_args["input_imgs"][reference_view_index : reference_view_index + 1, ...]
        reconstruction_target_poses_list = [p[:, reference_view_index : reference_view_index + 1, ...] for p in caching_input_poses_list]  
        caching_pipeline_args = {
            "input_imgs": caching_input_imgs,
            "prompt_imgs": caching_input_imgs, 
            "poses": [reconstruction_target_poses_list, caching_input_poses_list],
            "height": full_pipeline_args["height"],
            "width": full_pipeline_args["width"],
            "T_in": full_pipeline_args["T_in"],
            "T_out": 1,
        }
        return caching_pipeline_args
    
def prepare_noise_reschedule(reconstruction_target_img, 
                             pipeline, 
                             generator, 
                             reference_fidelity, 
                             noise_avg_samples, 
                             total_steps,
                             test = False):
    if reference_fidelity != 0:
        z0 = pipeline.vae.encode(reconstruction_target_img.to(pipeline.device)).latent_dist.sample() * pipeline.vae.config.scaling_factor
        timesteps = pipeline.scheduler.timesteps
        start_index = int(reference_fidelity * total_steps)
        start_timestep = timesteps[start_index]
        noise = get_averaged_noise(z0, generator, noise_avg_samples)
        noisy_latents = pipeline.scheduler.add_noise(z0, noise, start_timestep)
        remaining_timesteps_array = timesteps[start_index:]
        return noisy_latents, remaining_timesteps_array
    else:
        return None, None

# ======================================================================
# 2. 核心推理函数 (Main Inference Function)
# ======================================================================


def run_inference_task(config: dict):
    """一个纯净的推理函数，接收一个配置字典并执行从头到尾的完整流程。"""
    print(f"\n🚀 === 开始任务: {config.get('name', 'Untitled')} ===")
    
    ## --- 阶段一: 设置 ---
    pipeline = config['pipeline']
    dataloader = config['dataloader']
    main_generator = config['generator']
    static_inference_args = config.get('inference_args', {})
    device = pipeline.device
    weight_dtype = torch.float16
    ## parameters for attention cache
    use_cache = config.get("use_cache", True)  ## disable cache to test baseline
    rgs_function = config.get("attention_store", None)  ## func to manipulate the rgs-strength
    reference_view_index = config.get("reference_view_index", 1)  ## choose one reference-image to help guiding generation
    ## parameters for noise_reschedule
    reference_fidelity = config.get("reference_fidelity", 0) ## Reference Image Fidelity. NOTE: A LOWER value means MORE noise and GREATER change from the reference. Range [0, 1]. 0 means starting from pure noise.
    noise_avg_samples  = config.get("noise_avg_samples", 1) ## Number of noise samples to average for a more stable start. Set to 1 to disable averaging.
    total_steps = config.get('inference_args', {}).get('num_inference_steps', 50) ## Total number of inference/denoising steps, defining the scheduler's granularity.
    ## paremeters for debug and visualization
    save_attention_map = config.get("save_attention_map", False) ## [Debug] Whether to save attention maps for analysis.
    save_interlatents = config.get("save_interlatents", False) ## [Debug] Whether to save intermediate latents during the denoising process.
    callback_step = config.get("callback_step", 1) ## [Debug] Frequency of saving intermediate latents.
    test_inference = config.get("test_inference", False) ## [Debug] If True, save no logs and only print mertrics
    ## parameters for output
    sample_metrics = config.get('sample_metrics', None) ## [Debug] If test NinNout baseline outputs, only focus on the first output image to compare with our Nin1out

    
    ## initialize a call_back_handler to restore interlatents
    if save_interlatents:
        from models.util import FinalCallbackHandler
        callback_handler = FinalCallbackHandler(pipe=pipeline, output_dir="cache_interlatent", run_name="")
    else:
        callback_handler = None
    
    ## initialize a cache 
    if use_cache:
        cache = AttentionCache()
    else:
        cache = None
        
    ## initialize output 
    run_dir = setup_output_directory(config)
    if not test_inference:
        log_parameters(config, run_dir)
        all_metrics_across_batches = [] 
        
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"推理中...")):
        # a. 为当前批次准备完整的输入数据
        prepared_data = prepare_batch_for_pipeline(batch, device, weight_dtype)
        full_pipeline_args = prepared_data['pipeline_args']
        item_id = f"batch_{batch_idx:04d}"
        
        # ==========================================================
        # 1. 准备并执行“缓存”阶段 (3-in-1-out 重建)
        # ==========================================================
        # a. 从完整数据中“切片”出用于重建的参数
        if use_cache:
            cache.clear()
            cache_pipeline_args = prepare_data_for_cache(full_pipeline_args, reference_view_index)
            cache_generator = torch.Generator(device=device).manual_seed(999)
            # store.reset()
            with torch.autocast("cuda", dtype=weight_dtype):
                with torch.no_grad():
                    generated_images_np = pipeline(
                        generator=cache_generator,
                        **cache_pipeline_args,
                        **static_inference_args,
                        cross_attention_kwargs={
                            "attention_cache": cache,
                            "hijack_mode": "caching",
                            "save_attention_map":save_attention_map,
                        },
                        callback = callback_handler,
                        callback_steps = callback_step,
                    ).images[0]


        
        ## use reference_view_index to select the reference image
        if save_attention_map:
            noisy_latents = None
            remaining_timesteps_array = torch.ones(50)
        else:
            reconstruction_target_img = full_pipeline_args["input_imgs"][reference_view_index : reference_view_index + 1, ...]
            noisy_latents, remaining_timesteps_array = prepare_noise_reschedule(reconstruction_target_img, 
                                                                                pipeline, main_generator, 
                                                                                reference_fidelity, 
                                                                                noise_avg_samples,
                                                                                total_steps,
                                                                                )       
        
        with torch.autocast("cuda", dtype=weight_dtype):
            with torch.no_grad():
                generated_images_np = pipeline(
                    latents = noisy_latents,
                    num_inference_steps = len(remaining_timesteps_array) if remaining_timesteps_array !=None else 50,
                    **full_pipeline_args,
                    cross_attention_kwargs={
                        "attention_cache": cache,
                        "hijack_mode": "injecting",
                        'rgs_function': rgs_function,
                        "save_attention_map":save_attention_map,
                        "need_save":bool(save_attention_map)
                        },
                    output_type="numpy",
                    ).images
        if save_attention_map:
            captured_maps = save_attention_map.maps
            output_filename = f"cache_attention.pt"
            torch.save(captured_maps, output_filename)
            save_attention_map.reset()
            
            
        if test_inference:
            util.evaluate_and_save_single(generated_images_np, prepared_data['ground_truth'])
        else: 
            item_metrics = evaluate_and_log_batch(
                generated_images_np, 
                prepared_data['ground_truth'], 
                batch, 
                batch_idx,
                run_dir
            )
            all_metrics_across_batches.extend(item_metrics)

    if not test_inference:
        if all_metrics_across_batches:
            sample_metrics_limit = sample_metrics
            if sample_metrics_limit:
                print(f"注意: 最终指标仅基于前 {sample_metrics_limit} 个样本计算。")
                metrics_to_summarize = all_metrics_across_batches[:,:sample_metrics_limit]
            else:
                metrics_to_summarize = all_metrics_across_batches

        summary_metrics = calculate_average_metrics(metrics_to_summarize)
        print("\n--- 任务总结 ---")
        summary_str = ", ".join([f"Avg {k.upper()}: {v:.4f}" for k,v in summary_metrics.items()])
        print(summary_str)
        update_log_file(run_dir, {"summary": {"final_metrics": summary_metrics, "notes": f"Metrics calculated on {len(metrics_to_summarize)} samples."}})
        print(f"✅ === 任务完成: {config.get('name', 'Untitled')} ===")
        print(f"详细日志请见: {run_dir / 'run_logs.json'}") 
            