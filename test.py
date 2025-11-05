import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
import einops
from models import util
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import sys

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Zero123 training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="C:/Users/mindirui/Desktop/EscherNet/kxic/eschernet-6dof",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--T_in", type=int, default=3, help="Number of input views"
    )
    parser.add_argument(
        "--T_out", type=int, default=1, help="Number of output views"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/GSOdataset",
        help=(
            "The input data dir. Should contain the .png files (or other data files) for the task."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs_eval",
        help=(
            "The output directory where the model predictions and checkpoints will be written."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=False, help="Whether or not to use xformers."
    )
    
    parser.add_argument("--timesteps", nargs='*', type=int, default=[])
    parser.add_argument("--downblock", nargs='*', type=int, default=[])
    parser.add_argument("--upblock", nargs='*', type=int, default=[])
    parser.add_argument("--addition_num", type=int, default=0)
    parser.add_argument("--sample_type", type=str, default='random')
    parser.add_argument("--sample_metrics", type=int, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images."
        )

    return args


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

def prepare_data(batch, device = torch.device("cuda"), weight_dtype = torch.float16):
        input_images = batch["input_images"]
        input_images = resize_and_normalize_batch_images(input_images)
        pose_in = batch["input_poses"]
        pose_out = batch["output_poses"]
        pose_in = pose_in.squeeze(0).cpu().numpy()
        pose_out = pose_out.squeeze(0).cpu().numpy()
        pose_in_inv = np.linalg.inv(pose_in).transpose([0, 2, 1])
        pose_out_inv = np.linalg.inv(pose_out).transpose([0, 2, 1])
        pose_in_inv = torch.from_numpy(pose_in_inv).to(device).to(weight_dtype).unsqueeze(0)
        pose_out_inv = torch.from_numpy(pose_out_inv).to(device).to(weight_dtype).unsqueeze(0)
        pose_in = torch.from_numpy(pose_in).to(device).to(weight_dtype).unsqueeze(0)
        pose_out = torch.from_numpy(pose_out).to(device).to(weight_dtype).unsqueeze(0)
        input_images = einops.rearrange(input_images, "b t c h w -> (b t) c h w")
        T_in = pose_in.shape[1]
        T_out = pose_out.shape[1]
        return input_images, pose_in, pose_in_inv,pose_out, pose_out_inv, T_in, T_out
    

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from dataset import GSO_sequence 
    select_imgs=[
        {
            "obj_path": "D:/GSO_datasets/eschernet_data/3D_Dollhouse_Sofa/model",
            "input": ["006.png", "024.png","002.png"],
            "output": ["003.png","003.png"]
        }
    ]
    dataset_config = GSO_sequence.DatasetConfig(root = "D:/GSO_datasets/eschernet_data", 
                                                sample_type="select_imgs",  
                                                input_num = 3 , output_num = 1,
                                                seed = 42, 
                                                select_imgs=select_imgs)
    dataset = GSO_sequence.GSO_sequence(dataset_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sys.path.insert(0, "./models/")
    # use the customized diffusers modules
    from diffusers import DDIMScheduler
    from diffusers.utils import is_torch_version
    # from dataset import get_pose
    from CN_encoder import CN_encoder
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
    
    from hook_generator import (
    create_hook_manager,
    SelectiveUpBlockScalingHook
    # 如果需要，可以导入其他 hook 类，例如 ScaleHook
    )
    MY_HOOK_GROUPS = {
    "default": [], # 一个空的组，不执行任何操作
    "weaken_cross_attention": [
        SelectiveUpBlockScalingHook(
            upblock_indices={0, 1, }, # 作用于所有的 up_block
            hidden_states_scale=1.5,      # 上采样路径的 hidden_states 不变
            res_hidden_states_scale=1.0,  # 来自 U-Net 另一侧的 res_hidden_states 强度减半
            use_new_torch_logic=is_torch_version(">=", "1.11.0")
        )
    ],
    "boost_final_upblock": [
        SelectiveUpBlockScalingHook(
            upblock_indices={3},          # 仅作用于最后一个 up_block (假设索引为3)
            hidden_states_scale=1.0,      
            res_hidden_states_scale=1.5,  # 将 res_hidden_states 的影响增强50%
            use_new_torch_logic=is_torch_version(">=", "1.11.0")
        )
    ]
}
    selected_group_name = "default" # 您可以在这里切换组名来测试不同效果
    hooks_to_apply = MY_HOOK_GROUPS[selected_group_name]

    print(f"\n🚀 [Hook Manger] 即将使用 hook 组: '{selected_group_name}'")
    

    # Init pipeline
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path , subfolder="scheduler",
                                                revision=args.revision, local_files_only = True)
    image_encoder = CN_encoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        weight_dtype = torch.float16
    )
    pipeline.image_encoder = image_encoder
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output_dir = 'test_output'
    
    with create_hook_manager(pipeline.unet, hooks_to_apply) as manager:
        manager.print_active_hooks()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
            input_images, pose_in, pose_in_inv,pose_out, pose_out_inv, T_in, T_out = prepare_data(batch, )
            with torch.autocast("cuda"):
                output = pipeline(input_imgs=input_images, prompt_imgs=input_images, poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
                                    height=256, width=256, T_in=T_in, T_out=T_out,
                                    guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator,
                                    output_type="numpy" ).images
                gt = batch["output_images"]
            now = datetime.now()
            time_str = now.strftime("%H%M")  # 小时+分钟，如 "1423"
            filename = f"{time_str}.png"
            save_path = os.path.join(output_dir, filename)
            gt = gt[:,:1,:,:]
            output = output[:1,:,:]
            util.save_concat_images(gt, output, save_path)

            break
    

        
if __name__ == "__main__":
        
    import warnings
    import logging
    import os
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["XFORMERS_DISABLE_TRITON_WARNING"] = "1"  # 如果支持
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    
    args = parse_args()
    print("=== Running with params ===")
    print(f"timesteps: {args.timesteps}")
    print(f"downblock: {args.downblock}")
    print(f"upblock: {args.upblock}")
    print(f"addition_num: {args.addition_num}")
    print(f"T_in: {args.T_in}")
    print(f"T_out: {args.T_out}")
    main(args)

