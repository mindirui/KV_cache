# runner.py

import torch
from pathlib import Path
import sys
import warnings


# 忽略所有 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# ======================================================================
# 1. 导入 (Imports)
# ======================================================================

# --- 导入我们自己的模块 ---
from inference_core import run_inference_task
# from inference_core import run_inference_task

# from hook_generator import CustomStageScalingHook
# from hook_generator import SelectiveUpBlockScalingHook # 如果需要也可以导入其他 hook


sys.path.insert(0, "./models/")
# use the customized diffusers modulesc
from diffusers import DDIMScheduler
from diffusers.utils import is_torch_version
# from dataset import get_pose
from CN_encoder import CN_encoder
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from new_processor import AttentionCache, ProviderProcessor
from diffusers.models.attention import Attention

def setup_caching_processors(pipeline: torch.nn.Module) -> (AttentionCache, dict):
    """
    为给定的pipeline安装ProviderProcessors以准备进行缓存。

    Args:
        pipeline: 您已经初始化的diffusers pipeline对象。

    Returns:
        cache (AttentionCache): 一个新的、空的缓存实例。
        original_processors (dict): 一个备份了原始处理器的字典，用于后续恢复。
    """
    print("--- 正在设置缓存模式 ---")

    # 1. 初始化缓存对象
    cache = AttentionCache()

    # 2. 准备备份原始处理器
    original_processors = {}

    # 3. 遍历UNet，安装ProviderProcessor
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            # 备份原始处理器
            original_processors[name] = module.processor
            # 安装我们新的“提供方”处理器
            provider = ProviderProcessor(name, cache, original_processors[name])
            module.set_processor(provider)
            
    print(f"✅ ProviderProcessor已安装到 {len(original_processors)} 个Attention模块。")
    print("Pipeline已准备好进行缓存预计算。")
    
    return cache, original_processors

def create_rgs_schedule_func(schedule_config: dict):
    """
    根据配置创建一个返回alpha值的函数。
    """
    schedule_name = schedule_config.get('name', 'constant')
    
    if schedule_name == 'constant':
        constant_value = schedule_config.get('value', 1.0)
        return lambda t, total_steps: constant_value
    
    # 你可以在这里为 'linear', 'cosine' 等添加更多逻辑
    # ...
    
    else:
        raise ValueError(f"Unknown RGS schedule name: {schedule_name}")

# ======================================================================
#    初始化核心组件 (Initialize Core Components)
#    这些组件在所有实验中是共享的，所以只初始化一次
# ======================================================================
## initialize pipeline
print("正在初始化核心组件 (模型, 生成器等)...")
scheduler = DDIMScheduler.from_pretrained("C:/Users/mindirui/Desktop/EscherNet/kxic/eschernet-6dof" , subfolder="scheduler",
                                            revision=None, local_files_only = True)
image_encoder = CN_encoder.from_pretrained("C:/Users/mindirui/Desktop/EscherNet/kxic/eschernet-6dof", subfolder="image_encoder", revision=None)
pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
    "C:/Users/mindirui/Desktop/EscherNet/kxic/eschernet-6dof",
    revision=None,
    scheduler=scheduler,
    image_encoder=None,
    safety_checker=None,
    feature_extractor=None,
    torch_dtype=torch.torch.float32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.image_encoder = image_encoder
pipeline = pipeline.to(device)
pipeline.set_progress_bar_config(disable=True)
generator = torch.Generator(device="cuda").manual_seed(43)

from new_processor import AttentionStoreProcessor, AttentionStore
visualization_store = AttentionStore()
print("\n--- 正在为Pipeline安装 AttentionStoreProcessor ---")
for name, module in pipeline.unet.named_modules():
    if isinstance(module, Attention):
        # 每个处理器实例都知道自己的名字，并与同一个store关联
        processor = AttentionStoreProcessor(name=name, store=visualization_store)
        module.set_processor(processor)
print("✅ 处理器已安装。")

## Initialize dataset
from dataset import GSO_sequence
select_imgs=[
    {
        "obj_path": "D:/GSO_datasets/eschernet_data/Squirt_Strain_Fruit_Basket/model",
        "input": ["016.png", "012.png","009.png"],
        "output": ["020.png",]
    }
]
# select_imgs=[
#     {
#         "obj_path": "D:/GSO_datasets/eschernet_data/3D_Dollhouse_Sofa/model",
#         "input": ["006.png", "024.png","002.png"],
#         "output": ["003.png",]
#     }
# ]
dataset_config = GSO_sequence.DatasetConfig(root = "D:/GSO_datasets/eschernet_data", 
                                            sample_type="select_imgs",  
                                            input_num = 3 , output_num = 1,
                                            seed = 42, 
                                            select_imgs = select_imgs
)
dataset = GSO_sequence.GSO_sequence(dataset_config)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

## Initilaize output_config
root_output_dir = Path("./experimental_results_024+")


# ======================================================================
# 4. 定义实验配置列表 (Define Experiment Configuration List)
#    这是您的“实验控制面板”，所有的变化都在这里定义
# ======================================================================


# 1. Reference Fidelity (RF)
# 你的定义：值越低，噪声越多，变化越大
# rf_values = [0.2, 0.15 ,0.1, 0.05]
rf_values = [0.1,]

# 2. Reconstruction Guidance Scale (RGS)
# 我们将测试一系列的常数值
# rgs_constant_values = [1.0, 1.2, 1.4, 1.6]
rgs_constant_values = [1.2,]

# 3. Noise Averaging Samples (NAS)
# 我们先固定一个值，之后再回来调试它
nas_values = [1, 2, 3, 4, 5, 6, 7, 8] # 先测试1（基线）和5（你的当前值）

base_config = {
    "pipeline": pipeline,
    "dataloader": dataloader,
    "generator": torch.Generator(device="cuda").manual_seed(42),
    

    "inference_args": {
        "num_inference_steps": 50,
        "guidance_scale": 3,
    },
    "root_output_dir": root_output_dir,
    "reference_view_index": 1,
    "sample_metrics": None,
}
import copy
EXPERIMENT_CONFIGS = []
experiment_counter = 0

# for rf in rf_values:
#     for rgs_val in rgs_constant_values:
#         for nas in nas_values:
#             # 【关键】为每个实验创建一个独立的配置副本
#             config = base_config.copy()
#             # 创建一个唯一的实验名称，用于保存结果
#             experiment_name = f"exp_{experiment_counter}_rf_{rf}_rgs_{rgs_val}_nas_{nas}"
#             config["name"] = experiment_name
            
#             # --- 设置本次实验的超参数 ---
            
#             # 1. 设置 reference_fidelity
#             config["reference_fidelity"] = rf
            
#             # 2. 设置 noise_avg_samples
#             config["noise_avg_samples"] = nas
            
#             # 3. 创建并设置 rgs_function
#             rgs_config = {'name': 'constant', 'value': rgs_val}
#             alpha_func = create_rgs_schedule_func(rgs_config)
#             config["rgs_function"] = alpha_func

#             # 【重要】关于随机种子的说明 (见下文)
#             # 为了公平比较超参数，所有实验应使用相同的种子
#             config["generator"] = torch.Generator(device="cuda").manual_seed(42)
            
#             # 将配置好的实验方案加入列表
#             EXPERIMENT_CONFIGS.append(config)
#             experiment_counter += 1

# print(f"--- Generated {len(EXPERIMENT_CONFIGS)} total experiments. ---")

config = {
    "pipeline": pipeline,
    "dataloader": dataloader,
    "generator": torch.Generator(device="cuda").manual_seed(42),
    "use_cache" : False,
    "inference_args": {
        "num_inference_steps": 50,
        "guidance_scale": 3,
    },
    "root_output_dir": root_output_dir,
    "reference_view_index": 1,
    "sample_metrics": None,
    "noise_avg_samples":6,
    "reference_fidelity": 0.05,
    "test_inference":True
}
EXPERIMENT_CONFIGS.append(config)
# ======================================================================
# 5. 执行所有实验 (Execute All Experiments)
# ======================================================================
if __name__ == "__main__":
    print(f"即将开始执行 {len(EXPERIMENT_CONFIGS)} 个实验...")
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        # 调用我们的核心推理函数，传入当前实验的配置

        run_inference_task(config)
        torch.cuda.empty_cache()
    print("\n所有实验均已完成！")
