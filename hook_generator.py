import torch
from typing import List, Dict, Any, Type, Set
import re # 导入正则表达式库
from functools import partial # 导入 partial
import os,sys

# ----------------------------------------
# 1. Hook 基类和具体实现 (这部分保持不变)
# ----------------------------------------
class BaseHook:
    """所有 Hook 的基类。"""
    def __init__(self, target_module_class: Type[torch.nn.Module]):
        # 只有当 target_module_class 不是 None 时，才进行类型检查
        if target_module_class is not None and not issubclass(target_module_class, torch.nn.Module):
            raise ValueError("如果提供了 target_module_class，它必须是 torch.nn.Module 的子类。")
        self.target_module_class = target_module_class

    def __call__(self, module: torch.nn.Module, args: tuple, output: Any) -> Any:
        raise NotImplementedError("子类必须实现 __call__ 方法。")

    def description(self) -> str:
        return f"一个 {self.__class__.__name__}，作用于 {self.target_module_class.__name__} 模块。"

class ScaleHook(BaseHook):
    """一个可以缩放模块输出的 Hook。"""
    def __init__(self, target_module_class: Type[torch.nn.Module], scale_factor: float):
        super().__init__(target_module_class)
        self.scale_factor = scale_factor

    def __call__(self, module: torch.nn.Module, args: tuple, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output * self.scale_factor
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            modified_output = (output[0] * self.scale_factor,) + output[1:]
            return modified_output
        return output

    def description(self) -> str:
        return f"ScaleHook(factor={self.scale_factor}) on '{self.target_module_class.__name__}'"

class FeatureCaptureHook(BaseHook):
    """一个可以捕获并存储特征图的 Hook。"""
    captured_features: Dict[str, torch.Tensor] = {}
    def __init__(self, target_module_class: Type[torch.nn.Module], feature_name: str):
        super().__init__(target_module_class)
        self.feature_name = feature_name

    def __call__(self, module: torch.nn.Module, args: tuple, output: Any) -> Any:
        tensor_to_capture = None
        if isinstance(output, torch.Tensor):
            tensor_to_capture = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            tensor_to_capture = output[0]
        if tensor_to_capture is not None:
            FeatureCaptureHook.captured_features[self.feature_name] = tensor_to_capture.detach().cpu()
        return output

    def description(self) -> str:
        return f"FeatureCaptureHook(name='{self.feature_name}') on '{self.target_module_class.__name__}'"
    
    @classmethod
    def get_captured_features(cls) -> Dict[str, torch.Tensor]: return cls.captured_features
    @classmethod
    def clear_features(cls): cls.captured_features = {}
    
class SelectiveUpBlockScalingHook(BaseHook):
    """
    一个通过猴子补丁来修改 UpBlock2D 内部行为的特殊 Hook。

    它能定位到 UNet 中特定的 up_block 模块，并在拼接跨层连接的
    (hidden_states 和 res_hidden_states) 操作前，对它们分别进行缩放。
    这个版本是为推理模式特别优化的。
    """
    def __init__(self, 
                 upblock_indices: Set[int], 
                 hidden_states_scale: float = 1.0, 
                 res_hidden_states_scale: float = 1.0,
                 use_new_torch_logic: bool = False):
        """
        初始化 Hook。

        Args:
            upblock_indices (Set[int]): 目标 up_block 的索引集合。例如 {2, 3}。
            hidden_states_scale (float): 上采样路径的 hidden_states 的缩放因子。
            res_hidden_states_scale (float): 来自下采样路径的 res_hidden_states 的缩放因子。
            use_new_torch_logic (bool): 一个布尔标志，用于处理与 PyTorch 版本相关的逻辑。
                                        由主脚本在创建实例时传入判断结果。
        """
        # 这个 hook 的目标模块通过名字来定位，所以 target_module_class 设为 None。
        # HookManager 会通过 hasattr 判断其为“补丁型” hook。
        super().__init__(target_module_class=None)
        
        self.upblock_indices = upblock_indices
        self.hidden_states_scale = hidden_states_scale
        self.res_hidden_states_scale = res_hidden_states_scale
        self.use_new_torch_logic = use_new_torch_logic

        # 用于从模块名中解析 up_block 索引的正则表达式
        self.name_pattern = re.compile(r"up_blocks\.(\d+)")

    def is_target(self, name: str, module: torch.nn.Module) -> bool:
        """根据模块名称和类型判断是否为目标。"""
        # 动态导入您客制化的 UpBlock2D。
        # 这里假设调用此代码时，sys.path 已经被主脚本正确设置。
        try:
            sys.path.insert(0, "./models/")
            from diffusers.models.unet_2d_blocks import UpBlock2D
        except ImportError:
            print(f"错误: 无法从您的 diffusers 路径导入 UpBlock2D。Hook '{self.description()}' 将不会生效。")
            return False
        
        # 首先检查模块是否是我们想 hook 的类型
        if not isinstance(module, UpBlock2D):
            return False

        # 然后用正则表达式匹配模块名
        match = self.name_pattern.match(name)
        if not match:
            return False
        
        # 最后检查模块索引是否在我们指定的目标集合中
        up_idx = int(match.group(1))
        return up_idx in self.upblock_indices

    def create_patched_forward(self, module_instance: torch.nn.Module):
        """
        创建并返回一个新的、为推理模式优化的、打过补丁的 UpBlock2D.forward 方法。
        """
        # 为了在嵌套函数中清晰地访问 hook 的属性，我们先捕获它
        hook_self = self 

        # 定义新的 forward 函数。它的参数列表【完全匹配】您提供的原始 forward 函数
        def patched_forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            
            # 访问被 patch 模块的 resnets 属性
            for resnet in module_instance.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # --- <<< 核心修改点：在拼接前进行缩放 >>> ---
                scaled_hidden_states = hidden_states * hook_self.hidden_states_scale
                scaled_res_hidden_states = res_hidden_states * hook_self.res_hidden_states_scale
                
                hidden_states = torch.cat([scaled_hidden_states, scaled_res_hidden_states], dim=1)
                # --- <<< 修改结束 >>> ---

                # 【已简化】直接调用 resnet，移除了仅用于训练的 gradient_checkpointing 逻辑
                hidden_states = resnet(hidden_states, temb)

            # 处理 upsampler 的逻辑保持不变
            if module_instance.upsamplers is not None:
                for upsampler in module_instance.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        # 返回我们新创建的这个功能完备的函数
        return patched_forward
    
    def description(self) -> str:
        """返回一个描述此 hook 功能和配置的字符串。"""
        return (f"SelectiveUpBlockScalingHook on up_blocks:{self.upblock_indices} "
                f"with scales (hidden={self.hidden_states_scale}, res_hidden={self.res_hidden_states_scale})")

    def __call__(self, *args, **kwargs):
        """这个 hook 是通过猴子补丁工作的，不应该被直接调用。"""
        raise NotImplementedError("SelectiveUpBlockScalingHook is a patcher and not a standard callable hook.")

_CURRENT_ATTN_MODULE_NAME = [None]
def _set_attn_module_hook(name: str):
    def hook(module, input):
        _CURRENT_ATTN_MODULE_NAME[0] = name
    return hook

# ----------------------------------------
# 2. Hook Manager (这部分保持不变)
# ----------------------------------------
class HookManager:
    """管理 PyTorch hooks 的注册、移除和信息查询。"""
    def __init__(self, root_module: torch.nn.Module):
        if not isinstance(root_module, torch.nn.Module):
            raise ValueError("root_module 必须是 torch.nn.Module 的实例。")
        self.root_module = root_module
        self._hook_handles = [] # 存储标准 hook 的句柄
        self.active_hooks: List[BaseHook] = []
        
        # 新增：用于存储被猴子补丁修改的原始方法
        self._original_methods = []

    def register_hook(self, hook_instance: BaseHook):
        """注册一个 BaseHook 实例，并能自动处理标准型和补丁型 hook。"""
        if not isinstance(hook_instance, BaseHook):
            raise TypeError("hook_instance 必须是 BaseHook 的子类实例。")

        # 改进：使用鸭子类型判断是否为补丁型 Hook
        # 只要一个 hook 实例有 is_target 和 create_patched_forward 方法，就按补丁型处理
        is_patcher_hook = hasattr(hook_instance, 'is_target') and hasattr(hook_instance, 'create_patched_forward')

        if is_patcher_hook:
            print(f"Applying patch: {hook_instance.description()}")
            applied = False
            for name, module in self.root_module.named_modules():
                if hook_instance.is_target(name, module):
                    original_forward = module.forward
                    self._original_methods.append((module, original_forward))
                    
                    patched_forward = hook_instance.create_patched_forward(module)
                    module.forward = patched_forward
                    print(f"  - Patched module: '{name}'")
                    applied = True
            if not applied:
                print(f"  - Warning: No target modules found for patch '{hook_instance.description()}'")
            self.active_hooks.append(hook_instance)
            return

        # --- 以下是处理标准 hook 的逻辑 ---
        if hook_instance.target_module_class is None:
            print(f"警告: 标准 hook {hook_instance.__class__.__name__} 没有提供 target_module_class，已跳过。")
            return

        modules_to_hook = self._find_modules(hook_instance.target_module_class)
        if not modules_to_hook:
            print(f"警告: 在模型中没有找到 {hook_instance.target_module_class.__name__} 类型的模块。")
            return
        
        for module in modules_to_hook:
            handle = module.register_forward_hook(hook_instance)
            self._hook_handles.append(handle)
        
        self.active_hooks.append(hook_instance)

    def remove_hooks(self):
        """移除所有已注册的标准 hooks 和猴子补丁。"""
        # 移除标准 hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        # 新增：恢复所有被修改的原始方法
        print(f"Restoring {len(self._original_methods)} patched methods...")
        for module, original_method in self._original_methods:
            module.forward = original_method
        self._original_methods = []
        
        self.active_hooks = []

    # ... __init__, _find_modules, print_active_hooks, __enter__, __exit__ 等其他方法保持不变 ...
    # (确保 __exit__ 调用 remove_hooks)
    def _find_modules(self, module_class: Type[torch.nn.Module]) -> List[torch.nn.Module]:
        return [m for m in self.root_module.modules() if isinstance(m, module_class)]
    
    def print_active_hooks(self):
        print("="*20 + " Active Hooks " + "="*20)
        if not self.active_hooks: print("  No hooks are currently active.")
        else:
            for i, hook in enumerate(self.active_hooks): print(f"  [{i+1}] {hook.description()}")
        print("="*54)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        FeatureCaptureHook.clear_features()
        print("--- Hooks automatically removed and resources cleaned. ---")

# ----------------------------------------
# 3. 工厂函数 (修改后)
# ----------------------------------------
def create_hook_manager(root_module: torch.nn.Module, hooks_to_register: List[BaseHook]) -> HookManager:
    """
    根据提供的 hook 实例列表，创建并配置一个 HookManager。

    Args:
        root_module (torch.nn.Module): 要注入 hook 的根模块 (例如, pipe.unet)。
        hooks_to_register (List[BaseHook]): 一个包含 BaseHook 实例的列表。

    Returns:
        HookManager: 一个配置好的 HookManager 实例，用于 `with` 语句。
    """
    manager = HookManager(root_module)
    print(f"--- Loading {len(hooks_to_register)} custom hooks... ---")
    for hook_instance in hooks_to_register:
        manager.register_hook(hook_instance)
    return manager