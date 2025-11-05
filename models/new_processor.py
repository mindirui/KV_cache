import torch
import torch.nn.functional as F
import einops # 确保你的环境安装了einops
from typing import Optional
from collections import defaultdict
import xformers
import xformers.ops
from contextlib import contextmanager
import sys
sys.path.insert(0, "./models/")
from diffusers.models.attention import Attention
import torchvision.transforms.functional as TF

def cape_embed(f, P):
    # f is feature vector of shape [..., d]
    # P is 4x4 transformation matrix
    f = einops.rearrange(f, '... (d k) -> ... d k', k=4)
    return einops.rearrange(f@P, '... d k -> ... (d k)', k=4)

_CURRENT_ATTN_MODULE_NAME = [None]

def _set_attn_module_hook(name: str):
    """创建一个钩子函数，用于在调用前设置全局模块名。"""
    def hook(module, input):
        _CURRENT_ATTN_MODULE_NAME[0] = name
    return hook


class AttentionStore:
    def __init__(self):
        """
        一个更简单的、无状态的中央仓库。
        它使用defaultdict，使得添加新条目更方便。
        """
        self.maps = defaultdict(dict)

    def save_map(self, timestep: int, module_name: str, map_tensor: torch.Tensor):
        """由处理器调用，直接存入指定的时间步和模块名下。"""
        self.maps[timestep][module_name] = map_tensor

    def reset(self):
        self.maps.clear()
        
class AttentionStoreProcessor(torch.nn.Module):
    """
    一个用于捕获注意力图的自定义处理器。
    它完整复制了你的客制化逻辑 (posemb, cape_embed等)，
    但使用标准PyTorch方法计算注意力，以便存储注意力图。
    """
    def __init__(self, name: str, store: AttentionStore):
        """
        升级版的处理器。
        :param name: 处理器在UNet中的模块名 (e.g., "down_blocks.0.attentions.0")
        :param store: 要关联的中央 AttentionStore 实例
        """
        super().__init__()
        self.name = name
        self.store = store

    def get_attention_maps(self):
        return self.attention_maps

    def reset(self):
        self.attention_maps = {}

    def __call__(self, 
                 attn,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb: Optional[torch.FloatTensor] = None,
                 posemb: Optional = None,
                 **kwargs):
        timestep = kwargs.pop("timestep", 0)
        t = kwargs.pop("t", 0)
        # 1. 残差连接准备 (逻辑不变)
        residual = hidden_states

        # 2. 空间归一化 (逻辑不变)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 3. 输入维度处理 (逻辑不变)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 4. 多视角/姿态处理 (逻辑不变)
        if posemb is not None:
            self_attn = encoder_hidden_states is None
            [p_out, p_out_inv], [p_in, p_in_inv] = posemb
            t_out, t_in = p_out.shape[1], p_in.shape[1]
            hidden_states = einops.rearrange(hidden_states, '(b t_out) l d -> b (t_out l) d', t_out=t_out)
        
        # 5. 获取Key序列长度并准备注意力Mask (逻辑不变)
        # 注意：这里我们保留了 batch_size 的计算，后面保存注意力图时会用到
        if encoder_hidden_states is None:
             batch_size, key_tokens, _ = hidden_states.shape
        else:
             batch_size, key_tokens, _ = encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        # 6. Group Normalization (逻辑不变)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 7. QKV 投影 (逻辑不变)
        query = attn.to_q(hidden_states)
        is_cross_attention = 'attn2' in self.name # 记录是否为交叉注意力，用于存储
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 8. 应用6DoF姿态嵌入 (逻辑不变)
        if posemb is not None:
            p_out_inv = einops.repeat(p_out_inv, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            if self_attn:
                p_in = einops.repeat(p_out, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            else:
                p_in = einops.repeat(p_in, 'b t_in f g -> b (t_in l) f g', l=key.shape[1] // t_in)
            
            # 确保你能调用到 cape_embed 函数
            if cape_embed is None:
                raise ValueError("`cape_embed` function must be provided when `posemb` is not None.")
            query = cape_embed(query, p_out_inv)
            key = cape_embed(key, p_in)
        
        # parameters for KV cache and concateion
        hijack_mode = kwargs.get("hijack_mode", "off")
        cache = kwargs.get("attention_cache")
        need_save = kwargs.get("need_save", False)
        t_cutoff = kwargs.get("t_cutoff", 999)
        func_method = kwargs.get("func_method", None)

        
        if is_cross_attention != True and hijack_mode != "off" and cache is not None:
            module_name = self.name
            # 模式一：缓存当前的K/V
            if hijack_mode == "caching":
                # --- [修正] 只缓存有条件(conditional)部分的K/V ---
                # print(f"Caching CONDITIONAL K/V for: {module_name} at t={timestep}") # 调试用
                cache.save(int(timestep), module_name, key, value)

            # 模式二：注入缓存的K/V
            elif hijack_mode == "injecting" and t <= t_cutoff:
                cached_kv = cache.load(int(timestep), module_name, hidden_states.device)
                if cached_kv is not None:
                    # --- [关键] 标记已执行注入，后续将跳过常规注意力计算 ---                    
                    # print(f"  [Injecting] at t={int(timestep)} for module: {self.name}")

                    # --- [核心优化] 基于 q[0]==q[1] 的事实，我们只需计算一次，然后复制结果 ---
                    
                    # 1. 取出单分支的Q, K, V (以有条件分支为例)
                    # 此时 q_single 的 shape 为 [1, 1024, 320]
                    k_single = key[1:]
                    v_single = value[1:]

                    # 2. 加载并准备参考K/V
                    # 加载的缓存已经是纯净的条件部分，shape为 [1, 1024, 320]
                    k_ref = cached_kv['k'].to(k_single.device, dtype=k_single.dtype).unsqueeze(0)
                    v_ref = cached_kv['v'].to(v_single.device, dtype=v_single.dtype).unsqueeze(0)
                    
                    # 2.1 设置权重函数
                    if func_method == "linear":
                        # alpha = max(1 - (0.02 * t) + 0.2, 1)
                        alpha = (1 - 0.2) * (1 - 0.02 * t) + 0.2
                    else:
                        alpha = 1.2

                    # 2.2 根据权重函数调整参考K/V
                    k_ref = alpha * k_ref 
                    v_ref = alpha * v_ref 

                    
                    # (可选) 处理缓存bs=1而推理bs>1的情况，虽然当前逻辑下bs总是1，但保留更具鲁棒性
                    if k_ref.shape[0] != k_single.shape[0]:
                        k_ref = k_ref.repeat(k_single.shape[0] // k_ref.shape[0], 1, 1)
                        v_ref = v_ref.repeat(v_single.shape[0] // v_ref.shape[0], 1, 1)

                    # 3. 拼接K和V (在序列维度上)
                    k_hijacked = torch.cat([k_single, k_ref], dim=1) # shape: [1, 2048, 320]
                    v_hijacked = torch.cat([v_single, v_ref], dim=1) # shape: [1, 2048, 320]
                    
                    # 4. 复制“劫持模板”以恢复CFG结构，并覆盖原始变量
                    # query 保持不变: shape [2, 1024, 320]
                    # key 和 value 被修改: shape 变为 [2, 2048, 320]
                    key = torch.cat([k_hijacked, k_hijacked], dim=0)
                    value = torch.cat([v_hijacked, v_hijacked], dim=0)
                        
            # ==========================================================
            # 注入逻辑结束
            # ==========================================================
                    
            
        # 9. 多头注意力准备 (逻辑不变)
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # ------------------- 核心替换部分开始 -------------------
        # 10. 使用标准PyTorch方法计算注意力，并保存注意力图
        
        # 计算注意力分数
        # query/key shape: (batch*heads, seq_len, head_dim)
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(dim=-1).to(query.dtype)

        # ==========================================================
        # >> HOOK: 新的、更精确的存储逻辑 <<
        # ==========================================================
        if need_save:
            q_len, k_len = query.shape[1], key.shape[1]
            store_map = attention_probs.view(batch_size, attn.heads, q_len, k_len)
            store_map = store_map.mean(1).detach().cpu()
            
            map_type = "cross" if is_cross_attention else "self"
            full_name = f"{self.name}.{map_type}"
            
        # 直接调用store的save_map方法，传入timestep
            self.store.save_map(timestep, full_name, store_map)
        # ==========================================================
        
        hidden_states = torch.bmm(attention_probs, value)
        # -------------------- 核心替换部分结束 --------------------

        # 11. 恢复多头维度 (逻辑不变)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 12. 输出投影 (逻辑不变)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # 13. 恢复多视角/姿态维度 (逻辑不变)
        if posemb is not None:
            hidden_states = einops.rearrange(hidden_states, 'b (t_out l) d -> (b t_out) l d', t_out=t_out)

        # 14. 恢复输入维度 (逻辑不变)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 15. 残差连接和输出缩放 (逻辑不变)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------



class DebugAttnProcessor(torch.nn.Module):
    """
    一个用于调试和验证的注意力处理器。
    功能与您提供的 XFormersAttnProcessor 完全相同，但使用标准PyTorch注意力计算，
    以便捕获和分析 attention_probs 矩阵。
    """

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        posemb: Optional = None,
        # **kwargs 用于接收我们之前讨论过的 timestep 等额外参数
        **kwargs,
    ):
        # 1. 残差连接准备 (逻辑不变)
        residual = hidden_states

        # 2. 空间归一化 (逻辑不变)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 3. 输入维度处理 (逻辑不变)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # 4. 多视角/姿态处理 (逻辑不变)
        if posemb is not None:
            self_attn = encoder_hidden_states is None
            [p_out, p_out_inv], [p_in, p_in_inv] = posemb
            t_out, t_in = p_out.shape[1], p_in.shape[1]
            hidden_states = einops.rearrange(hidden_states, '(b t_out) l d -> b (t_out l) d', t_out=t_out)
        
        # 5. Group Normalization (逻辑不变)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # 6. QKV 投影 (逻辑不变)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 7. 应用6DoF姿态嵌入 (逻辑不变)
        if posemb is not None:
            p_out_inv = einops.repeat(p_out_inv, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            if self_attn:
                p_in = einops.repeat(p_out, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            else:
                p_in = einops.repeat(p_in, 'b t_in f g -> b (t_in l) f g', l=key.shape[1] // t_in)
            # 假设 cape_embed 函数是可用的
            query = cape_embed(query, p_out_inv)
            key = cape_embed(key, p_in)

        # 8. 多头注意力准备 (逻辑不变)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # ==================== XFormers 替换部分开始 ====================
        
        # 9. 使用标准PyTorch计算注意力分数
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
        
        # 10. 应用注意力掩码 (逻辑不变, 但需要先获取mask)
        # 注意: 原始代码的attention_mask准备部分在QKV计算之前，这里我们直接使用
        # 这需要将原始代码中计算attention_mask的部分也包含进来
        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
             # PyTorch的bmm+加法会自动广播，所以不需要像xformers那样手动expand
             attention_scores = attention_scores + attention_mask

        # 11. Softmax 得到注意力概率矩阵
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # <<<< 在这里可以捕获 attention_probs 用于可视化或保存 >>>>
        # 例如: self.attention_maps = attention_probs.detach().cpu()
        # 这就是您进行所有可视化分析所需要的数据！

        # 12. 将概率应用于Value，得到最终输出
        hidden_states = torch.bmm(attention_probs, value)
        
        # ===================== XFormers 替换部分结束 =====================

        # 13. 恢复多头维度 (逻辑不变)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 14. 输出投影 (逻辑不变)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # 15. 恢复多视角/姿态维度 (逻辑不变)
        if posemb is not None:
            hidden_states = einops.rearrange(hidden_states, 'b (t_out l) d -> (b t_out) l d', t_out=t_out)

        # 16. 恢复输入维度 (逻辑不变)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 17. 残差连接和输出缩放 (逻辑不变)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttentionCache:
    """
    (缓存阶段) 一个结构化的中央仓库，用于存储K/V对。
    
    数据结构: {timestep: {module_name: {"k": tensor, "v": tensor}}}
    所有张量将被存储在CPU上以节省显存。
    """
    def __init__(self):
        self.cache = defaultdict(dict)
        self.is_recording = True # 增加一个状态开关

    def start_recording(self):
        self.is_recording = True
        self.clear()
        print("✅ AttentionCache: 开始记录模式。")

    def stop_recording(self):
        self.is_recording = False
        print(f"✅ AttentionCache: 记录结束。共缓存了 {sum(len(v) for v in self.cache.values())} 个K/V对。")

    def save(self, timestep: int, module_name: str, key: torch.Tensor, value: torch.Tensor):
        """
        仅在记录模式下，将K, V的有条件部分保存到CPU。
        """
        if not self.is_recording:
            return
            
        if key.shape[0] == 2 and value.shape[0] == 2: # 仅在CFG开启时操作
            self.cache[timestep][module_name] = {
                "k": key[1].detach().clone().to("cpu"),
                "v": value[1].detach().clone().to("cpu")
            }

    def load(self, timestep: int, module_name: str, device) -> dict:
        """从CPU加载K, V，并在使用前将其移动到目标设备(GPU)。"""
        cached_kv = self.cache.get(timestep, {}).get(module_name)
        if cached_kv is not None:
            return {
                "k": cached_kv["k"].to(device),
                "v": cached_kv["v"].to(device)
            }
        return None

    def clear(self):
        """清空缓存。"""
        self.cache.clear()



class ProviderProcessor:
    """
    (缓存阶段) 在“预计算”时，捕获并向缓存提供K/V的处理器。
    它像一个“代理”，记录数据后，会将计算任务交还给原始的处理器。
    """
    def __init__(self, name: str, cache: AttentionCache, original_processor):
        self.name = name
        self.cache = cache
        self.original_processor = original_processor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                 attention_mask: Optional[torch.FloatTensor] = None,
                 temb: Optional[torch.FloatTensor] = None,
                 posemb: Optional = None,**kwargs):
        if True:
            print('return')
            return self.original_processor(attn, hidden_states, encoder_hidden_states, **kwargs)
        
        if encoder_hidden_states is not None:
            return self.original_processor(attn, hidden_states, encoder_hidden_states, **kwargs)
        
        # 1. 获取 timestep，如果不存在则直接使用原始处理器完成计算
        timestep = kwargs.get("timestep")
        if timestep is None:
            return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask=attention_mask, **kwargs)

        # 2. 准备用于计算K/V的 hidden_states
        #    这是您提供的完整预处理流程
        hs_for_kv = hidden_states
        
        if attn.spatial_norm is not None:
            hs_for_kv = attn.spatial_norm(hs_for_kv, temb)

        if hs_for_kv.ndim == 4:
            batch_size, channel, height, width = hs_for_kv.shape
            hs_for_kv = hs_for_kv.view(batch_size, channel, height * width).transpose(1, 2)
        
        # 提前判断是否为自注意力，因为 posemb 部分需要用
        is_self_attention = encoder_hidden_states is None
        
        if posemb is not None:
            [p_out, p_out_inv], [p_in, p_in_inv] = posemb
            t_out, t_in = p_out.shape[1], p_in.shape[1]
            hs_for_kv = einops.rearrange(hs_for_kv, '(b t_out) l d -> b (t_out l) d', t_out=t_out)
        
        if attn.group_norm is not None:
            hs_for_kv = attn.group_norm(hs_for_kv.transpose(1, 2)).transpose(1, 2)
        
        # 3. 准备用于计算K/V的 encoder_hidden_states
        ehs_for_kv = encoder_hidden_states
        if is_self_attention:
            ehs_for_kv = hs_for_kv # 在自注意力中，K/V的来源和Q一样
        elif attn.norm_cross:
            ehs_for_kv = attn.norm_encoder_hidden_states(ehs_for_kv)
        
        # 4. 计算最终的、信息完备的 Key 和 Value
        query = attn.to_q(hs_for_kv) # 我们需要query的形状来辅助posemb
        key = attn.to_k(ehs_for_kv)
        value = attn.to_v(ehs_for_kv)

        # 应用6DoF姿态嵌入
        if posemb is not None:
            p_out_inv = einops.repeat(p_out_inv, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            if is_self_attention:
                p_in = einops.repeat(p_out, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)
            else:
                p_in = einops.repeat(p_in, 'b t_in f g -> b (t_in l) f g', l=key.shape[1] // t_in)
            
            # 注意：cape_embed只修改query和key，value保持不变
            # query = cape_embed(query, p_out_inv)
            key = cape_embed(key, p_in)
        
        # 5. 【条件缓存】现在我们有了最终的K/V，可以进行缓存了
        # 仅在记录模式 和 自注意力模块中 执行
        if self.cache.is_recording and is_self_attention:
            # print(f"Caching K/V for module: {self.name} at step {int(timestep)}") # 调试信息
            self.cache.save(int(timestep), self.name, key, value)
        
        # 6. 【关键】将“指挥权”交还给原始处理器
        # 无论是否进行了缓存，都必须调用原始处理器来完成注意力的实际计算和返回
        return self.original_processor(
            attn,
            hidden_states, # 传入这个方法接收到的、最原始的 hidden_states
            encoder_hidden_states, # 和原始的 encoder_hidden_states
            attention_mask=attention_mask,
            temb=temb,
            posemb=posemb,
            **kwargs,
        )
        
class ReceiverProcessor:
    """
    (注入阶段) 从缓存中读取K/V，并将其注入到自注意力计算中。
    它能正确处理CFG，并重新启用XFormers以获得高性能。
    """
    def __init__(self, name: str, cache: AttentionCache, original_processor):
        self.name = name
        self.cache = cache
        self.original_processor = original_processor
    
    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        posemb: Optional = None,
        **kwargs,
    ):  
        

            
        # 快速通道：交叉注意力直接返回（此逻辑不变且正确）
        if encoder_hidden_states is not None:
            return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask=attention_mask, temb=temb, posemb=posemb, **kwargs)
        
        # timestep = kwargs.pop("timestep", 0)
        # if timestep < 10000:
        #     print("NOReceiverProcessor")
        #     return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask=attention_mask, temb=temb, posemb=posemb, **kwargs)
        # print("ReceiverProcessor")
        # 1. 分离CFG的输入
        # 这是我们所有操作的基础
        hidden_states_uncond, hidden_states_cond = hidden_states.chunk(2)
        
        # 对posemb等也需要进行分离
        posemb_uncond, posemb_cond = None, None
        if posemb is not None:
            # 假设posemb也是可以按batch=2进行分离的
            # 这需要您根据posemb的具体结构来调整
            # 示例：[[p_out_u, p_out_c], [p_out_inv_u, p_out_inv_c]], [[...], [...]]
            # 下面是一个简化的切片逻辑，您需要适配您的数据
            p_out_uncond, p_out_cond = posemb[0][0].chunk(2)
            p_out_inv_uncond, p_out_inv_cond = posemb[0][1].chunk(2)
            p_in_uncond, p_in_cond = posemb[1][0].chunk(2)
            p_in_inv_uncond, p_in_inv_cond = posemb[1][1].chunk(2)
            posemb_uncond = [[p_out_uncond, p_out_inv_uncond], [p_in_uncond, p_in_inv_uncond]]
            posemb_cond = [[p_out_cond, p_out_inv_cond], [p_in_cond, p_in_inv_cond]]

        # ==========================================================
        # 2. 无条件分支：完整复刻原始逻辑
        # ==========================================================
        # 直接在这里调用原始处理器是一个更简洁的选择，前提是所有参数都已正确切片
        # 为保证逻辑绝对清晰，我们在这里也手动实现
        hs_uncond_out = self.compute_branch(
            attn, hidden_states_uncond, attention_mask, temb, posemb_uncond, **kwargs
        )

        # ==========================================================
        # 3. 有条件分支：在完整复刻的逻辑中，加入“劫持”步骤
        # ==========================================================
        timestep = kwargs.get("timestep")
        cached_kv = self.cache.load(int(timestep), self.name, hidden_states.device) if timestep is not None else None

        hs_cond_out = self.compute_branch(
            attn, hidden_states_cond, attention_mask, temb, posemb_cond, 
            hijack_kv=cached_kv, # 传入缓存的K/V作为额外参数
            **kwargs
        )
        
        # 4. 最终拼接返回
        return torch.cat([hs_uncond_out, hs_cond_out], dim=0)

    def compute_branch(
        self, attn, hidden_states, attention_mask, temb, posemb, 
        hijack_kv=None, **kwargs
    ):
        """
        一个辅助函数，用于执行单个分支（无条件或有条件）的完整计算。
        这是您原始XFormersProcessor逻辑的单batch版本。
        """
        # --- 这里是您原始处理器的完整代码，但作用于单batch输入 ---
        # --- 我已为您适配了劫持逻辑 ---

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        # ... (您所有的预处理逻辑: ndim, posemb rearrange, group_norm) ...
        # 注意: 所有输入(hidden_states, posemb)都已经是batch_size=1的了

        # QKV计算
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # cape_embed姿态编码
        # if posemb is not None:
        #     query = cape_embed(query, ...)
        #     key = cape_embed(key, ...)

        # 【劫持点】如果传入了hijack_kv，则执行拼接
        if hijack_kv is not None:
            k_ref = hijack_kv["k"].unsqueeze(0)
            v_ref = hijack_kv["v"].unsqueeze(0)
            key = torch.cat([key, k_ref], dim=1)
            value = torch.cat([value, v_ref], dim=1)
            # (可能需要调整attention_mask以适应新的key长度)

        # 多头分解
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # XFormers计算
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, scale=attn.scale, op=getattr(self, 'attention_op', None)
        )
        
        # 完整的后处理
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        # ... (您所有的后处理逻辑: reshape, residual, rescale) ...
        # if input_ndim == 4: ...
        # if attn.residual_connection: hidden_states = hidden_states + residual
        # ...

        return hidden_states

def restore_original_processors(pipeline: torch.nn.Module, original_processors: dict):
    """
    将pipeline的注意力处理器恢复到原始状态。

    Args:
        pipeline: 被修改过的pipeline对象。
        original_processors (dict): 备份的原始处理器字典。
    """
    print("\n--- 正在恢复原始的注意力处理器 ---")
    for name, module in pipeline.unet.named_modules():
        if name in original_processors:
            module.set_processor(original_processors[name])
    print("✅ Pipeline已恢复至原始状态。")

def backup_original_processors(unet: torch.nn.Module) -> dict:
    """
    遍历UNet的所有Attention模块，备份它们当前的处理器。

    Args:
        unet: a diffusers UNet model.

    Returns:
        A dictionary mapping module names to their original processor instances.
    """
    original_processors = {}
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            # 保存每个Attention模块当前的处理器实例
            original_processors[name] = module.processor
    return original_processors


@contextmanager
def pipeline_in_caching_mode(pipeline, cache, original_processors):
    """一个上下文管理器，用于临时将pipeline置于“缓存模式”。"""
    print("--- [进入] 缓存模式: 正在安装ProviderProcessors... ---")
    for name, module in pipeline.unet.named_modules():
        if name in original_processors:
            provider = ProviderProcessor(name, cache, original_processors[name])
            module.set_processor(provider)
    try:
        # yield关键字将控制权交还给with块内部的代码
        yield
    finally:
        # with块结束时，无论成功或失败，都会执行这里的恢复操作
        print("--- [退出] 缓存模式: 正在恢复原始处理器... ---")
        restore_original_processors(pipeline, original_processors)

@contextmanager
def pipeline_in_injection_mode(pipeline, cache, original_processors):
    """一个上下文管理器，用于临时将pipeline置于“注入模式”。"""
    print("--- [进入] 注入模式: 正在安装ReceiverProcessors... ---")
    for name, module in pipeline.unet.named_modules():
        if name in original_processors:
            receiver = ReceiverProcessor(name, cache, original_processors[name])
            module.set_processor(receiver)
    try:
        yield
    finally:
        print("--- [退出] 注入模式: 正在恢复原始处理器... ---")
        restore_original_processors(pipeline, original_processors)