import torch
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class ImageCentricConfig:
    """Configuration for image-centric head detection"""
    tau: float = 5.0
    rho: float = 0.0010
    summ: float = 0.78
    dim_sink: list = None
    verbose: bool = False
    
    def __post_init__(self):
        if self.dim_sink is None:
            self.dim_sink = [2533, 1415]


class SinkTokenDetector:
    """Detects sink tokens based on RMS norm values."""
    
    def __init__(self, config: ImageCentricConfig):
        self.config = config
        self.tau = config.tau
        self.dim_sink = config.dim_sink
        self.sink_indices_per_layer = defaultdict(lambda: None)
    
    @staticmethod
    def rmsnorm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply RMS normalization."""
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + eps)
        return normalized
    
    def detect_sink_tokens(self, hidden_states: torch.Tensor, layer_idx: int, image_start: int = None, image_len: int = None) -> torch.Tensor:
        """Detect sink tokens from hidden states using RMS norm."""
        
        # ⭐⭐⭐ 遵循源代码逻辑 ⭐⭐⭐
        # 源代码: rms_norm_hs = torch.abs(cls.rmsnorm(hs))
        rms_norm_hs = torch.abs(self.rmsnorm(hidden_states))  # [bsz, tok, dim]
        
        # ⭐⭐⭐ 分离检测逻辑和可视化逻辑 ⭐⭐⭐
        
        # 1. 用于 SINK TOKEN 检测（遵循源代码）
        rms_values = torch.stack(
            [rms_norm_hs[:, :, idx] for idx in self.dim_sink], 
            dim=-1
        )  # [bsz, tok, 2]
        max_rms_values = torch.max(rms_values, dim=-1)[0]  # [bsz, tok]
        
        # 2. 用于可视化（使用原始激活强度，不是 abs(rmsnorm)）
        # 计算 L2 norm：sqrt(mean(x^2))
        raw_activation = torch.sqrt((hidden_states ** 2).mean(dim=-1))  # [bsz, tok]

        # ⭐⭐⭐ 详细的激活值分析和保存 ⭐⭐⭐
        if layer_idx == 0 and image_start is not None and image_len is not None:
            print(f"\n{'='*80}")
            print(f"📊 [Layer {layer_idx}] ACTIVATION ANALYSIS")
            print(f"{'='*80}")
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            print(f"\n🔍 Basic Info:")
            print(f"   - Sequence length: {seq_len}")
            print(f"   - Hidden dimension: {hidden_dim}")
            print(f"   - Image region: [{image_start}, {image_start + image_len})")
            print(f"   - Text before image: {image_start} tokens")
            print(f"   - Image tokens: {image_len} tokens")
            print(f"   - Text after image: {seq_len - image_start - image_len} tokens")
            
            # ⭐ 使用原始激活强度（L2 norm）用于可视化
            # activations = raw_activation[0].detach().cpu().numpy()
            activations = raw_activation[0].detach().cpu().float().contiguous().numpy()
            
            # 分割不同区域的激活值
            text_before = activations[:image_start]
            image_tokens = activations[image_start:image_start + image_len]
            text_after = activations[image_start + image_len:]
            
            print(f"\n📈 Activation Statistics (L2 norm: sqrt(mean(x^2))):")
            print(f"\n   Text Before Image ({len(text_before)} tokens):")
            if len(text_before) > 0:
                print(f"      - Mean: {text_before.mean():.6f}")
                print(f"      - Std:  {text_before.std():.6f}")
                print(f"      - Min:  {text_before.min():.6f}")
                print(f"      - Max:  {text_before.max():.6f}")
            
            print(f"\n   Image Tokens ({len(image_tokens)} tokens):")
            print(f"      - Mean: {image_tokens.mean():.6f}")
            print(f"      - Std:  {image_tokens.std():.6f}")
            print(f"      - Min:  {image_tokens.min():.6f}")
            print(f"      - Max:  {image_tokens.max():.6f}")
            
            print(f"\n   Text After Image ({len(text_after)} tokens):")
            if len(text_after) > 0:
                print(f"      - Mean: {text_after.mean():.6f}")
                print(f"      - Std:  {text_after.std():.6f}")
                print(f"      - Min:  {text_after.min():.6f}")
                print(f"      - Max:  {text_after.max():.6f}")
            
            # 计算差异
            if len(text_before) > 0 and len(text_after) > 0:
                text_all = activations[list(range(image_start)) + list(range(image_start + image_len, seq_len))]
                ratio = image_tokens.mean() / text_all.mean()
                print(f"\n   📊 Comparison:")
                print(f"      - Image mean / Text mean: {ratio:.4f}x")
                print(f"      - Image max / Text max: {image_tokens.max() / text_all.max():.4f}x")
                
                # 诊断建议
                if ratio < 0.95:
                    print(f"\n   ⚠️  WARNING: Image activation is LOWER than text!")
                    print(f"       Possible causes:")
                    print(f"       1. Using post-attention hidden_states (smoothed)")
                    print(f"       2. Image region boundary might be wrong")
                elif ratio > 1.2:
                    print(f"\n   ✅ Image tokens show HIGHER activation (sink token pattern detected)")
                else:
                    print(f"\n   🟡 Weak or no sink token pattern")
            
            # ⭐⭐⭐ 额外：展示 dim_sink 维度的检测结果 ⭐⭐⭐
            print(f"\n   🔬 Sink Token Detection (using dim_sink = {self.dim_sink}):")
            
            # 展示特定维度的激活值
            for dim_idx in self.dim_sink:
                dim_values = rms_norm_hs[0, :, dim_idx].cpu().numpy()
                image_dim = dim_values[image_start:image_start + image_len]
                text_before_dim = dim_values[:image_start] if image_start > 0 else np.array([])
                text_after_dim = dim_values[image_start + image_len:] if image_start + image_len < seq_len else np.array([])
                
                if len(text_before_dim) > 0 and len(text_after_dim) > 0:
                    text_dim = np.concatenate([text_before_dim, text_after_dim])
                elif len(text_before_dim) > 0:
                    text_dim = text_before_dim
                else:
                    text_dim = text_after_dim
                
                print(f"      Dim {dim_idx}:")
                print(f"         Image: mean={image_dim.mean():.4f}, max={image_dim.max():.4f}")
                print(f"         Text:  mean={text_dim.mean():.4f}, max={text_dim.max():.4f}")
                print(f"         Ratio: {image_dim.mean() / text_dim.mean():.4f}x")
            
            # 展示检测到的 sink tokens
            detected_sinks = torch.nonzero(max_rms_values > self.tau)
            if detected_sinks.numel() > 0:
                sink_positions = detected_sinks[:, 1].unique().cpu().numpy()
                image_sinks = sink_positions[(image_start <= sink_positions) & (sink_positions < image_start + image_len)]
                text_sinks = sink_positions[~np.isin(sink_positions, image_sinks)]
                
                print(f"\n      Detection Results (tau={self.tau}):")
                print(f"         Total sink tokens detected: {len(sink_positions)}")
                print(f"         In image region: {len(image_sinks)} ({len(image_sinks)/image_len*100:.1f}% of image)")
                print(f"         In text region: {len(text_sinks)}")
                
                if len(image_sinks) > 0:
                    print(f"         Sample image sink positions: {image_sinks[:5].tolist()}")
            else:
                print(f"\n      Detection Results (tau={self.tau}):")
                print(f"         ❌ No sink tokens detected!")
                print(f"         Suggestion: Lower tau threshold (current: {self.tau})")
                
                # 提供建议的 tau 值
                max_val = max_rms_values.max().item()
                suggested_tau = max_val * 0.8
                print(f"         Max detection value: {max_val:.4f}")
                print(f"         Suggested tau: {suggested_tau:.4f}")
            
            # ⭐⭐⭐ 保存数据用于可视化 ⭐⭐⭐
            save_data = {
                'activations': activations,  # 使用 L2 norm
                'image_start': image_start,
                'image_len': image_len,
                'seq_len': seq_len,
                'layer_idx': layer_idx,
                'method': 'L2_norm',
                'detection_method': 'abs_rmsnorm_specific_dims'
            }
            np.save('/root/LLaVA-MoICE/llava/model/language_model/llama/activation/layer0.npy', save_data)
            print(f"\n💾 Saved activation data to: /root/LLaVA-MoICE/llava/model/language_model/llama/activation/layer0.npy")
            print(f"{'='*80}\n")
        
        # ⭐⭐⭐ 遵循源代码的检测逻辑 ⭐⭐⭐
        # 源代码: indices = torch.nonzero(max_rms_values > cls.tau)[:, 1]
        sink_positions = torch.nonzero(max_rms_values > self.tau)
        
        if sink_positions.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=hidden_states.device)
        
        # Extract token indices (第二维是 token 维度)
        sink_indices = sink_positions[:, 1].unique()
        
        # Cache
        self.sink_indices_per_layer[layer_idx] = sink_indices
        
        return sink_indices
    
    def clear_cache(self):
        """Clear all cached sink indices."""
        self.sink_indices_per_layer.clear()


class ImageCentricHeadDetector:
    """Detector for identifying image-centric attention heads using sink tokens."""
    
    def __init__(self, config: Optional[ImageCentricConfig] = None):
        self.config = config or ImageCentricConfig()
        self.sink_detector = SinkTokenDetector(self.config)
        self.detection_cache = {}
    
    def detect_from_attention(
        self,
        attn: torch.Tensor,
        hidden_states: torch.Tensor,
        image_start: int,
        image_len: int,
        layer_idx: int
    ) -> torch.Tensor:
        """Detect image-centric heads from attention weights and hidden states."""
        im, pa = image_start, image_len
        
        # Step 1: Detect sink tokens
        sink_indices = self.sink_detector.detect_sink_tokens(
            hidden_states, 
            layer_idx,
            image_start=im,
            image_len=pa
        )
        
        if sink_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=attn.device).reshape(0, 3)
        
        # Step 2: Filter sink indices within image region
        vis_sink_inds = sink_indices[(im <= sink_indices) & (sink_indices < im + pa)]
        
        if len(vis_sink_inds) == 0:
            return torch.tensor([], dtype=torch.long, device=attn.device).reshape(0, 3)
        
        # Step 3: Extract attention to image region
        image_attn = attn[:, :, :, im:im+pa]
        
        # Step 4: Calculate metrics
        local_sink_inds = vis_sink_inds - im
        sink_attn = torch.sum(image_attn[:, :, :, local_sink_inds], dim=-1)
        total_attn = torch.sum(image_attn, dim=-1) + 1e-6
        
        portion = sink_attn / total_attn
        summation = torch.sum(image_attn, dim=-1)
        
        # Step 5: Apply conditions
        portion_condition = portion <= self.config.rho
        summation_condition = summation >= self.config.summ
        
        candidate_coords = torch.nonzero(portion_condition & summation_condition)
        
        # Cache detection results
        self.detection_cache[layer_idx] = candidate_coords.clone()
        
        return candidate_coords
    
    def get_image_centric_head_mask(self, layer_idx: int, num_heads: int, device: torch.device) -> torch.Tensor:
        """Get a boolean mask indicating which heads are image-centric."""
        if layer_idx not in self.detection_cache:
            return torch.zeros(num_heads, dtype=torch.bool, device=device)
        
        coords = self.detection_cache[layer_idx]
        if coords.numel() == 0:
            return torch.zeros(num_heads, dtype=torch.bool, device=device)
        
        unique_heads = torch.unique(coords[:, 1])
        mask = torch.zeros(num_heads, dtype=torch.bool, device=device)
        mask[unique_heads] = True
        
        return mask
    
    def clear_cache(self):
        """Clear detection cache."""
        self.detection_cache.clear()
        self.sink_detector.clear_cache()
