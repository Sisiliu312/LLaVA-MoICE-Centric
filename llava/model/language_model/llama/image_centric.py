
import torch
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ImageCentricConfig:
    """Configuration for image-centric head detection"""
    tau: float = 5.0
    rho: float = 0.0010
    summ: float = 0.8
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
        # Apply RMS norm
        rms_norm_hs = torch.abs(self.rmsnorm(hidden_states))
        
        # Extract values at specific dimensions
        rms_values = torch.stack(
            [rms_norm_hs[:, :, idx] for idx in self.dim_sink], 
            dim=-1
        )
        
        # Get maximum across dimensions
        max_rms_values = torch.max(rms_values, dim=-1)[0]

        if layer_idx == 0 and image_start is not None and image_len is not None:
            print(f"\n{'='*70}")
            print(f"🔍 [TRAINING] Layer {layer_idx} RMS Analysis:")
            print(f"   Sequence length: {max_rms_values.shape[1]}")
            print(f"   Image region: [{image_start}, {image_start+image_len})")
            print(f"   dim_sink indices: {self.dim_sink}")
            
            # 打印原始 hidden states
            print(f"\n   🔬 Raw Hidden States:")
            print(f"      - min: {hidden_states.min().item():.6f}")
            print(f"      - max: {hidden_states.max().item():.6f}")
            print(f"      - mean: {hidden_states.mean().item():.6f}")
            
            # 打印 RMS norm 后的值
            print(f"\n   🔬 RMS Normalized:")
            print(f"      - min: {rms_norm_hs.min().item():.6f}")
            print(f"      - max: {rms_norm_hs.max().item():.6f}")
        
        # Find indices where max > threshold
        sink_positions = torch.nonzero(max_rms_values > self.tau)
        
        if sink_positions.numel() == 0:
            # print(f"   ❌ No sink tokens found (all values < {self.tau})")
            # print(f"{'='*70}\n")
            return torch.tensor([], dtype=torch.long, device=hidden_states.device)
        
        # Extract token indices
        sink_indices = sink_positions[:, 1].unique()
        # print(f"   ✅ Found {len(sink_indices)} sink tokens")
        # print(f"   Sink token indices: {sink_indices.tolist()}")
        # print(f"{'='*70}\n")
        
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
            image_start=im,  # ⭐ 传入 image_start
            image_len=pa     # ⭐ 传入 image_len
        )
        
        if sink_indices.numel() == 0:
            # print(f"❌ STEP 1 FAILED: No sink tokens detected\n")
            return torch.tensor([], dtype=torch.long, device=attn.device).reshape(0, 3)
        
        # print(f"✅ STEP 1 Found {len(sink_indices)} sink tokens")
        
        # Step 2: Filter sink indices within image region
        vis_sink_inds = sink_indices[(im <= sink_indices) & (sink_indices < im + pa)]
        # print("vis_sink_inds:",vis_sink_inds)
        
        if len(vis_sink_inds) == 0:
            # print(f"   ❌ STEP 2 FAILED: No sink tokens in image region!")
            # print(f"   💡 Analysis:")
            # print(f"      - Total sink tokens: {len(sink_indices)}")
            # print(f"      - Image region: [{im}, {im+pa})")
            # print(f"      - All sink positions: {sink_indices.tolist()}")
            # print(f"      - Tokens before image (< {im}): {(sink_indices < im).sum().item()}")
            # print(f"      - Tokens after image (>= {im+pa}): {(sink_indices >= im+pa).sum().item()}")
            # print(f"\n{'🔵'*35}\n")
            return torch.tensor([], dtype=torch.long, device=attn.device).reshape(0, 3)
        
        # print(f"✅ STEP 2 Visual sink indices: {vis_sink_inds.tolist()}\n")
        
        # Step 3: Extract attention to image region
        image_attn = attn[:, :, :, im:im+pa]
        # print(f"✅ STEP 3 RESULT:")
        # print(f"   image_attn.shape: {image_attn.shape}")
        
        # Step 4: Calculate metrics
        local_sink_inds = vis_sink_inds - im
        sink_attn = torch.sum(image_attn[:, :, :, local_sink_inds], dim=-1)
        total_attn = torch.sum(image_attn, dim=-1) + 1e-6
        
        portion = sink_attn / total_attn
        summation = torch.sum(image_attn, dim=-1)

        # print(f"\n✅ STEP 4 RESULT:")
        # print(f"   sink_attn.shape: {sink_attn.shape}")

        
        # Step 5: Apply conditions
        portion_condition = portion <= self.config.rho
        summation_condition = summation >= self.config.summ
        
        candidate_coords = torch.nonzero(portion_condition & summation_condition)
        # if layer_idx == 0:
        #     print(f"\n📊 [Layer {layer_idx}] Detailed Statistics:")
        #     print(f"   Portion values:")
        #     print(f"     - min: {portion.min().item():.4f}")
        #     print(f"     - max: {portion.max().item():.4f}")
        #     print(f"     - mean: {portion.mean().item():.4f}")
        #     print(f"     - median: {portion.median().item():.4f}")
        #     print(f"     - 25th percentile: {torch.quantile(portion.float(), 0.25).item():.4f}")
        #     print(f"     - 75th percentile: {torch.quantile(portion.float(), 0.75).item():.4f}")
            
        #     print(f"\n   Summation values:")
        #     print(f"     - min: {summation.min().item():.4f}")
        #     print(f"     - max: {summation.max().item():.4f}")
        #     print(f"     - mean: {summation.mean().item():.4f}")
        #     print(f"     - median: {summation.median().item():.4f}")
        #     print(f"     - 25th percentile: {torch.quantile(summation.float(), 0.25).item():.4f}")
        #     print(f"     - 75th percentile: {torch.quantile(summation.float(), 0.75).item():.4f}")
            
        #     print(f"\n   Condition results:")
        #     print(f"     - portion <= {self.config.rho}: {portion_condition.sum().item()} / {portion_condition.numel()}")
        #     print(f"     - summation >= {self.config.summ}: {summation_condition.sum().item()} / {summation_condition.numel()}")
        #     print(f"     - Both conditions: {(portion_condition & summation_condition).sum().item()}")


        # print(f"\n✅ STEP 5 RESULT:")
        # print(f"   candidate_coords.shape: {candidate_coords.shape}")
        
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