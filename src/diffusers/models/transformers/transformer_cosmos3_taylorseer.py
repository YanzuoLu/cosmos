# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


class Cosmos3TaylorSeerAttnProcessor:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate causal (understanding) and full
    (generation) attention pathways. The generation pathway cross-attends to both und and gen keys/values.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Cosmos3TaylorSeerPackedMoTAttention",
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-pathway projections
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # Apply rotary position embeddings per pathway
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        # Causal pathway (understanding): und tokens self-attend with causal masking.
        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)

        # Full pathway (generation): gen tokens cross-attend to all (und + gen) keys/values.
        all_k = torch.cat([k_und, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        full_out = full_out.squeeze(0).flatten(-2, -1)

        # Per-pathway output projection
        und_out = attn.to_out(causal_out)
        gen_out = attn.to_add_out(full_out)
        return und_out, gen_out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3TaylorSeerVLTextRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: tuple[int, int, int]):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_axes_dim = rope_axes_dim

    def apply_interleaved_mrope(self, freqs, rope_axes_dim):
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved
        [THTHWHTHW...TT], preserving frequency continuity across the 3 grids."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids, device, dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3,B,1,N]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.rope_axes_dim)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)  # each: [B,N,head_dim]


class Cosmos3TaylorSeerVLTextMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3TaylorSeerDomainAwareLinear(nn.Module):
    """Linear projection with one weight/bias pair per embodiment domain."""

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size)
        self.bias = nn.Embedding(self.num_domains, self.output_size)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                "Cosmos3 action domain_id batch size must match action tokens: "
                f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
            )
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(f"Cosmos3 action domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}.")
        weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError(f"Cosmos3TaylorSeerDomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")


class Cosmos3TaylorSeerPackedMoTAttention(nn.Module, AttentionModuleMixin):
    """Dual-pathway packed attention for Qwen3VL MoT — separate projections for
    understanding (causal) and generation (full) token streams."""

    _default_processor_cls = Cosmos3TaylorSeerAttnProcessor
    _available_processors = [Cosmos3TaylorSeerAttnProcessor]

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
        processor=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Understanding pathway. norm_q / norm_k are applied per-head (only on
        # head_dim), so no reshape is needed after them.
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        # Generation pathway
        self.add_q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.add_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.add_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_add_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.norm_added_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_added_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, und_seq, gen_seq, rotary_emb)

    def build_und_kv_cache(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_und = self.to_k(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_und = self.to_v(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        k_und = self.norm_k(k_und)
        cos_und, sin_und, _, _ = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        return k_und.detach(), v_und.detach()

    def forward_gen_with_und_kv_cache(
        self,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        und_k: torch.Tensor,
        und_v: torch.Tensor,
    ) -> torch.Tensor:
        q_gen = self.add_q_proj(gen_seq).view(-1, self.num_attention_heads, self.head_dim)
        k_gen = self.add_k_proj(gen_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_gen = self.add_v_proj(gen_seq).view(-1, self.num_key_value_heads, self.head_dim)
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)
        _, _, cos_gen, sin_gen = rotary_emb
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen
        und_k = und_k.to(dtype=k_gen.dtype, device=k_gen.device)
        und_v = und_v.to(dtype=v_gen.dtype, device=v_gen.device)
        all_k = torch.cat([und_k, k_gen], dim=0)
        all_v = torch.cat([und_v, v_gen], dim=0)
        full_out = dispatch_attention_fn(
            q_gen.unsqueeze(0),
            all_k.unsqueeze(0),
            all_v.unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
            backend=self.processor._attention_backend,
            parallel_config=self.processor._parallel_config,
        )
        full_out = full_out.squeeze(0).flatten(-2, -1)
        return self.to_add_out(full_out)

    def forward_und(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        q_und = self.to_q(und_seq).view(-1, self.num_attention_heads, self.head_dim)
        k_und = self.to_k(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_und = self.to_v(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        cos_und, sin_und, _, _ = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            backend=self.processor._attention_backend,
            parallel_config=self.processor._parallel_config,
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)
        return self.to_out(causal_out)


class Cosmos3TaylorSeerVLTextMoTDecoderLayer(nn.Module):
    """
    Qwen3VL text MoT (Mixture of Tokens) decoder layer. Features dual-pathway attention for understanding vs
    generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = Cosmos3TaylorSeerPackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = Cosmos3TaylorSeerVLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.mlp_moe_gen = Cosmos3TaylorSeerVLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.post_attention_layernorm_moe_gen = RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False
        )

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen

    def forward_with_gen_delta(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        gen_next = residual_gen + mlp_out_gen
        gen_delta = gen_attn_out + mlp_out_gen
        return residual_und + mlp_out_und, gen_next, gen_delta, gen_attn_out, mlp_out_gen

    def forward_with_predicted_gen_mlp_delta(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gen_mlp_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        return residual_und + mlp_out_und, residual_gen + gen_mlp_delta.to(
            dtype=gen_seq.dtype, device=gen_seq.device
        )

    def build_und_kv_cache(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_und_kv_cache(self.input_layernorm(und_seq), rotary_emb)

    def forward_gen_with_und_kv_cache(
        self,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        und_k: torch.Tensor,
        und_v: torch.Tensor,
    ) -> torch.Tensor:
        gen_norm = self.input_layernorm_moe_gen(gen_seq)
        gen_attn_out = self.self_attn.forward_gen_with_und_kv_cache(gen_norm, rotary_emb, und_k, und_v)
        residual_gen = gen_seq + gen_attn_out
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return residual_gen + mlp_out_gen

    def forward_und_only(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        und_norm = self.input_layernorm(und_seq)
        und_attn_out = self.self_attn.forward_und(und_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        return residual_und + mlp_out_und



Cosmos3TaylorSeerCalType = Literal["full", "Taylor", "ToCa", "Delta-Cache"]


@dataclass
class Cosmos3TaylorSeerConfig:
    enabled: bool = False
    interval: int = 5
    fresh_threshold: int | None = None
    force_scheduler: bool = False
    max_order: int = 1
    first_enhance: int = 1
    last_enhance: int = 1
    force_final_full: bool = True
    layer_indices: tuple[int, ...] | None = None
    cache_und: bool = True
    cache_max_bytes: int | None = None
    branches: Literal["both", "cond", "uncond"] = "both"
    delta_change_threshold: float | None = None
    prediction_target: Literal["layer_delta", "attention_delta", "mlp_delta", "gen_component_delta", "und_cache"] = "layer_delta"
    stagger_layers: bool = False
    slope_scale: float = 1.0

@dataclass
class Cosmos3TaylorSeerLayerState:
    und_after: torch.Tensor | None = None
    factors: list[torch.Tensor] = field(default_factory=list)
    gen_attn_factors: list[torch.Tensor] = field(default_factory=list)
    gen_mlp_factors: list[torch.Tensor] = field(default_factory=list)
    last_full_step: int | None = None
    cal_threshold: int | None = None
    full_count: int = 0
    predicted_count: int = 0
    prediction_allowed: bool = True
    last_delta_change_ratio: float | None = None
    und_k: torch.Tensor | None = None
    und_v: torch.Tensor | None = None


@dataclass
class Cosmos3TaylorSeerBranchState:
    signature: tuple[Any, ...] | None = None
    layers: dict[int, Cosmos3TaylorSeerLayerState] = field(default_factory=dict)
    full_steps: set[int] = field(default_factory=set)
    predicted_steps: set[int] = field(default_factory=set)


@dataclass
class Cosmos3TaylorSeerRunContext:
    branch: Literal["cond", "uncond"]
    step_index: int
    timestep: int
    num_steps: int


@dataclass
class Cosmos3TaylorSeerStats:
    enabled: bool
    interval: int
    fresh_threshold: int | None
    force_scheduler: bool
    max_order: int
    first_enhance: int
    force_final_full: bool
    last_enhance: int
    branches: str
    delta_change_threshold: float | None
    prediction_target: str
    cache_und: bool
    stagger_layers: bool
    slope_scale: float
    selected_layers: list[int]
    cache_max_gib: float | None
    estimated_cache_gib: float | None
    skipped_by_memory: bool
    full_steps_by_branch: dict[str, int]
    predicted_steps_by_branch: dict[str, int]
    full_layer_calls_by_branch: dict[str, int]
    predicted_layer_calls_by_branch: dict[str, int]


class Cosmos3OmniTaylorSeerTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Cosmos3TaylorSeerVLTextMoTDecoderLayer"]
    _repeated_blocks = ["Cosmos3TaylorSeerVLTextMoTDecoderLayer"]
    _skip_layerwise_casting_patterns = ["embed_tokens", "time_embedder", "norm"]
    _keep_in_fp32_modules = ["time_embedder"]
    # `dtype` is injected into init_dict by ModelMixin.from_pretrained (configuration_utils.py:289),
    # so __init__ must accept it. Excluding it here keeps save_pretrained from writing it into
    # config.json — the value is a load-time runtime hint, not part of the model architecture.
    ignore_for_config = ["dtype"]

    @register_to_config
    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        dtype: str = "bfloat16",  # required by the loader (see `ignore_for_config` above); not read here
        head_dim: int = 128,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        latent_channel: int = 48,
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 15000,
        latent_patch_size: int = 2,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 36,
        num_key_value_heads: int = 8,
        patch_latent_dim: int = 192,
        rms_norm_eps: float = 1e-6,
        rope_scaling: dict | None = None,
        rope_theta: float = 5000000.0,
        action_dim: int | None = None,
        action_gen: bool = False,
        num_embodiment_domains: int = 32,
        sound_dim: int | None = None,
        sound_gen: bool = False,
        sound_latent_fps: float = 25.0,
        timestep_scale: float = 0.001,
        vocab_size: int = 151936,
    ):
        super().__init__()

        rope_axes_dim = rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling is not None else [24, 20, 20]
        self.register_to_config(rope_axes_dim=rope_axes_dim)

        # Text-model layers live directly on the transformer (flat layout). The published
        # checkpoint must be re-keyed with the leading `model.` prefix stripped — see
        # scripts/build_flat_layout_repo.py for the rewrite.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Cosmos3TaylorSeerVLTextMoTDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.norm_moe_gen = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        self.rotary_emb = Cosmos3TaylorSeerVLTextRotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes_dim
        )

        # Modality projection heads + timestep embedding.
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.proj_in = nn.Linear(patch_latent_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, patch_latent_dim, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.action_gen = action_gen
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        if action_gen:
            if self.action_dim is None:
                raise ValueError("`action_dim` must be provided when `action_gen=True`.")
            self.action_proj_in = Cosmos3TaylorSeerDomainAwareLinear(self.action_dim, hidden_size, self.num_embodiment_domains)
            self.action_proj_out = Cosmos3TaylorSeerDomainAwareLinear(hidden_size, self.action_dim, self.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(torch.zeros(hidden_size))
        if sound_gen:
            if sound_dim is None:
                raise ValueError("`sound_dim` must be provided when `sound_gen=True`.")
            self.audio_proj_in = nn.Linear(sound_dim, hidden_size, bias=True)
            self.audio_proj_out = nn.Linear(hidden_size, sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(torch.zeros(hidden_size))

        self.gradient_checkpointing = False
        self._taylorseer_config = Cosmos3TaylorSeerConfig()
        self._taylorseer_branch_states = {
            "cond": Cosmos3TaylorSeerBranchState(),
            "uncond": Cosmos3TaylorSeerBranchState(),
        }
        self._taylorseer_context: Cosmos3TaylorSeerRunContext | None = None
        self._taylorseer_active_run = False
        self._taylorseer_num_steps = 0
        self._taylorseer_branch_count = 1
        self._taylorseer_memory_checked = False
        self._taylorseer_cache_estimate_bytes: int | None = None
        self._taylorseer_skipped_by_memory = False
        self._taylorseer_last_stats: Cosmos3TaylorSeerStats | None = None


    # -------------------------------------------------------------------------
    # TaylorSeer acceleration state.
    # -------------------------------------------------------------------------

    def enable_taylorseer(
        self,
        *,
        interval: int = 5,
        fresh_threshold: int | None = None,
        force_scheduler: bool = False,
        max_order: int = 1,
        first_enhance: int = 1,
        last_enhance: int = 1,
        force_final_full: bool = True,
        layer_indices: Iterable[int] | None = None,
        cache_und: bool = True,
        cache_max_gib: float | None = 64.0,
        branches: Literal["both", "cond", "uncond"] = "both",
        delta_change_threshold: float | None = None,
        prediction_target: Literal["layer_delta", "attention_delta", "mlp_delta", "gen_component_delta", "und_cache"] = "layer_delta",
        stagger_layers: bool = False,
        slope_scale: float = 1.0,
    ) -> None:
        if interval < 1:
            raise ValueError("TaylorSeer interval must be >= 1.")
        if fresh_threshold is not None and fresh_threshold < 1:
            raise ValueError("TaylorSeer fresh_threshold must be >= 1 or None.")
        if max_order not in {0, 1}:
            raise ValueError("TaylorSeer max_order must be 0 or 1.")
        if first_enhance < 0:
            raise ValueError("TaylorSeer first_enhance must be >= 0.")
        if last_enhance < 0:
            raise ValueError("TaylorSeer last_enhance must be >= 0.")
        if not cache_und and prediction_target == "und_cache":
            raise ValueError("TaylorSeer prediction_target='und_cache' requires cache_und=True.")
        if cache_max_gib is not None and cache_max_gib <= 0:
            raise ValueError("TaylorSeer cache_max_gib must be positive or None.")
        if branches not in {"both", "cond", "uncond"}:
            raise ValueError("TaylorSeer branches must be 'both', 'cond', or 'uncond'.")
        if delta_change_threshold is not None and delta_change_threshold <= 0:
            raise ValueError("TaylorSeer delta_change_threshold must be positive or None.")
        if prediction_target not in {"layer_delta", "attention_delta", "mlp_delta", "gen_component_delta", "und_cache"}:
            raise ValueError(
                "TaylorSeer prediction_target must be 'layer_delta', 'attention_delta', 'mlp_delta', "
                "'gen_component_delta', or 'und_cache'."
            )
        if not math.isfinite(slope_scale) or slope_scale < 0:
            raise ValueError("TaylorSeer slope_scale must be finite and >= 0.")

        normalized_layer_indices: tuple[int, ...] | None = None
        if layer_indices is not None:
            normalized_layer_indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
            for index in normalized_layer_indices:
                if index < 0 or index >= len(self.layers):
                    raise ValueError(
                        f"TaylorSeer layer index {index} is outside valid range [0, {len(self.layers)})."
                    )

        cache_max_bytes = None if cache_max_gib is None else int(cache_max_gib * (1024**3))
        self._taylorseer_config = Cosmos3TaylorSeerConfig(
            enabled=True,
            interval=int(interval),
            fresh_threshold=None if fresh_threshold is None else int(fresh_threshold),
            force_scheduler=bool(force_scheduler),
            max_order=int(max_order),
            first_enhance=int(first_enhance),
            last_enhance=int(last_enhance),
            force_final_full=bool(force_final_full),
            layer_indices=normalized_layer_indices,
            cache_und=bool(cache_und),
            cache_max_bytes=cache_max_bytes,
            branches=branches,
            delta_change_threshold=None if delta_change_threshold is None else float(delta_change_threshold),
            prediction_target=prediction_target,
            stagger_layers=bool(stagger_layers),
            slope_scale=float(slope_scale),
        )
        self.reset_taylorseer_cache(clear_stats=True)

    def is_taylorseer_enabled(self) -> bool:
        return bool(self._taylorseer_config.enabled)

    def disable_taylorseer(self) -> None:
        self._taylorseer_config.enabled = False
        self.reset_taylorseer_cache(clear_stats=True)

    def reset_taylorseer_cache(self, *, clear_stats: bool = True) -> None:
        self._taylorseer_branch_states = {
            "cond": Cosmos3TaylorSeerBranchState(),
            "uncond": Cosmos3TaylorSeerBranchState(),
        }
        self._taylorseer_context = None
        self._taylorseer_active_run = False
        self._taylorseer_num_steps = 0
        self._taylorseer_branch_count = 1
        self._taylorseer_memory_checked = False
        self._taylorseer_cache_estimate_bytes = None
        self._taylorseer_skipped_by_memory = False
        if clear_stats:
            self._taylorseer_last_stats = None

    def begin_taylorseer_run(self, *, num_steps: int, do_classifier_free_guidance: bool) -> None:
        self.reset_taylorseer_cache(clear_stats=False)
        self._taylorseer_num_steps = int(num_steps)
        accelerates_both_cfg_branches = do_classifier_free_guidance and self._taylorseer_config.branches == "both"
        self._taylorseer_branch_count = 2 if accelerates_both_cfg_branches else 1
        self._taylorseer_active_run = bool(self._taylorseer_config.enabled)

    def set_taylorseer_context(
        self, *, branch: Literal["cond", "uncond"], step_index: int, timestep: int
    ) -> None:
        if branch not in {"cond", "uncond"}:
            raise ValueError("TaylorSeer branch must be 'cond' or 'uncond'.")
        if isinstance(timestep, torch.Tensor):
            timestep = int(timestep.detach().reshape(-1)[0].item()) if timestep.numel() else 0
        self._taylorseer_context = Cosmos3TaylorSeerRunContext(
            branch=branch,
            step_index=int(step_index),
            timestep=int(timestep),
            num_steps=self._taylorseer_num_steps,
        )

    def _taylorseer_branch_enabled(self, branch: Literal["cond", "uncond"]) -> bool:
        return self._taylorseer_config.branches == "both" or self._taylorseer_config.branches == branch

    def clear_taylorseer_context(self) -> None:
        self._taylorseer_context = None

    def finish_taylorseer_run(self) -> None:
        if self._taylorseer_active_run or self._taylorseer_memory_checked:
            self._taylorseer_last_stats = self._taylorseer_build_stats()
        self._taylorseer_context = None
        self._taylorseer_active_run = False
        self._taylorseer_branch_states = {
            "cond": Cosmos3TaylorSeerBranchState(),
            "uncond": Cosmos3TaylorSeerBranchState(),
        }

    def get_taylorseer_stats(self) -> dict[str, Any]:
        if self._taylorseer_last_stats is not None and not self._taylorseer_active_run:
            return self._taylorseer_stats_to_dict(self._taylorseer_last_stats)
        return self._taylorseer_stats_to_dict(self._taylorseer_build_stats())

    def _taylorseer_selected_layers(self) -> tuple[int, ...]:
        if self._taylorseer_config.layer_indices is not None:
            return self._taylorseer_config.layer_indices
        return tuple(range(len(self.layers)))

    def _taylorseer_estimate_cache_bytes(
        self,
        *,
        selected_layers: tuple[int, ...],
        branches: int,
        gen_seq: torch.Tensor,
        und_seq: torch.Tensor | None,
    ) -> int:
        layer_count = len(selected_layers)
        factor_bytes_per_element = gen_seq.element_size()
        if self._taylorseer_config.prediction_target == "und_cache":
            gen_bytes = 0
        else:
            factor_tensor_count = 1 + self._taylorseer_config.max_order
            if self._taylorseer_config.prediction_target == "gen_component_delta":
                factor_tensor_count *= 2
            gen_bytes = branches * layer_count * factor_tensor_count * gen_seq.numel() * factor_bytes_per_element
        und_bytes = 0
        if self._taylorseer_config.cache_und and und_seq is not None:
            if self._taylorseer_config.prediction_target == "und_cache":
                und_cache_elements = und_seq.numel()
                if self.layers:
                    first_attn = self.layers[0].self_attn
                    und_cache_elements += 2 * und_seq.shape[0] * first_attn.num_key_value_heads * first_attn.head_dim
                und_bytes = branches * layer_count * und_cache_elements * und_seq.element_size()
            elif self._taylorseer_config.prediction_target != "mlp_delta":
                und_bytes = branches * layer_count * und_seq.numel() * und_seq.element_size()
        return int(gen_bytes + und_bytes)

    def _taylorseer_force_scheduler_threshold(self, step_index: int, num_steps: int) -> int:
        threshold = self._taylorseer_config.fresh_threshold
        if threshold is None:
            threshold = self._taylorseer_config.interval
        if not self._taylorseer_config.force_scheduler:
            return int(threshold)
        linear_step_weight = 0.0
        step_factor = 1.0
        if num_steps > 0:
            step_factor = 1 - linear_step_weight + 2 * linear_step_weight * int(step_index) / int(num_steps)
        return max(1, int(round(float(threshold) / step_factor)))

    def _taylorseer_cal_type(
        self,
        entry: Cosmos3TaylorSeerLayerState,
        step_index: int,
        num_steps: int,
        *,
        layer_index: int | None = None,
    ) -> Cosmos3TaylorSeerCalType:
        if entry.last_full_step is None:
            return "full"
        if step_index < self._taylorseer_config.first_enhance:
            return "full"
        if self._taylorseer_config.force_final_full:
            final_full_steps = max(1, self._taylorseer_config.last_enhance)
            if step_index >= max(0, num_steps - final_full_steps):
                return "full"
        if self._taylorseer_config.delta_change_threshold is not None and not entry.prediction_allowed:
            return "full"
        if self._taylorseer_config.prediction_target == "und_cache" and entry.und_k is not None and entry.und_v is not None:
            return "Taylor"

        cadence_threshold = self._taylorseer_config.interval
        if self._taylorseer_config.fresh_threshold is not None:
            cadence_threshold = entry.cal_threshold or self._taylorseer_force_scheduler_threshold(
                entry.last_full_step, num_steps
            )

        if self._taylorseer_config.stagger_layers and layer_index is not None and cadence_threshold > 1:
            if step_index % cadence_threshold == layer_index % cadence_threshold:
                return "full"
            return "Taylor"
        if step_index - entry.last_full_step >= cadence_threshold:
            return "full"
        return "Taylor"

    def _taylorseer_should_full(
        self,
        entry: Cosmos3TaylorSeerLayerState,
        step_index: int,
        num_steps: int,
        *,
        layer_index: int | None = None,
    ) -> bool:
        return self._taylorseer_cal_type(entry, step_index, num_steps, layer_index=layer_index) == "full"

    def _taylorseer_update_prediction_guard(
        self, entry: Cosmos3TaylorSeerLayerState, sample: torch.Tensor, old_sample: torch.Tensor | None
    ) -> None:
        threshold = self._taylorseer_config.delta_change_threshold
        if threshold is None:
            entry.prediction_allowed = True
            entry.last_delta_change_ratio = None
            return
        if old_sample is None:
            entry.prediction_allowed = False
            entry.last_delta_change_ratio = None
            return
        sample_float = sample.float()
        old_sample_float = old_sample.float()
        diff_rms = (sample_float - old_sample_float).pow(2).mean().sqrt()
        sample_rms = sample_float.pow(2).mean().sqrt().clamp_min(torch.finfo(sample_float.dtype).tiny)
        change_ratio = diff_rms / sample_rms
        entry.last_delta_change_ratio = float(change_ratio.detach().item())
        entry.prediction_allowed = entry.last_delta_change_ratio <= threshold

    def _taylorseer_updated_factor_list(
        self, factors: list[torch.Tensor], sample: torch.Tensor, step_index: int, last_full_step: int | None
    ) -> list[torch.Tensor]:
        old_factor0 = factors[0] if factors else None
        if old_factor0 is None or self._taylorseer_config.max_order == 0:
            return [sample]
        if last_full_step is None:
            raise RuntimeError("TaylorSeer full samples must have a previous full step before slope prediction.")
        dt = step_index - last_full_step
        if dt <= 0:
            raise RuntimeError("TaylorSeer full samples must advance denoising steps.")
        return [sample, ((sample - old_factor0) / dt).detach()]

    def _taylorseer_finish_factor_update(
        self, entry: Cosmos3TaylorSeerLayerState, step_index: int, num_steps: int | None
    ) -> None:
        entry.last_full_step = int(step_index)
        entry.cal_threshold = self._taylorseer_force_scheduler_threshold(
            step_index, self._taylorseer_num_steps if num_steps is None else num_steps
        )

    def _taylorseer_update_factors(
        self, entry: Cosmos3TaylorSeerLayerState, sample: torch.Tensor, step_index: int, num_steps: int | None = None
    ) -> None:
        detached_sample = sample.detach()
        old_factor0 = entry.factors[0] if entry.factors else None
        self._taylorseer_update_prediction_guard(entry, detached_sample, old_factor0)
        entry.factors = self._taylorseer_updated_factor_list(
            entry.factors, detached_sample, step_index, entry.last_full_step
        )
        self._taylorseer_finish_factor_update(entry, step_index, num_steps)

    def _taylorseer_update_gen_component_factors(
        self,
        entry: Cosmos3TaylorSeerLayerState,
        gen_attn_delta: torch.Tensor,
        gen_mlp_delta: torch.Tensor,
        step_index: int,
        num_steps: int | None = None,
    ) -> None:
        detached_attn_delta = gen_attn_delta.detach()
        detached_mlp_delta = gen_mlp_delta.detach()
        old_attn_delta = entry.gen_attn_factors[0] if entry.gen_attn_factors else None
        old_mlp_delta = entry.gen_mlp_factors[0] if entry.gen_mlp_factors else None
        old_component_delta = (old_attn_delta + old_mlp_delta) if old_attn_delta is not None and old_mlp_delta is not None else None
        self._taylorseer_update_prediction_guard(
            entry, detached_attn_delta + detached_mlp_delta, old_component_delta
        )
        last_full_step = entry.last_full_step
        entry.gen_attn_factors = self._taylorseer_updated_factor_list(
            entry.gen_attn_factors, detached_attn_delta, step_index, last_full_step
        )
        entry.gen_mlp_factors = self._taylorseer_updated_factor_list(
            entry.gen_mlp_factors, detached_mlp_delta, step_index, last_full_step
        )
        entry.factors = []
        self._taylorseer_finish_factor_update(entry, step_index, num_steps)

    def _taylorseer_predict_from_factors(
        self, factors: list[torch.Tensor], last_full_step: int | None, step_index: int
    ) -> torch.Tensor:
        if not factors:
            raise RuntimeError("TaylorSeer prediction requested before a full layer sample exists.")
        age = 0 if last_full_step is None else step_index - last_full_step
        if self._taylorseer_config.max_order == 1 and len(factors) > 1:
            return factors[0] + factors[1] * age * self._taylorseer_config.slope_scale
        return factors[0]

    def _taylorseer_predict(self, entry: Cosmos3TaylorSeerLayerState, step_index: int) -> torch.Tensor:
        return self._taylorseer_predict_from_factors(entry.factors, entry.last_full_step, step_index)

    def _taylorseer_build_stats(self) -> Cosmos3TaylorSeerStats:
        cache_max_bytes = self._taylorseer_config.cache_max_bytes
        estimate_bytes = self._taylorseer_cache_estimate_bytes
        full_steps_by_branch: dict[str, int] = {}
        predicted_steps_by_branch: dict[str, int] = {}
        full_layer_calls_by_branch: dict[str, int] = {}
        predicted_layer_calls_by_branch: dict[str, int] = {}
        for branch, state in self._taylorseer_branch_states.items():
            full_steps_by_branch[branch] = len(state.full_steps)
            predicted_steps_by_branch[branch] = len(state.predicted_steps)
            full_layer_calls_by_branch[branch] = sum(entry.full_count for entry in state.layers.values())
            predicted_layer_calls_by_branch[branch] = sum(entry.predicted_count for entry in state.layers.values())

        return Cosmos3TaylorSeerStats(
            enabled=bool(self._taylorseer_config.enabled),
            interval=self._taylorseer_config.interval,
            fresh_threshold=self._taylorseer_config.fresh_threshold,
            force_scheduler=self._taylorseer_config.force_scheduler,
            max_order=self._taylorseer_config.max_order,
            first_enhance=self._taylorseer_config.first_enhance,
            last_enhance=self._taylorseer_config.last_enhance,
            force_final_full=self._taylorseer_config.force_final_full,
            branches=self._taylorseer_config.branches,
            delta_change_threshold=self._taylorseer_config.delta_change_threshold,
            prediction_target=self._taylorseer_config.prediction_target,
            cache_und=self._taylorseer_config.cache_und,
            stagger_layers=self._taylorseer_config.stagger_layers,
            slope_scale=self._taylorseer_config.slope_scale,
            selected_layers=list(self._taylorseer_selected_layers()) if self._taylorseer_config.enabled else [],
            cache_max_gib=None if cache_max_bytes is None else cache_max_bytes / (1024**3),
            estimated_cache_gib=None if estimate_bytes is None else estimate_bytes / (1024**3),
            skipped_by_memory=self._taylorseer_skipped_by_memory,
            full_steps_by_branch=full_steps_by_branch,
            predicted_steps_by_branch=predicted_steps_by_branch,
            full_layer_calls_by_branch=full_layer_calls_by_branch,
            predicted_layer_calls_by_branch=predicted_layer_calls_by_branch,
        )

    def _taylorseer_stats_to_dict(self, stats: Cosmos3TaylorSeerStats) -> dict[str, Any]:
        return {
            "enabled": stats.enabled,
            "interval": stats.interval,
            "fresh_threshold": stats.fresh_threshold,
            "force_scheduler": stats.force_scheduler,
            "max_order": stats.max_order,
            "first_enhance": stats.first_enhance,
            "last_enhance": stats.last_enhance,
            "force_final_full": stats.force_final_full,
            "branches": stats.branches,
            "delta_change_threshold": stats.delta_change_threshold,
            "prediction_target": stats.prediction_target,
            "cache_und": stats.cache_und,
            "stagger_layers": stats.stagger_layers,
            "slope_scale": stats.slope_scale,
            "selected_layers": list(stats.selected_layers),
            "cache_max_gib": stats.cache_max_gib,
            "estimated_cache_gib": stats.estimated_cache_gib,
            "skipped_by_memory": stats.skipped_by_memory,
            "full_steps_by_branch": dict(stats.full_steps_by_branch),
            "predicted_steps_by_branch": dict(stats.predicted_steps_by_branch),
            "full_layer_calls_by_branch": dict(stats.full_layer_calls_by_branch),
            "predicted_layer_calls_by_branch": dict(stats.predicted_layer_calls_by_branch),
        }

    def _taylorseer_branch_signature(
        self, *, sequence_length: int, und_len: int, gen_seq: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[Any, ...]:
        first_position = -1
        if position_ids.numel():
            first_position = int(position_ids[0, 0].item()) if position_ids.ndim >= 2 else int(position_ids[0].item())
        return (
            sequence_length,
            und_len,
            gen_seq.shape[0],
            gen_seq.shape[1],
            gen_seq.dtype,
            gen_seq.device.type,
            gen_seq.device.index,
            position_ids.shape,
            first_position,
        )

    def _taylorseer_forward_layers(
        self,
        *,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        sequence_length: int,
        und_len: int,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._taylorseer_context
        if context is None:
            raise RuntimeError("TaylorSeer context is required during an active TaylorSeer run")

        if not self._taylorseer_branch_enabled(context.branch):
            for decoder_layer in self.layers:
                und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
            return und_seq, gen_seq

        selected_layers = self._taylorseer_selected_layers()
        selected_layer_set = set(selected_layers)
        branch_state = self._taylorseer_branch_states[context.branch]
        signature = self._taylorseer_branch_signature(
            sequence_length=sequence_length, und_len=und_len, gen_seq=gen_seq, position_ids=position_ids
        )
        if branch_state.signature is None:
            branch_state.signature = signature
        elif branch_state.signature != signature:
            raise RuntimeError("TaylorSeer branch signature changed within an active run")

        if selected_layers and not self._taylorseer_memory_checked:
            estimate = self._taylorseer_estimate_cache_bytes(
                selected_layers=selected_layers,
                branches=self._taylorseer_branch_count,
                gen_seq=gen_seq,
                und_seq=und_seq,
            )
            self._taylorseer_cache_estimate_bytes = estimate
            self._taylorseer_memory_checked = True
            cache_max_bytes = self._taylorseer_config.cache_max_bytes
            if cache_max_bytes is not None and estimate > cache_max_bytes:
                self._taylorseer_skipped_by_memory = True
                self._taylorseer_last_stats = self._taylorseer_build_stats()
                raise RuntimeError(
                    "TaylorSeer cache estimate exceeds limit: "
                    f"estimate={estimate} bytes, limit={cache_max_bytes} bytes"
                )

        for layer_index, decoder_layer in enumerate(self.layers):
            if layer_index not in selected_layer_set:
                und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
                continue

            entry = branch_state.layers.get(layer_index)
            if entry is None:
                entry = Cosmos3TaylorSeerLayerState()
                branch_state.layers[layer_index] = entry

            if self._taylorseer_should_full(
                entry,
                context.step_index,
                context.num_steps,
                layer_index=layer_index,
            ):
                if self._taylorseer_config.prediction_target == "und_cache":
                    entry.und_k, entry.und_v = decoder_layer.build_und_kv_cache(und_seq, rotary_emb)
                    und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
                    entry.und_after = und_seq.detach() if self._taylorseer_config.cache_und else None
                    entry.last_full_step = int(context.step_index)
                    entry.cal_threshold = self._taylorseer_force_scheduler_threshold(
                        context.step_index, context.num_steps
                    )
                else:
                    und_seq, gen_seq, gen_delta, gen_attn_delta, gen_mlp_delta = decoder_layer.forward_with_gen_delta(
                        und_seq, gen_seq, rotary_emb
                    )
                    entry.und_after = (
                        und_seq.detach()
                        if self._taylorseer_config.cache_und
                        and self._taylorseer_config.prediction_target != "mlp_delta"
                        else None
                    )
                    if self._taylorseer_config.prediction_target == "gen_component_delta":
                        self._taylorseer_update_gen_component_factors(
                            entry, gen_attn_delta, gen_mlp_delta, context.step_index, context.num_steps
                        )
                    else:
                        if self._taylorseer_config.prediction_target == "attention_delta":
                            factor_sample = gen_attn_delta
                        elif self._taylorseer_config.prediction_target == "mlp_delta":
                            factor_sample = gen_mlp_delta
                        else:
                            factor_sample = gen_delta
                        self._taylorseer_update_factors(entry, factor_sample, context.step_index, context.num_steps)
                entry.full_count += 1
                branch_state.full_steps.add(context.step_index)
            else:
                if self._taylorseer_config.prediction_target == "und_cache":
                    if entry.und_after is None or entry.und_k is None or entry.und_v is None:
                        raise RuntimeError("TaylorSeer UND cache prediction requested before cache was populated.")
                    gen_seq = decoder_layer.forward_gen_with_und_kv_cache(gen_seq, rotary_emb, entry.und_k, entry.und_v)
                    und_seq = entry.und_after.to(dtype=und_seq.dtype, device=und_seq.device)
                else:
                    if self._taylorseer_config.prediction_target == "mlp_delta":
                        predicted_delta = self._taylorseer_predict(entry, context.step_index)
                        predicted_delta = predicted_delta.to(dtype=gen_seq.dtype, device=gen_seq.device)
                        und_seq, gen_seq = decoder_layer.forward_with_predicted_gen_mlp_delta(
                            und_seq, gen_seq, rotary_emb, predicted_delta
                        )
                    else:
                        if self._taylorseer_config.cache_und and entry.und_after is not None:
                            und_seq = entry.und_after.to(dtype=und_seq.dtype, device=und_seq.device)
                        elif not self._taylorseer_config.cache_und:
                            und_seq = decoder_layer.forward_und_only(und_seq, rotary_emb)
                        if self._taylorseer_config.prediction_target == "gen_component_delta":
                            predicted_attn_delta = self._taylorseer_predict_from_factors(
                                entry.gen_attn_factors, entry.last_full_step, context.step_index
                            ).to(dtype=gen_seq.dtype, device=gen_seq.device)
                            predicted_mlp_delta = self._taylorseer_predict_from_factors(
                                entry.gen_mlp_factors, entry.last_full_step, context.step_index
                            ).to(dtype=gen_seq.dtype, device=gen_seq.device)
                            gen_seq = gen_seq + predicted_attn_delta + predicted_mlp_delta
                        else:
                            predicted_delta = self._taylorseer_predict(entry, context.step_index)
                            predicted_delta = predicted_delta.to(dtype=gen_seq.dtype, device=gen_seq.device)
                            if self._taylorseer_config.prediction_target == "attention_delta":
                                residual_gen = gen_seq + predicted_delta
                                mlp_out_gen = decoder_layer.mlp_moe_gen(
                                    decoder_layer.post_attention_layernorm_moe_gen(residual_gen)
                                )
                                gen_seq = residual_gen + mlp_out_gen
                            else:
                                gen_seq = gen_seq + predicted_delta
                entry.predicted_count += 1
                branch_state.predicted_steps.add(context.step_index)

        return und_seq, gen_seq

    # -------------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (no layer state).
    # -------------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes: list[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            # Broadcast [N, 1] + [spatial_numel_i] → [N, spatial_numel_i]
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self,
        tokens_vision: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: list[torch.Tensor] = []
        original_latent_shapes: list[tuple[int, int, int]] = []
        for latent in tokens_vision:
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: list[tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: list[torch.Tensor] = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def _pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """List of ``[C, T]`` tensors → packed ``[total_T, C]`` tensor."""
        return torch.cat(
            [sound[:, : shape[0]].permute(1, 0) for sound, shape in zip(tokens_sound, token_shapes_sound)],
            dim=0,
        )

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, C]`` predictions → list of ``[C, T]`` tensors (zeros at conditioned positions)."""
        sound_dim = self.config.sound_dim
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros((sound_dim, T), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    def _pack_action_latents(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, int, int]],
        domain_ids_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List of ``[T, D]`` tensors → packed ``[total_T, D]`` plus per-token domain ids."""
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for action, shape, domain_id in zip(tokens_action, token_shapes_action, domain_ids_action):
            token_count = shape[0]
            packed.append(action[:token_count])
            domain_ids.append(domain_id.reshape(1).expand(token_count))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def _unpack_action_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, int, int]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, D]`` predictions → list of ``[T, D]`` tensors."""
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros((T, self.action_dim), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[noisy_idxs] = packed_preds[start_idx : start_idx + t_n]
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # -------------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision/sound/action → run layers →
    # decode vision/sound/action. Pipeline calls this once per CFG pass.
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_tokens: list[torch.Tensor] | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_ids: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Run a full denoising-step forward pass.

        Args:
            input_ids: Text token IDs placed at ``text_indexes`` in the joint sequence.
            text_indexes: Indices of text tokens in the joint sequence.
            position_ids: ``[3, sequence_length]`` mRoPE position IDs for the full joint sequence.
            und_len: Length of the causal text (understanding) prefix; generation tokens follow.
            sequence_length: Total length of the joint packed sequence.
            vision_tokens: Per-item vision latent tensors before patchify.
            vision_token_shapes: Patch grid shapes ``(T, H, W)`` per vision item.
            vision_sequence_indexes: Indices of vision tokens in the joint sequence.
            vision_mse_loss_indexes: Indices used to read vision predictions after the backbone.
            vision_timesteps: Per-patch diffusion timesteps for vision tokens.
            vision_noisy_frame_indexes: Noisy frame indices per vision item.
            sound_tokens: Optional sound latent tensors before packing.
            sound_token_shapes: Optional patch grid shapes for sound items.
            sound_sequence_indexes: Optional indices of sound tokens in the joint sequence.
            sound_mse_loss_indexes: Optional indices used to read sound predictions.
            sound_timesteps: Optional per-token diffusion timesteps for sound.
            sound_noisy_frame_indexes: Optional noisy frame indices per sound item.
            action_tokens: Optional action latent tensors before packing.
            action_token_shapes: Optional patch grid shapes ``(T, H, W)`` per action item.
            action_sequence_indexes: Optional indices of action tokens in the joint sequence.
            action_mse_loss_indexes: Optional indices used to read action predictions after the backbone.
            action_timesteps: Optional per-token diffusion timesteps for action tokens.
            action_noisy_frame_indexes: Optional noisy frame indices per action item.
            action_domain_ids: Optional per-item domain IDs selecting the action head weights.

        Returns:
            ``(preds_vision, preds_sound, preds_action)`` — lists of per-modality predictions. Optional modalities
            return ``None`` when their inputs are omitted.
        """
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

        # Embed text tokens into the joint hidden_states buffer at their sequence positions.
        packed_text_embedding = self.embed_tokens(input_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        hidden_states[text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then add timestep embeddings to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        timesteps_vision = vision_timesteps * self.config.timestep_scale
        packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        hidden_states[vision_sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (when present); all sound frames are noisy.
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound_tokens, sound_token_shapes).to(target_dtype)
            packed_tokens_sound = self.audio_proj_in(packed_tokens_sound) + self.audio_modality_embed
            timesteps_sound = sound_timesteps * self.config.timestep_scale
            packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound_noisy_frame_indexes,
                token_shapes=sound_token_shapes,
            )
            hidden_states[sound_sequence_indexes] = packed_tokens_sound

        # Pack + project action latents (when present). Domain ids select the action head weights.
        if has_action:
            packed_tokens_action, per_token_domain_ids = self._pack_action_latents(
                action_tokens, action_token_shapes, action_domain_ids
            )
            packed_tokens_action = packed_tokens_action.to(target_dtype)
            per_token_domain_ids = per_token_domain_ids.to(device=packed_tokens_action.device)
            packed_tokens_action = self.action_proj_in(packed_tokens_action, per_token_domain_ids)
            packed_tokens_action = packed_tokens_action + self.action_modality_embed
            if action_mse_loss_indexes.numel() > 0:
                timesteps_action = action_timesteps * self.config.timestep_scale
                packed_timestep_embeds_action = self.time_embedder(self.time_proj(timesteps_action))
                packed_timestep_embeds_action = packed_timestep_embeds_action.to(target_dtype)
                packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_tokens_action,
                    packed_timestep_embeds=packed_timestep_embeds_action,
                    noisy_frame_indexes=action_noisy_frame_indexes,
                    token_shapes=action_token_shapes,
                )
            hidden_states[action_sequence_indexes] = packed_tokens_action

        # Compute rotary embeddings once for the joint sequence, then slice into und/gen halves.
        _meta_tensor = torch.tensor([], dtype=hidden_states.dtype, device=hidden_states.device)
        cos, sin = self.rotary_emb(
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # cos, sin: [1, N, head_dim] (1-D pos_ids) or [3, 1, N, head_dim] (mrope pos_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        if not self._taylorseer_config.enabled or not self._taylorseer_active_run or torch.is_grad_enabled():
            for decoder_layer in self.layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    und_seq, gen_seq = self._gradient_checkpointing_func(
                        decoder_layer.__call__, und_seq, gen_seq, rotary_emb
                    )
                else:
                    und_seq, gen_seq = decoder_layer(und_seq, gen_seq, rotary_emb)
        else:
            und_seq, gen_seq = self._taylorseer_forward_layers(
                und_seq=und_seq,
                gen_seq=gen_seq,
                rotary_emb=rotary_emb,
                sequence_length=sequence_length,
                und_len=und_len,
                position_ids=position_ids,
            )
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)
        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision predictions from the joint hidden state.
        preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        preds_action: list[torch.Tensor] | None = None
        if has_action:
            per_noisy_domain_ids = [
                domain_id.reshape(1).expand(len(noisy_idxs))
                for domain_id, noisy_idxs in zip(action_domain_ids, action_noisy_frame_indexes)
            ]
            per_noisy_domain_ids = torch.cat(per_noisy_domain_ids, dim=0).to(device=last_hidden_state.device)
            preds_action_packed = self.action_proj_out(
                last_hidden_state[action_mse_loss_indexes], per_noisy_domain_ids
            )
            preds_action = self._unpack_action_latents(
                preds_action_packed, action_token_shapes, action_noisy_frame_indexes
            )

        return preds_vision, preds_sound, preds_action
