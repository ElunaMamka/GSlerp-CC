import inspect
import math
import os
from bisect import bisect_left
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "convert_kvcache_qwen2_grslerp_v5",
    "Qwen2AttentionGrSlerpV5Wrapper",
    "compress_kv_states_grslerp_v5",
    "slerp_merge",
    "register_grslerp_v5_analysis_hook",
]

_GSLERP_V5_ANALYSIS_HOOK: Optional[Callable[..., None]] = None


def register_grslerp_v5_analysis_hook(hook: Optional[Callable[..., None]]) -> None:
    global _GSLERP_V5_ANALYSIS_HOOK
    _GSLERP_V5_ANALYSIS_HOOK = hook


def _set_submodule(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parts = module_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _safe_to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def slerp_merge(token_a: torch.Tensor, token_b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """Slerp merge with 2-dim RoPE grouping on the last dim."""

    if token_a.shape != token_b.shape:
        raise ValueError(f"Mismatched token shapes for slerp: {token_a.shape} vs {token_b.shape}")

    orig_dtype = token_a.dtype
    head_dim = token_a.shape[-1]

    # Fallback for degenerate cases.
    if head_dim < 2 or (head_dim % 2 != 0):
        t_tensor = torch.as_tensor(t, device=token_a.device, dtype=token_a.dtype)
        return ((1 - t_tensor) * token_a + t_tensor * token_b).to(dtype=orig_dtype)

    token_a = token_a.float()
    token_b = token_b.float()

    original_shape = token_a.shape
    num_groups = head_dim // 2

    a_group = token_a.reshape(*original_shape[:-1], num_groups, 2)
    b_group = token_b.reshape(*original_shape[:-1], num_groups, 2)

    norm_a = a_group.norm(dim=-1, keepdim=True).clamp(min=eps)
    norm_b = b_group.norm(dim=-1, keepdim=True).clamp(min=eps)
    unit_a = a_group / norm_a
    unit_b = b_group / norm_b

    dot = (unit_a * unit_b).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.acos(dot)
    sin_omega = omega.sin().clamp(min=eps)

    t_tensor = torch.as_tensor(t, device=token_a.device, dtype=token_a.dtype)
    t_tensor = t_tensor.reshape([1] * (omega.ndim - 1) + [1]).expand_as(omega)

    factor_a = torch.sin((1 - t_tensor) * omega) / sin_omega
    factor_b = torch.sin(t_tensor * omega) / sin_omega

    # Near-collinear vectors: fallback to LERP to avoid numerical explosion.
    lerp_merged = (1 - t_tensor) * a_group + t_tensor * b_group
    stable_mask = sin_omega > 1e-4
    slerp_merged = factor_a * a_group + factor_b * b_group
    merged = torch.where(stable_mask, slerp_merged, lerp_merged)

    # Keep vector norm interpolation stable.
    merged_norm = (1 - t_tensor) * norm_a + t_tensor * norm_b
    merged = merged * merged_norm

    merged = torch.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
    merged = merged.reshape(original_shape)
    merged = torch.clamp(merged, min=-100.0, max=100.0)
    return merged.to(dtype=orig_dtype)


def value_select_merge(
    token_a: torch.Tensor,
    token_b: torch.Tensor,
    anchor_score: float,
    drop_score: float,
) -> torch.Tensor:
    if drop_score > anchor_score:
        return token_b
    return token_a


def _select_mid_indices(length: int, compression_ratio: float, attn_score: torch.Tensor) -> Tuple[List[int], List[int]]:
    if length <= 1:
        return list(range(length)), []

    target_len = max(1, int(math.ceil(length * compression_ratio)))
    target_len = min(target_len, length)
    if target_len >= length:
        keep = list(range(length))
        return keep, []

    scores_for_topk = attn_score[:length].detach().float()
    topk = torch.topk(scores_for_topk, target_len, largest=True)
    keep = sorted(topk.indices.tolist())
    keep_set = set(keep)
    drop = [idx for idx in range(length) if idx not in keep_set]
    return keep, drop


def _map_s2_to_s1(keep_indices: Sequence[int], drop_indices: Sequence[int]) -> Dict[int, int]:
    if not keep_indices:
        return {}

    mapping: Dict[int, int] = {}
    for idx in drop_indices:
        pos = bisect_left(keep_indices, idx)
        if pos <= 0:
            neighbor = keep_indices[0]
        else:
            neighbor = keep_indices[pos - 1]
        mapping[idx] = neighbor
    return mapping


def _merge_s1s2(
    states: torch.Tensor,
    attn_score: torch.Tensor,
    keep_indices: Sequence[int],
    drop_indices: Sequence[int],
    mapping: Dict[int, int],
    mode: str,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not keep_indices:
        return states, attn_score

    keep_indices = sorted(set(keep_indices))
    new_states = states[:, :, keep_indices, :].clone()
    new_scores = attn_score[keep_indices].clone()

    pos_lookup = {idx: pos for pos, idx in enumerate(keep_indices)}

    for drop_idx in drop_indices:
        if drop_idx not in mapping:
            continue
        anchor_idx = mapping[drop_idx]
        if anchor_idx not in pos_lookup:
            continue

        anchor_pos = pos_lookup[anchor_idx]
        anchor_vec = new_states[:, :, anchor_pos, :]
        drop_vec = states[:, :, drop_idx, :]

        anchor_score = float(attn_score[anchor_idx].item())
        drop_score = float(attn_score[drop_idx].item())

        if mode == "slerp":
            denom = anchor_score + drop_score + eps
            t = anchor_score / denom if denom > eps else 0.5
            merged = slerp_merge(anchor_vec, drop_vec, float(t))
        elif mode == "select":
            merged = value_select_merge(anchor_vec, drop_vec, anchor_score, drop_score)
        else:
            raise ValueError(f"Unknown merge mode: {mode}")

        new_states[:, :, anchor_pos, :] = merged
        new_scores[anchor_pos] = new_scores[anchor_pos] + attn_score[drop_idx]

    new_states = torch.nan_to_num(new_states, nan=0.0, posinf=0.0, neginf=0.0)
    new_states = torch.clamp(new_states, min=-100.0, max=100.0)
    return new_states, new_scores


def _iterative_mid_merge(
    states: torch.Tensor,
    attn_score: torch.Tensor,
    compression_ratio: float,
    mode: str,
) -> torch.Tensor:
    cur_states = states
    cur_scores = _safe_to_float_tensor(attn_score).clone()

    while cur_states.shape[2] > 0:
        cur_len = cur_states.shape[2]
        target_len = max(1, int(math.ceil(cur_len * compression_ratio)))
        if cur_len <= target_len:
            break

        keep_indices, drop_indices = _select_mid_indices(cur_len, compression_ratio, cur_scores)
        if not drop_indices:
            break

        mapping = _map_s2_to_s1(keep_indices, drop_indices)
        cur_states, cur_scores = _merge_s1s2(
            cur_states,
            cur_scores,
            keep_indices,
            drop_indices,
            mapping,
            mode=mode,
        )

    return cur_states


def local_gslerp_mask_s1s2(
    states: torch.Tensor,
    attn_score: torch.Tensor,
    start_budget: int,
    recent_budget: int,
    compression_ratio: float,
    mode: str,
) -> torch.Tensor:
    if states.ndim != 4:
        return states

    _, _, seq_len, _ = states.shape
    if seq_len <= start_budget + recent_budget:
        return states

    prefix = states[:, :, :start_budget, :]
    suffix_begin = max(seq_len - recent_budget, start_budget)
    mid = states[:, :, start_budget:suffix_begin, :]
    suffix = states[:, :, suffix_begin:, :]

    if compression_ratio <= 0:
        return torch.cat([prefix, suffix], dim=2)

    if mid.shape[2] == 0:
        return states

    mid_scores = attn_score[start_budget:suffix_begin]
    compressed_mid = _iterative_mid_merge(mid, mid_scores, compression_ratio, mode=mode)
    out = torch.cat([prefix, compressed_mid, suffix], dim=2)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.clamp(out, min=-100.0, max=100.0)
    return out


def compress_kv_states_grslerp_v5(
    states: torch.Tensor,
    attn_score: torch.Tensor,
    *,
    start_keep: int,
    recent_ratio: float,
    compression_ratio: float,
    mode: str,
) -> torch.Tensor:
    """v5 compression: top-k S1 keep + S2->S1 merge on middle region."""

    if states.ndim != 4:
        return states

    _, _, seq_len, _ = states.shape
    if seq_len <= 2:
        return states

    start_budget = min(max(0, int(start_keep)), seq_len)
    recent_budget = int(float(recent_ratio) * seq_len)
    recent_budget = min(max(0, recent_budget), max(0, seq_len - start_budget))

    if seq_len <= start_budget + recent_budget:
        return states

    attn_score = _safe_to_float_tensor(attn_score)
    if attn_score.numel() != seq_len:
        attn_score = torch.ones(seq_len, device=states.device, dtype=torch.float32)

    return local_gslerp_mask_s1s2(
        states,
        attn_score,
        start_budget=start_budget,
        recent_budget=recent_budget,
        compression_ratio=float(compression_ratio),
        mode=mode,
    )


def _schedule_weights(num_layers: int, schedule_type: str) -> Optional[List[float]]:
    if num_layers <= 1:
        return None

    raw_weights: List[float] = []
    for idx in range(num_layers):
        if schedule_type == "linear":
            raw = 1.0 - 0.5 * (idx / float(num_layers - 1))
        elif schedule_type == "cosine":
            raw = 0.5 * (1.0 + math.cos(math.pi * idx / float(num_layers - 1)))
        elif schedule_type == "flat":
            raw = 1.0
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        raw_weights.append(raw)

    avg_w = sum(raw_weights) / len(raw_weights)
    return [w / avg_w for w in raw_weights]


class Qwen2AttentionGrSlerpV5Wrapper(nn.Module):
    """Qwen2 attention wrapper with GrSlerp-v5 KV cache compression."""

    def __init__(
        self,
        base_attn: nn.Module,
        *,
        layer_idx: int,
        compression_ratio: float,
        recent_ratio: float,
        start_keep: int,
        force_output_attentions: bool = True,
    ) -> None:
        super().__init__()
        self.base_attn = base_attn
        self.layer_idx = int(layer_idx)

        self.compression_ratio = float(compression_ratio)
        self.recent_ratio = float(recent_ratio)
        self.start_keep = int(start_keep)

        self.force_output_attentions = bool(force_output_attentions)

        self._forward_sig = inspect.signature(self.base_attn.forward)
        self._param_names = list(self._forward_sig.parameters.keys())
        self._output_attn_pos = self._param_names.index("output_attentions") if "output_attentions" in self._param_names else None
        self._past_kv_pos = self._param_names.index("past_key_value") if "past_key_value" in self._param_names else None

        self._debug_budget = 0
        debug_env = os.environ.get("GSLERP_DEBUG_BUDGET")
        if debug_env:
            try:
                self._debug_budget = max(0, int(debug_env))
            except ValueError:
                self._debug_budget = 10

    def _extract_past_kv_object(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if "past_key_value" in kwargs:
            return kwargs["past_key_value"]
        if self._past_kv_pos is not None and self._past_kv_pos < len(args):
            return args[self._past_kv_pos]
        return None

    def _extract_avg_attn(self, attn_weights: Optional[torch.Tensor], kv_len: int, device: torch.device) -> torch.Tensor:
        if attn_weights is None or not torch.is_tensor(attn_weights):
            return torch.ones(kv_len, device=device, dtype=torch.float32)

        # Common shape: [bsz, num_heads, q_len, kv_len]
        if attn_weights.ndim < 4:
            return torch.ones(kv_len, device=device, dtype=torch.float32)

        avg_attn = attn_weights.mean(dim=1)[:, -1, :].mean(dim=0)
        avg_attn = _safe_to_float_tensor(avg_attn)
        if avg_attn.numel() != kv_len:
            return torch.ones(kv_len, device=device, dtype=torch.float32)
        return avg_attn

    def _compress_pair(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is None or value_states is None:
            return key_states, value_states

        if key_states.ndim != 4 or value_states.ndim != 4:
            return key_states, value_states

        kv_len = key_states.shape[2]
        recent_budget = int(self.recent_ratio * kv_len)
        cache_budget = min(kv_len, self.start_keep + recent_budget)
        if kv_len <= cache_budget:
            return key_states, value_states

        avg_attn = self._extract_avg_attn(attn_weights, kv_len, key_states.device)

        hook = _GSLERP_V5_ANALYSIS_HOOK
        if hook is not None:
            hook(
                layer_idx=self.layer_idx,
                vector_type="key",
                states=key_states.detach(),
                attn_score=avg_attn.detach(),
                start_budget=min(self.start_keep, kv_len),
                recent_budget=min(recent_budget, max(0, kv_len - min(self.start_keep, kv_len))),
                compression_ratio=self.compression_ratio,
            )

        key_states_new = compress_kv_states_grslerp_v5(
            key_states,
            avg_attn,
            start_keep=self.start_keep,
            recent_ratio=self.recent_ratio,
            compression_ratio=self.compression_ratio,
            mode="slerp",
        )

        hook = _GSLERP_V5_ANALYSIS_HOOK
        if hook is not None:
            hook(
                layer_idx=self.layer_idx,
                vector_type="value",
                states=value_states.detach(),
                attn_score=avg_attn.detach(),
                start_budget=min(self.start_keep, kv_len),
                recent_budget=min(recent_budget, max(0, kv_len - min(self.start_keep, kv_len))),
                compression_ratio=self.compression_ratio,
            )

        value_states_new = compress_kv_states_grslerp_v5(
            value_states,
            avg_attn,
            start_keep=self.start_keep,
            recent_ratio=self.recent_ratio,
            compression_ratio=self.compression_ratio,
            mode="select",
        )

        key_states_new = torch.nan_to_num(key_states_new, nan=0.0, posinf=0.0, neginf=0.0)
        value_states_new = torch.nan_to_num(value_states_new, nan=0.0, posinf=0.0, neginf=0.0)
        key_states_new = torch.clamp(key_states_new, min=-100.0, max=100.0)
        value_states_new = torch.clamp(value_states_new, min=-100.0, max=100.0)

        if self._debug_budget > 0:
            print(
                f"[qwen2-grslerp-v5] layer={self.layer_idx} kv_len={kv_len} -> {key_states_new.shape[2]} "
                f"start_keep={self.start_keep} recent_ratio={self.recent_ratio:.3f} "
                f"comp={self.compression_ratio:.3f}",
                flush=True,
            )
            self._debug_budget -= 1

        return key_states_new, value_states_new

    def _maybe_mutate_cache_obj(self, cache_obj: Any, attn_weights: Optional[torch.Tensor]) -> None:
        if cache_obj is None:
            return

        # HF DynamicCache/StaticCache style.
        if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
            key_cache = cache_obj.key_cache
            value_cache = cache_obj.value_cache
            if self.layer_idx >= len(key_cache) or self.layer_idx >= len(value_cache):
                return

            key_states = key_cache[self.layer_idx]
            value_states = value_cache[self.layer_idx]
            if key_states is None or value_states is None:
                return

            key_new, value_new = self._compress_pair(key_states, value_states, attn_weights)
            cache_obj.key_cache[self.layer_idx] = key_new
            cache_obj.value_cache[self.layer_idx] = value_new

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        requested_output_attn = False
        call_args = args
        call_kwargs = dict(kwargs)

        if self.force_output_attentions and self._output_attn_pos is not None:
            if "output_attentions" in call_kwargs:
                requested_output_attn = bool(call_kwargs["output_attentions"])
                call_kwargs["output_attentions"] = True
            elif self._output_attn_pos < len(call_args):
                call_args_mut = list(call_args)
                requested_output_attn = bool(call_args_mut[self._output_attn_pos])
                call_args_mut[self._output_attn_pos] = True
                call_args = tuple(call_args_mut)
            else:
                call_kwargs["output_attentions"] = True

        outputs = self.base_attn(*call_args, **call_kwargs)

        if not isinstance(outputs, tuple):
            return outputs

        out_list = list(outputs)
        attn_weights = out_list[1] if len(out_list) > 1 else None

        # Legacy present tuple path.
        if len(out_list) > 2 and isinstance(out_list[2], tuple) and len(out_list[2]) >= 2:
            present = out_list[2]
            key_states = present[0]
            value_states = present[1]
            if torch.is_tensor(key_states) and torch.is_tensor(value_states):
                key_new, value_new = self._compress_pair(key_states, value_states, attn_weights)
                out_list[2] = (key_new, value_new, *present[2:])

        # Cache-object mutation path.
        cache_obj = self._extract_past_kv_object(call_args, call_kwargs)
        self._maybe_mutate_cache_obj(cache_obj, attn_weights)

        if self.force_output_attentions and not requested_output_attn and len(out_list) > 1:
            out_list[1] = None

        return tuple(out_list)


# Backward-friendly alias (same naming style as v3 file).
Qwen2AttentionGrSlerpWrapper = Qwen2AttentionGrSlerpV5Wrapper


def convert_kvcache_qwen2_grslerp_v5(
    model: nn.Module,
    config: Any,
    *,
    schedule_type: str = "linear",
    force_output_attentions: bool = True,
) -> nn.Module:
    """Wrap all Qwen2 attention modules with GrSlerp-v5 KV compression wrappers."""

    base_ratio = float(getattr(config, "compression_ratio", 0.375))
    recent_ratio = float(getattr(config, "recent_ratio", 0.1))
    start_keep = int(getattr(config, "start_keep", 1))

    modules = dict(model.named_modules())
    attn_names: List[str] = []
    for name, module in modules.items():
        cls_name = module.__class__.__name__.lower()
        if "qwen2" in cls_name and "attention" in cls_name and not isinstance(module, Qwen2AttentionGrSlerpV5Wrapper):
            attn_names.append(name)

    if not attn_names:
        raise ValueError(
            "No Qwen2 attention modules were found. "
            "Please load a Qwen2.5 model with attn_implementation='eager' before conversion."
        )

    layer_weights = _schedule_weights(len(attn_names), schedule_type)

    for idx, module_name in enumerate(attn_names):
        old_module = modules[module_name]
        ratio_i = base_ratio
        if layer_weights is not None:
            ratio_i = base_ratio * layer_weights[idx]

        wrapper = Qwen2AttentionGrSlerpV5Wrapper(
            old_module,
            layer_idx=getattr(old_module, "layer_idx", idx),
            compression_ratio=ratio_i,
            recent_ratio=recent_ratio,
            start_keep=start_keep,
            force_output_attentions=force_output_attentions,
        )
        _set_submodule(model, module_name, wrapper)

    return model
