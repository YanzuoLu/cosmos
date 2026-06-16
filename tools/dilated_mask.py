from __future__ import annotations

import torch


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("block size must be positive")
    return (a + b - 1) // b


def _frame_keep_matrix(
    t_lat: int,
    core: int,
    stride: int,
    rng: int,
    device: torch.device,
) -> torch.Tensor:
    frames = torch.arange(t_lat, device=device, dtype=torch.int64)
    distance = (frames[None, :] - frames[:, None]).abs()
    core_keep = distance <= core
    dilated_keep = (distance <= rng) & (distance.remainder(stride) == 0)
    return core_keep | dilated_keep


def _frame_presence(
    block_pos: torch.Tensor,
    valid: torch.Tensor,
    q_len: int,
    h_lat: int,
    w_lat: int,
    t_lat: int,
) -> torch.Tensor:
    block_count = int(block_pos.shape[0])
    plane = h_lat * w_lat
    flat = block_pos.clamp(min=0, max=q_len - 1)
    frame = (flat // plane).clamp(min=0, max=t_lat - 1)
    block_ids = torch.arange(
        block_count,
        device=block_pos.device,
        dtype=torch.int64,
    )[:, None].expand_as(block_pos)

    present = torch.zeros(
        (block_count, t_lat),
        dtype=torch.bool,
        device=block_pos.device,
    )
    present[block_ids[valid], frame[valid]] = True
    return present


def build_dilated_temporal_block_mask(
    t_lat,
    h_lat,
    w_lat,
    und_len,
    q_len,
    kv_len,
    core,
    stride,
    rng,
    blkq=64,
    blkk=128,
    device="cpu",
) -> torch.BoolTensor:
    """Build a static full-spatial dilated-temporal block mask.

    The returned mask has shape [Q_blocks, K_blocks]. True means the block is
    kept. Query blocks cover visual tokens only. KV blocks cover text prefix
    tokens followed by visual tokens; any block containing text is always kept.

    Per-block geometry uses exact sets of temporal frames covered by each block.
    The requested pattern is token-level on time and unrestricted spatially, so
    keeping a block when any query-frame/key-frame pair matches is a safe block
    over-approximation: it can keep extra token pairs inside selected blocks but
    will not drop a requested temporal pair.
    """

    t_lat = int(t_lat)
    h_lat = int(h_lat)
    w_lat = int(w_lat)
    und_len = int(und_len)
    q_len = int(q_len)
    kv_len = int(kv_len)
    core = int(core)
    stride = int(stride)
    rng = int(rng)
    blkq = int(blkq)
    blkk = int(blkk)

    if min(t_lat, h_lat, w_lat) <= 0:
        raise ValueError("latent grid dimensions must be positive")
    if min(und_len, q_len, kv_len, core, rng) < 0:
        raise ValueError("lengths and temporal ranges must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if rng < core:
        raise ValueError("rng must be greater than or equal to core")
    if blkq <= 0 or blkk <= 0:
        raise ValueError("block sizes must be positive")

    assert q_len == t_lat * h_lat * w_lat, "q_len must equal t_lat*h_lat*w_lat"
    assert kv_len >= und_len, "kv_len must include the und/text prefix"
    assert (
        kv_len - und_len == q_len
    ), "this dilated builder expects KV to be [text prefix, visual tokens]"

    dev = torch.device(device)
    q_blocks = _ceil_div(q_len, blkq)
    k_blocks = _ceil_div(kv_len, blkk)

    if core >= t_lat - 1:
        return torch.ones((q_blocks, k_blocks), dtype=torch.bool, device=dev)

    keep_frames = _frame_keep_matrix(t_lat, core, stride, rng, dev)
    if keep_frames.all().item():
        return torch.ones((q_blocks, k_blocks), dtype=torch.bool, device=dev)

    q_pos = torch.arange(q_blocks * blkq, device=dev, dtype=torch.int64).reshape(
        q_blocks, blkq
    )
    q_valid = q_pos < q_len
    q_present = _frame_presence(q_pos, q_valid, q_len, h_lat, w_lat, t_lat)

    k_pos = torch.arange(k_blocks * blkk, device=dev, dtype=torch.int64).reshape(
        k_blocks, blkk
    )
    k_valid = k_pos < kv_len
    k_text = k_valid & (k_pos < und_len)
    k_visual = k_valid & (k_pos >= und_len)
    text_blocks = k_text.any(dim=1)
    visual_blocks = k_visual.any(dim=1)

    k_visual_pos = k_pos - und_len
    k_present = _frame_presence(k_visual_pos, k_visual, q_len, h_lat, w_lat, t_lat)

    q_keep = q_present.to(torch.float32).matmul(keep_frames.to(torch.float32))
    mask = q_keep.matmul(k_present.to(torch.float32).t()) > 0

    if text_blocks.any().item():
        mask[:, text_blocks] = True
    empty_k_blocks = ~(text_blocks | visual_blocks)
    if empty_k_blocks.any().item():
        mask[:, empty_k_blocks] = False
    return mask
