from __future__ import annotations

from typing import Iterable

import torch


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("block size must be positive")
    return (a + b - 1) // b


def _coords_from_visual_index(
    flat: torch.Tensor, h_lat: int, w_lat: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plane = h_lat * w_lat
    t = flat // plane
    rem = flat % plane
    h = rem // w_lat
    w = rem % w_lat
    return t, h, w


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


def _visual_flat_ranges(
    flat: torch.Tensor,
    valid: torch.Tensor,
    q_len: int,
    h_lat: int,
    w_lat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = _coords_from_visual_index(flat, h_lat, w_lat)
    visual_flat = (t * h_lat + h) * w_lat + w
    big = q_len + 10
    mins = torch.where(valid, visual_flat, torch.full_like(visual_flat, big)).amin(dim=1)
    maxs = torch.where(valid, visual_flat, torch.full_like(visual_flat, -1)).amax(dim=1)
    return mins, maxs


def build_window_anchor_block_mask(
    t_lat,
    h_lat,
    w_lat,
    und_len,
    q_len,
    kv_len,
    w_local,
    keyframes,
    blkq=64,
    blkk=128,
    device="cpu",
) -> torch.BoolTensor:
    """Build a static full-spatial window+anchor temporal block mask.

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
    w_local = int(w_local)
    blkq = int(blkq)
    blkk = int(blkk)

    if min(t_lat, h_lat, w_lat) <= 0:
        raise ValueError("latent grid dimensions must be positive")
    if min(und_len, q_len, kv_len) < 0:
        raise ValueError("lengths must be non-negative")

    assert q_len == t_lat * h_lat * w_lat, "q_len must equal t_lat*h_lat*w_lat"
    assert kv_len - und_len == q_len, "KV must be [text prefix, visual tokens]"
    assert w_local >= 0, "w_local must be non-negative"
    assert blkq > 0 and blkk > 0, "block sizes must be positive"

    dev = torch.device(device)
    q_blocks = _ceil_div(q_len, blkq)
    k_blocks = _ceil_div(kv_len, blkk)

    frames = torch.arange(t_lat, device=dev, dtype=torch.int64)
    distance = (frames[None, :] - frames[:, None]).abs()
    keep_frames = distance <= w_local

    valid_keyframes = sorted(
        {
            int(frame)
            for frame in keyframes
            if 0 <= int(frame) < t_lat
        }
    )
    if valid_keyframes:
        anchor_frames = torch.tensor(valid_keyframes, device=dev, dtype=torch.int64)
        keep_frames[:, anchor_frames] = True

    if w_local >= t_lat - 1 or keep_frames.all().item():
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

    q_flat = q_pos.clamp(min=0, max=q_len - 1)
    k_flat = k_visual_pos.clamp(min=0, max=q_len - 1)
    q_min, q_max = _visual_flat_ranges(q_flat, q_valid, q_len, h_lat, w_lat)
    k_min, k_max = _visual_flat_ranges(k_flat, k_visual, q_len, h_lat, w_lat)
    diagonal = (q_min[:, None] <= k_max[None, :]) & (
        k_min[None, :] <= q_max[:, None]
    )
    mask = mask | (diagonal & visual_blocks[None, :])

    if text_blocks.any().item():
        mask[:, text_blocks] = True
    empty_k_blocks = ~(text_blocks | visual_blocks)
    if empty_k_blocks.any().item():
        mask[:, empty_k_blocks] = False
    return mask


REAL_GEOMETRY = {
    "t_lat": 48,
    "h_lat": 23,
    "w_lat": 40,
    "und_len": 43,
    "q_len": 44160,
    "kv_len": 44203,
    "blkq": 64,
    "blkk": 128,
}


def _skip_fraction(mask: torch.Tensor) -> float:
    return 1.0 - mask.float().mean().item()


def _build_real(w_local: int, keyframes: Iterable[int]) -> torch.Tensor:
    return build_window_anchor_block_mask(
        REAL_GEOMETRY["t_lat"],
        REAL_GEOMETRY["h_lat"],
        REAL_GEOMETRY["w_lat"],
        REAL_GEOMETRY["und_len"],
        REAL_GEOMETRY["q_len"],
        REAL_GEOMETRY["kv_len"],
        w_local,
        keyframes,
        blkq=REAL_GEOMETRY["blkq"],
        blkk=REAL_GEOMETRY["blkk"],
    )


def _test_dense_equivalence() -> None:
    mask = _build_real(47, set())
    assert mask.shape == (690, 346), f"bad dense shape: {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"bad dtype: {mask.dtype}"
    assert mask.all().item(), "full-window real mask must be all True"


def _test_text_blocks() -> None:
    mask = _build_real(10, {0, 8, 16, 24, 32, 40, 47})
    assert mask.shape == (690, 346), f"bad shape: {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"bad dtype: {mask.dtype}"
    assert mask[:, 0].all().item(), "k_block 0 contains text and must be all True"


def _run(name: str, fn):
    try:
        value = fn()
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        raise
    print(f"PASS {name}")
    return value


def main() -> None:
    _run("dense_equivalence", _test_dense_equivalence)
    _run("text_block_always_kept", _test_text_blocks)

    configs = [
        ("A1", 14, {0, 12, 24, 36, 47}),
        ("A2", 10, {0, 8, 16, 24, 32, 40, 47}),
    ]
    for name, w_local, keyframes in configs:
        mask = _build_real(w_local, keyframes)
        print(f"{name} skip_fraction: {_skip_fraction(mask):.6f}")


if __name__ == "__main__":
    main()
