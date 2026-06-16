from __future__ import annotations

import time
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


def _axis_ranges(
    flat: torch.Tensor,
    valid: torch.Tensor,
    t_lat: int,
    h_lat: int,
    w_lat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, h, w = _coords_from_visual_index(flat, h_lat, w_lat)

    def reduce_axis(axis: torch.Tensor, empty_min: int) -> tuple[torch.Tensor, torch.Tensor]:
        mins = torch.where(valid, axis, torch.full_like(axis, empty_min)).amin(dim=1)
        maxs = torch.where(valid, axis, torch.full_like(axis, -1)).amax(dim=1)
        return mins, maxs

    t_min, t_max = reduce_axis(t, t_lat)
    h_min, h_max = reduce_axis(h, h_lat)
    w_min, w_max = reduce_axis(w, w_lat)
    return t_min, t_max, h_min, h_max, w_min, w_max


def _intervals_within_window(
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    window: int,
) -> torch.Tensor:
    return (q_min[:, None] - window <= k_max[None, :]) & (
        k_min[None, :] - window <= q_max[:, None]
    )


def build_sta_block_mask(
    t_lat,
    h_lat,
    w_lat,
    und_len,
    q_len,
    kv_len,
    wt,
    wh,
    ww,
    blkq=64,
    blkk=128,
    device="cpu",
) -> torch.BoolTensor:
    """Build a static 3D sliding-tile-attention block mask.

    The returned mask has shape [Q_blocks, K_blocks]. True means the block is
    kept. Query blocks cover visual tokens only. KV blocks cover text prefix
    tokens followed by visual tokens; any block containing text is always kept.

    Per-block geometry uses min/max boxes over each block's token coordinates.
    Contiguous flattened blocks do not necessarily form exact boxes in (t,h,w),
    so this intentionally over-approximates the token-level window. That is safe
    for block sparse attention: it can keep extra blocks but will not drop a
    token pair that is inside the requested window.
    """

    t_lat = int(t_lat)
    h_lat = int(h_lat)
    w_lat = int(w_lat)
    und_len = int(und_len)
    q_len = int(q_len)
    kv_len = int(kv_len)
    wt = int(wt)
    wh = int(wh)
    ww = int(ww)
    blkq = int(blkq)
    blkk = int(blkk)

    if min(t_lat, h_lat, w_lat) <= 0:
        raise ValueError("latent grid dimensions must be positive")
    if min(und_len, q_len, kv_len, wt, wh, ww) < 0:
        raise ValueError("lengths and window half-widths must be non-negative")
    if blkq <= 0 or blkk <= 0:
        raise ValueError("block sizes must be positive")

    assert q_len == t_lat * h_lat * w_lat, "q_len must equal t_lat*h_lat*w_lat"
    assert kv_len >= und_len, "kv_len must include the und/text prefix"
    assert (
        kv_len - und_len == q_len
    ), "this STA builder expects KV to be [text prefix, visual tokens]"

    dev = torch.device(device)
    q_blocks = _ceil_div(q_len, blkq)
    k_blocks = _ceil_div(kv_len, blkk)

    if wt >= t_lat - 1 and wh >= h_lat - 1 and ww >= w_lat - 1:
        return torch.ones((q_blocks, k_blocks), dtype=torch.bool, device=dev)

    q_pos = torch.arange(q_blocks * blkq, device=dev, dtype=torch.int64).reshape(
        q_blocks, blkq
    )
    q_valid = q_pos < q_len
    q_flat = q_pos.clamp(max=q_len - 1)
    q_t_min, q_t_max, q_h_min, q_h_max, q_w_min, q_w_max = _axis_ranges(
        q_flat, q_valid, t_lat, h_lat, w_lat
    )

    k_pos = torch.arange(k_blocks * blkk, device=dev, dtype=torch.int64).reshape(
        k_blocks, blkk
    )
    k_valid = k_pos < kv_len
    k_text = k_valid & (k_pos < und_len)
    k_visual = k_valid & (k_pos >= und_len)
    text_blocks = k_text.any(dim=1)
    visual_blocks = k_visual.any(dim=1)

    k_flat = (k_pos - und_len).clamp(min=0, max=q_len - 1)
    k_t_min, k_t_max, k_h_min, k_h_max, k_w_min, k_w_max = _axis_ranges(
        k_flat, k_visual, t_lat, h_lat, w_lat
    )

    t_ok = _intervals_within_window(q_t_min, q_t_max, k_t_min, k_t_max, wt)
    h_ok = _intervals_within_window(q_h_min, q_h_max, k_h_min, k_h_max, wh)
    w_ok = _intervals_within_window(q_w_min, q_w_max, k_w_min, k_w_max, ww)
    mask = t_ok & h_ok & w_ok

    if text_blocks.any():
        mask[:, text_blocks] = True
    empty_k_blocks = ~(text_blocks | visual_blocks)
    if empty_k_blocks.any():
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


def _build_real(wt: int, wh: int, ww: int) -> torch.Tensor:
    return build_sta_block_mask(
        REAL_GEOMETRY["t_lat"],
        REAL_GEOMETRY["h_lat"],
        REAL_GEOMETRY["w_lat"],
        REAL_GEOMETRY["und_len"],
        REAL_GEOMETRY["q_len"],
        REAL_GEOMETRY["kv_len"],
        wt,
        wh,
        ww,
        REAL_GEOMETRY["blkq"],
        REAL_GEOMETRY["blkk"],
    )


def _test_tiny_grid() -> None:
    mask = build_sta_block_mask(
        t_lat=2,
        h_lat=2,
        w_lat=2,
        und_len=2,
        q_len=8,
        kv_len=10,
        wt=0,
        wh=0,
        ww=0,
        blkq=2,
        blkk=2,
    )
    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, False, False, True, False],
            [True, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask.cpu(), expected), f"unexpected tiny mask:\n{mask}"
    assert mask[:, 0].all().item(), "text KV block must be kept for every Q block"

    dense = build_sta_block_mask(
        t_lat=2,
        h_lat=2,
        w_lat=2,
        und_len=2,
        q_len=8,
        kv_len=10,
        wt=2,
        wh=2,
        ww=2,
        blkq=2,
        blkk=2,
    )
    assert dense.all().item(), "large tiny-grid window must be dense"


def _test_dense_equivalence() -> None:
    mask = _build_real(48, 23, 40)
    assert mask.shape == (690, 346), f"bad dense shape: {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"bad dtype: {mask.dtype}"
    assert mask.all().item(), "full-window real mask must be all True"


def _test_shape_and_text() -> float:
    mask = _build_real(6, 6, 6)
    assert mask.shape == (690, 346), f"bad moderate shape: {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"bad dtype: {mask.dtype}"
    assert mask[:, 0].all().item(), "k_block 0 contains text and must be all True"
    return _skip_fraction(mask)


def _test_monotonicity() -> tuple[int, int]:
    small = _build_real(3, 3, 3)
    large = _build_real(8, 8, 8)
    small_kept = int(small.sum().item())
    large_kept = int(large.sum().item())
    assert small_kept <= large_kept, (
        f"larger window must keep a superset: {small_kept} > {large_kept}"
    )
    return small_kept, large_kept


def _measure_skip_fractions(
    windows: Iterable[tuple[int, int, int]]
) -> list[tuple[tuple[int, int, int], float, float]]:
    results = []
    for wt, wh, ww in windows:
        started = time.perf_counter()
        mask = _build_real(wt, wh, ww)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results.append(((wt, wh, ww), _skip_fraction(mask), elapsed_ms))
    return results


def _run(name: str, fn):
    try:
        value = fn()
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        raise
    print(f"PASS {name}")
    return value


def main() -> None:
    _run("tiny_grid", _test_tiny_grid)
    _run("dense_equivalence", _test_dense_equivalence)
    moderate_skip = _run("shape_text_and_moderate_window", _test_shape_and_text)
    print(f"moderate_window_skip_fraction (6,6,6): {moderate_skip:.6f}")
    small_kept, large_kept = _run("monotonicity", _test_monotonicity)
    print(f"monotonicity_kept_counts (3,3,3)<=(8,8,8): {small_kept}<={large_kept}")

    windows = [(2, 2, 2), (3, 3, 3), (6, 4, 4), (6, 6, 6), (12, 12, 12)]
    for window, skip, elapsed_ms in _measure_skip_fractions(windows):
        print(f"skip_fraction {window}: {skip:.6f} build_ms={elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
