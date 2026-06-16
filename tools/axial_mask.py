from __future__ import annotations

import torch


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("block size must be positive")
    return (a + b - 1) // b


def _coords_from_visual_index(flat, h_lat, w_lat):
    plane = h_lat * w_lat
    t = flat // plane
    rem = flat % plane
    h = rem // w_lat
    w = rem % w_lat
    return t, h, w


def build_axial_block_mask(
    t_lat,
    h_lat,
    w_lat,
    und_len,
    q_len,
    kv_len,
    blkq=64,
    blkk=128,
    device="cpu",
) -> torch.BoolTensor:
    """Static 3D axial (decomposed) attention block mask, shape [Q_blocks, K_blocks].

    Each query token (t,h,w) attends to the union of its 3 axis-lines: temporal
    (same h,w; all t), vertical (same t,w; all h), horizontal (same t,h; all w).
    A visual K-token (tk,hk,wk) is kept by query token (tq,hq,wq) iff
      (hk==hq and wk==wq)  # temporal line
      or (tk==tq and wk==wq)  # vertical line
      or (tk==tq and hk==hq)  # horizontal line
    A Q-block keeps a K-block iff ANY token pair satisfies the rule. Text-prefix
    K-blocks (kv_pos<und_len) and the diagonal K-block are always kept. This is
    EXACT at token level (no over-approximation), so it cannot under-keep.
    """
    t_lat = int(t_lat); h_lat = int(h_lat); w_lat = int(w_lat)
    und_len = int(und_len); q_len = int(q_len); kv_len = int(kv_len)
    blkq = int(blkq); blkk = int(blkk)

    if min(t_lat, h_lat, w_lat) <= 0:
        raise ValueError("latent grid dimensions must be positive")
    if min(und_len, q_len, kv_len) < 0:
        raise ValueError("lengths must be non-negative")
    if blkq <= 0 or blkk <= 0:
        raise ValueError("block sizes must be positive")
    assert q_len == t_lat * h_lat * w_lat, "q_len must equal t_lat*h_lat*w_lat"
    assert kv_len >= und_len, "kv_len must include the und/text prefix"
    assert kv_len - und_len == q_len, "KV must be [text prefix, visual tokens]"

    dev = torch.device(device)
    q_blocks = _ceil_div(q_len, blkq)
    k_blocks = _ceil_div(kv_len, blkk)

    # K-block bookkeeping over the full kv (text prefix + visual).
    k_pos = torch.arange(k_blocks * blkk, device=dev, dtype=torch.int64).reshape(k_blocks, blkk)
    k_valid = k_pos < kv_len
    k_text = k_valid & (k_pos < und_len)
    k_visual = k_valid & (k_pos >= und_len)
    text_blocks = k_text.any(dim=1)
    visual_blocks = k_visual.any(dim=1)

    # visual flat index for each kv slot (only meaningful where k_visual)
    k_flat = (k_pos - und_len).clamp(min=0, max=max(q_len - 1, 0))
    k_t, k_h, k_w = _coords_from_visual_index(k_flat, h_lat, w_lat)
    # axis keys for every kv slot (flattened to [k_blocks*blkk])
    k_hw = (k_h * w_lat + k_w).reshape(-1)        # temporal-line key (fixed h,w)
    k_tw = (k_t * w_lat + k_w).reshape(-1)        # vertical-line key (fixed t,w)
    k_th = (k_t * h_lat + k_h).reshape(-1)        # horizontal-line key (fixed t,h)
    k_visual_flatmask = k_visual.reshape(-1)
    kk = k_blocks * blkk

    # visual range of each K-block (for diagonal force-keep)
    big = q_len + 10
    k_visual_min = torch.where(k_visual, k_flat, torch.full_like(k_flat, big)).amin(dim=1)
    k_visual_max = torch.where(k_visual, k_flat, torch.full_like(k_flat, -1)).amax(dim=1)

    n_hw = h_lat * w_lat
    n_tw = t_lat * w_lat
    n_th = t_lat * h_lat

    mask = torch.zeros((q_blocks, k_blocks), dtype=torch.bool, device=dev)

    for qb in range(q_blocks):
        q_start = qb * blkq
        q_end = min(q_start + blkq, q_len)
        q_flat = torch.arange(q_start, q_end, device=dev, dtype=torch.int64)
        q_t, q_h, q_w = _coords_from_visual_index(q_flat, h_lat, w_lat)

        hw_keep = torch.zeros(n_hw, dtype=torch.bool, device=dev)
        tw_keep = torch.zeros(n_tw, dtype=torch.bool, device=dev)
        th_keep = torch.zeros(n_th, dtype=torch.bool, device=dev)
        hw_keep[(q_h * w_lat + q_w)] = True
        tw_keep[(q_t * w_lat + q_w)] = True
        th_keep[(q_t * h_lat + q_h)] = True

        keep_slot = (hw_keep[k_hw] | tw_keep[k_tw] | th_keep[k_th]) & k_visual_flatmask
        keep_slot = keep_slot.reshape(k_blocks, blkk)
        row = keep_slot.any(dim=1)

        # force-keep text-prefix blocks
        row = row | text_blocks
        # force-keep diagonal (K-block visual range overlaps Q-block visual range)
        diag = (k_visual_min < q_end) & (k_visual_max >= q_start) & visual_blocks
        row = row | diag
        # clear fully-empty blocks (defensive; no-op for real geometry)
        empty = ~(text_blocks | visual_blocks)
        row = row & ~empty

        mask[qb] = row

    return mask
