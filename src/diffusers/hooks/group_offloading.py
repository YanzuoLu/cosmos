import torch


def apply_group_offloading(*args, **kwargs):
    raise NotImplementedError("Group offloading is not included in the local Cosmos3 T2V runtime subset.")


def _is_group_offload_enabled(module: torch.nn.Module) -> bool:
    return False


def _get_group_onload_device(module: torch.nn.Module) -> torch.device:
    raise ValueError("Group offloading is not enabled.")


def _maybe_remove_and_reapply_group_offloading(module: torch.nn.Module) -> None:
    return None
