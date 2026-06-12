DEFAULT_SKIP_MODULES_PATTERN = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")


def apply_layerwise_casting(*args, **kwargs):
    raise NotImplementedError("Layerwise casting is not included in the local Cosmos3 T2V runtime subset.")


def apply_layerwise_casting_hook(*args, **kwargs):
    raise NotImplementedError("Layerwise casting is not included in the local Cosmos3 T2V runtime subset.")
