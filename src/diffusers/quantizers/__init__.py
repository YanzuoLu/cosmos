class DiffusersQuantizer:
    pass


class DiffusersAutoQuantizer:
    @staticmethod
    def merge_quantization_configs(config, override):
        return override if override is not None else config

    @staticmethod
    def from_config(*args, **kwargs):
        raise NotImplementedError("Quantized model loading is not included in the local Cosmos3 T2V runtime subset.")


class PipelineQuantizationConfig:
    def _resolve_quant_config(self, *args, **kwargs):
        return None


__all__ = ["DiffusersAutoQuantizer", "DiffusersQuantizer", "PipelineQuantizationConfig"]
