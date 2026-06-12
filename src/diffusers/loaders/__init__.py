class FromOriginalModelMixin:
    pass


class PeftAdapterMixin:
    _hf_peft_config_loaded = False


__all__ = ["FromOriginalModelMixin", "PeftAdapterMixin"]
