from enum import Enum


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODEL_OPT = "modelopt"
    AUTO_ROUND = "auto-round"
