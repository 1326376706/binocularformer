from enum import Enum

class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"