"""
Network architecture configuration schema.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NetworkType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


class NetworkConfig(BaseModel):
    network_type: NetworkType = Field(default=NetworkType.MLP)
    hidden_sizes: List[int] = Field(default=[256, 256])
    cnn_sizes: Optional[List[List[int]]] = None
    # LSTM
    hidden_size: Optional[int] = 64
    num_layers: Optional[int] = 2
    # Transformer
    d_model: Optional[int] = 128
    nhead: Optional[int] = 4
    dim_feedforward: Optional[int] = 256
