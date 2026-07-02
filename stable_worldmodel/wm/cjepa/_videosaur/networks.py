"""MLP network module, vendored from martius-lab/videosaur (MIT License).

Source: videosaur/modules/networks.py — trimmed to the ``MLP`` class only
(the ``two_layer_mlp`` variant used by VideoSAUR's PushT config, i.e.
LayerNorm -> Linear -> ReLU -> Linear). Dropped the YAML-driven `build()`
function and the custom weight-init helper (`init_parameters`): this
project always overwrites weights via `load_state_dict` right after
construction, so the random-init scheme used before loading doesn't matter.
"""

from typing import List, Union

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = 'relu',
        final_activation: Union[bool, str] = False,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = 'relu'
            layers.append(_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp


def _activation_fn(name_or_instance: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_instance, nn.Module):
        return name_or_instance
    if name_or_instance.lower() == 'relu':
        return nn.ReLU(inplace=True)
    if name_or_instance.lower() == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unknown activation function {name_or_instance}')
