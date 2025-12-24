# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn
import torch_npu

from diffusers.utils.import_utils import is_torch_version
from ..utils import get_world_size


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, is_tp: bool = False):
        super().__init__()
        if is_tp:
            dim_out = dim_out // get_world_size()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type == "mps" and is_torch_version("<", "2.0.0"):
            # fp16 gelu not supported on mps before torch 2.0
            return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
def fast_gelu(input, approximate='none'):
    return torch_npu.npu_fast_gelu(input)

def enable_fast_gelu():
    torch.nn.functional.gelu = fast_gelu