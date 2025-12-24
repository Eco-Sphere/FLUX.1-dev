# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Union, List, Tuple
import numpy as np

import os
import torch
from torch import nn
import torch_npu

from mindiesd import rotary_position_embedding

class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.use_cache = False
        self.seq_parallel = False

        self.use_cache = bool(int(os.environ.get('POSEMB_CACHE', 0)))
        if self.use_cache:
            self.steps_count = int(os.environ.get('INFER_STEPS', 50))
            self.cache = None
            self._cur_step = 0

    def enable_seq_parallel(self, group=None):
        import torch.distributed as dist
        self.seq_parallel = True
        self.world_size = dist.get_world_size(group=group)

    def _counter(self):
        self._cur_step += 1  
        if self._cur_step == self.steps_count:
            self._cur_step = 0  

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device).to(torch.bfloat16)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device).to(torch.bfloat16)
        if self.seq_parallel:
            from FLUX1dev.parallel.sequence_length_tracker import get_global_seq
            from FLUX1dev.parallel.freqs_utils import get_rotary_emb_sp
            txt_seq_length = get_global_seq(tensor_name="txt")
            freqs_cos_sp, freqs_sin_sp = get_rotary_emb_sp((freqs_cos, freqs_sin), txt_seq_length, world_size=self.world_size)
            return (freqs_cos, freqs_sin), (freqs_cos_sp, freqs_sin_sp)
        else:
            return (freqs_cos, freqs_sin), (freqs_cos, freqs_sin)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.use_cache:
            if self._cur_step == 0:
                output = self._forward(ids)
                self.cache = output
            else:
                output = self.cache
            self._counter()
            return output
        else:
            return self._forward(ids)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs).float()  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    layout="BNSD"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if layout == "BNSD":
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        cos, sin = cos.to(x.device), sin.to(x.device)
    
        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def apply_rotary_emb_mindiesd(x, freqs_cis, layout="BNSD"):
    cos, sin = freqs_cis
    if layout == "BNSD":
        cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
        sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
        cos, sin = cos.to(x.device), sin.to(x.device)
        output = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=True, fused=True)
    else:
        cos = cos.reshape(1, cos.shape[0], 1, cos.shape[-1])
        sin = sin.reshape(1, sin.shape[0], 1, sin.shape[-1])
        cos, sin = cos.to(x.device), sin.to(x.device)
        output = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=False, fused=True)
    return output