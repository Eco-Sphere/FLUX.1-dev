# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

from typing import Any, Tuple
import functools

import torch
import torch.distributed as dist
from torch import Tensor
import torch.distributed

from ..sequence_length_tracker import *

# ====================
# All-To-All-Single
# ====================

def all_to_all_single_4D(
            hidden_states: Tensor,
            group: dist.ProcessGroup,
            world_size: int = 1,
            rank: int = 0,
            scatter_dim: int = 2,
            gather_dim: int = 1,
            tensor_name=None,
            async_op=False,
    ) -> Tensor:
    # 默认是等长切分
    if tensor_name is None:
        if scatter_dim == 2 and gather_dim == 1:
            return SeqAllToAllSingle4D.gather_seq_equal(hidden_states, group, world_size, async_op=async_op)
        elif scatter_dim == 1 and gather_dim == 2:
            return SeqAllToAllSingle4D.gather_head_equal(hidden_states, group, world_size, async_op=async_op)
    else:
        split_state = get_split_states(tensor_name)
        # 等长切分
        if split_state:
            if scatter_dim == 2 and gather_dim == 1:
                return SeqAllToAllSingle4D.gather_seq_equal(hidden_states, group, world_size, async_op=async_op)
            elif scatter_dim == 1 and gather_dim == 2:
                return SeqAllToAllSingle4D.gather_head_equal(hidden_states, group, world_size, async_op=async_op)
        # 不等长切分
        else:
            if scatter_dim == 2 and gather_dim == 1:
                return SeqAllToAllSingle4D.gather_seq_unequal(hidden_states, group, world_size, tensor_name, async_op=async_op)
            elif scatter_dim == 1 and gather_dim == 2:
                return SeqAllToAllSingle4D.gather_head_unequal(hidden_states, group, world_size, tensor_name, async_op=async_op)
            
class SeqAllToAllSingle4D():
    """
    hidden_states必须符合BNSD的4D格式
    """
    @staticmethod
    def gather_seq_equal(
            hidden_states: Tensor,
            group: dist.ProcessGroup,
            world_size: int,
            async_op=False
    ) -> Tensor:
        b, n, s, d = hidden_states.shape
        split_n = n // world_size
        hidden_states_split = hidden_states.view(b, world_size, split_n, s, d).permute(1,0,2,3,4).contiguous()
        hidden_states_all_seq = torch.empty_like(hidden_states_split)
        # world_size, b, split_n, s, d
        def func():
            hidden_states = hidden_states_all_seq.permute(1,2,0,3,4).contiguous().view(b, split_n, s*world_size, d)
            return hidden_states
        
        if async_op:
            work = dist.all_to_all_single(hidden_states_all_seq, hidden_states_split, group=group, async_op=async_op)
            return work, func
        else:
            dist.all_to_all_single(hidden_states_all_seq, hidden_states_split, group=group, async_op=async_op)
            hidden_states = func()
            return hidden_states

    @staticmethod
    def gather_seq_unequal(
            hidden_states: Tensor,
            group: dist.ProcessGroup,
            world_size: int,
            tensor_name=None,
            async_op=False
    ) -> Tensor:
        b, n, s, d = hidden_states.shape
        split_n = n // world_size
        total_s = get_global_seq(tensor_name)
        input_split = [s for i in range(world_size)]
        output_split = get_split_seq_list(tensor_name)

        hidden_states = hidden_states.view(b, world_size, split_n, s, d).contiguous()
        hidden_states = hidden_states.transpose(0, 1).reshape(world_size, b*split_n, s, d).transpose(1,2)
        hidden_states_split = hidden_states.reshape(world_size*s, b*split_n, d)

        hidden_states_all_seq = torch.empty([total_s, b*split_n, d], dtype=hidden_states.dtype, device="npu")
        # world_size, b, split_n, s, d

        def func():
            hidden_states = hidden_states_all_seq.transpose(0, 1).reshape(b, split_n, total_s, d)
            return hidden_states

        if async_op:
            work = dist.all_to_all_single(
                hidden_states_all_seq, 
                hidden_states_split, 
                group=group,
                input_split_sizes=input_split, 
                output_split_sizes=output_split, 
                async_op=async_op)
            return work, func
        
        else:
            dist.all_to_all_single(
                hidden_states_all_seq, 
                hidden_states_split, 
                group=group,
                input_split_sizes=input_split, 
                output_split_sizes=output_split, 
                async_op=async_op)
            
            hidden_states = func()
            return hidden_states
    
    @staticmethod
    def gather_head_equal(
            hidden_states: Tensor,
            group: dist.ProcessGroup,
            world_size: int,
            async_op=False
    ) -> Tensor:
        b, n, s, d = hidden_states.shape
        split_s = s // world_size
        hidden_states_split = hidden_states.view(b, n, world_size, split_s, d).permute(2,0,1,3,4).contiguous()
        hidden_states_all_head = torch.empty_like(hidden_states_split)
        # world_size, b, n, split_s, d
        def func():
            hidden_states = hidden_states_all_head.permute(1,0,2,3,4).contiguous().view(b, n*world_size, split_s, d)
            return hidden_states
        
        if async_op:
            work = dist.all_to_all_single(hidden_states_all_head, hidden_states_split, group=group, async_op=async_op)
            return work, func
        else:
            dist.all_to_all_single(hidden_states_all_head, hidden_states_split, group=group, async_op=async_op)
            hidden_states = func()
            return hidden_states

    @staticmethod
    def gather_head_unequal(
            hidden_states: Tensor,
            group: dist.ProcessGroup,
            world_size: int,
            tensor_name=None,
            async_op=False
    ) -> Tensor:
        rank = torch.distributed.get_rank()
        b, n, s, d = hidden_states.shape
        split_s = s // world_size
        input_split = get_split_seq_list(tensor_name)
        hidden_states_split = [t.contiguous() for t in torch.split(hidden_states, input_split, dim=2)]
        hidden_states_all_head = [torch.empty([b, n, input_split[rank], d], dtype=hidden_states.dtype, device="npu") for i in range(world_size)]
        def func():
            dist.all_to_all(hidden_states_all_head, hidden_states_split, group=group, async_op=async_op)
            return torch.cat(hidden_states_all_head, dim=1).contiguous()
         
        if async_op:
            work = dist.all_to_all(hidden_states_all_head, hidden_states_split, group=group, async_op=async_op)
            return work, func
        else:
            dist.all_to_all(hidden_states_all_head, hidden_states_split, group=group, async_op=async_op)
            return func()