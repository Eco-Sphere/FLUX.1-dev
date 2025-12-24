#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
from ..sequence_length_tracker import get_split_states, get_split_seq_list

def split_equal(input_, world_size, rank, dim):
    input_ = input_.chunk(world_size, dim=dim)[rank]
    return input_
def split_unequal(input_, world_size, rank, dim, seq_name="img"):
    split_size_list = get_split_seq_list(seq_name)
    input_ = torch.split(input_, split_size_list, dim=dim)[rank].contiguous()
    return input_
def split(input_, world_size, rank, dim, seq_name="img"):
    img_split_state = get_split_states(seq_name)
    if img_split_state:
        input_ = split_equal(input_, world_size, rank, dim)
    else:
        input_ = split_unequal(input_, world_size, rank, dim)
    return input_
def gather_forward_split_equal(input_, group, world_size):
    input_shape = input_.shape
    input_full = torch.empty([world_size, *input_shape], dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(input_full, input_, group=group)
    input_full = input_full.permute(1, 0, 2, 3).reshape(input_shape[0], -1, input_shape[2])
    return input_full
def gather_forward_split_unequal(input_, group, world_size, seq_name="img"):
    b, s, h = input_.shape
    split_size_list = get_split_seq_list(seq_name)
    input_full_list = [torch.empty([b, split_size_list[i], h], dtype=input_.dtype, device=input_.device) for i in range(world_size)]
    torch.distributed.all_gather(input_full_list, input_, group=group)
    input_full = torch.cat(input_full_list, dim=1)
    return input_full

def gather(input_, group, world_size, seq_name="img"):
    img_split_state = get_split_states(seq_name)
    if img_split_state:
        input_ = gather_forward_split_equal(input_, group, world_size)
    else:
        input_ = gather_forward_split_unequal(input_, group, world_size)
    return input_