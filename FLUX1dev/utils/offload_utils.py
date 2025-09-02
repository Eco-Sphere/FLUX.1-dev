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
import torch_npu
from torch_npu.contrib import transfer_to_npu

def onload_parameters_to_device(module: torch.nn.Module):
    for name, p in module.named_parameters():
        p.data.untyped_storage().resize_(p.storage_size)
        if p.is_slice_tensor:
            p.data.copy_(p.p_cpu, non_blocking=True)
        else:
            p.data.untyped_storage().copy_(p.p_cpu.untyped_storage(), non_blocking=True)

def onload_buffers_to_device(module: torch.nn.Module):
    for name, p in module.named_buffers():
        p.data.untyped_storage().resize_(p.storage_size)
        if p.is_slice_tensor:
            p.data.copy_(p.p_cpu, non_blocking=True)
        else:
            p.data.untyped_storage().copy_(p.p_cpu.untyped_storage(), non_blocking=True)

def offload_parameters_to_memory(module: torch.nn.Module):
    for name, p in module.named_parameters():
        p.data.untyped_storage().resize_(0)  
    
def offload_buffers_to_memory(module: torch.nn.Module):
    for name, p in module.named_buffers():
        p.data.untyped_storage().resize_(0) 

def initialize_parameters_on_memory(module: torch.nn.Module):
    for name, p in module.named_parameters():
        p_cpu = torch.empty(p.data.shape, dtype=p.dtype, pin_memory=True, device="cpu")
        setattr(p, "p_cpu", p_cpu) 

        is_slice_tensor = p.data.untyped_storage().size() != p.data.numel()  
        storage_size = p.data.untyped_storage().size()
        if is_slice_tensor:
            p.p_cpu.copy_(p.data, non_blocking=True)  
        else:
            p.p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
        
        setattr(p, "storage_size", storage_size)
        setattr(p, "is_slice_tensor", is_slice_tensor)

        p.data.untyped_storage().resize_(0)  

def initialize_buffers_on_memory(module: torch.nn.Module):
    for name, p in module.named_buffers():
        p_cpu = torch.empty(p.data.shape, dtype=p.dtype, pin_memory=True, device="cpu")
        setattr(p, "p_cpu", p_cpu)  

        is_slice_tensor = p.data.untyped_storage().size() != p.data.numel()  
        storage_size = p.data.untyped_storage().size()
        if is_slice_tensor:
            p.p_cpu.copy_(p.data, non_blocking=True)  
        else:
            p.p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
        
        setattr(p, "storage_size", storage_size)
        setattr(p, "is_slice_tensor", is_slice_tensor)

        p.data.untyped_storage().resize_(0)  


class BlockOffloadHook():
    r"""
    Block Level Offload
    """

    def __init__(
        self, 
        model,
        h2d_stream,
        d2h_stream,
        events,
        block_nums,
        block_on_npu_nums,
    ) -> None:
        self.model = model
        self.events = events
        self.h2d_stream = h2d_stream
        self.d2h_stream = d2h_stream
        self.block_nums = block_nums
        self.block_on_npu_nums = block_on_npu_nums

    def onload_block_to_device(self, module: torch.nn.Module, input):
        to_device_index = module.index + self.block_on_npu_nums
        forward_event = torch.npu.Event()
        forward_event.record()
        if to_device_index < self.block_nums:
            with torch.npu.stream(self.h2d_stream):
                self.h2d_stream.wait_event(forward_event)
                onload_parameters_to_device(self.model[to_device_index])
                onload_buffers_to_device(self.model[to_device_index])
                self.events[to_device_index].record()
        torch.npu.current_stream().wait_event(self.events[module.index])

    def offload_block_to_memory(self, module: torch.nn.Module, input, output):
        to_device_index = module.index
        if to_device_index >= self.block_on_npu_nums:
            forward_event = torch.npu.Event()
            forward_event.record()
            with torch.npu.stream(self.d2h_stream):
                self.d2h_stream.wait_event(forward_event)  
                offload_parameters_to_memory(module)
                offload_buffers_to_memory(module)
        torch.npu.current_stream().wait_stream(self.d2h_stream)



def apply_block_level_offload(
    module: torch.nn.Module,
    onload_device,
    block_on_npu_nums):

    block_nums = len(module)
    h2d_stream = torch.npu.Stream()
    d2h_stream = torch.npu.Stream()
    block_on_npu_nums = block_on_npu_nums
    events = []
    for _ in range(block_nums):
        events.append(torch.npu.Event())

    hook = BlockOffloadHook(
        module,
        h2d_stream,
        d2h_stream,
        events,
        block_nums,
        block_on_npu_nums
        )
    
    for idx, block in enumerate(module):
        block.index = idx

    with torch.npu.stream(h2d_stream):
        for blk_idx in range(block_nums):
            module[blk_idx].to(onload_device)
            
            if blk_idx >= block_on_npu_nums:
                initialize_parameters_on_memory(module[blk_idx])
                initialize_buffers_on_memory(module[blk_idx])

    torch.npu.current_stream().wait_stream(h2d_stream)

    for blk_idx in range(block_nums):
        module[blk_idx].register_forward_pre_hook(hook.onload_block_to_device)  
        if blk_idx >= block_on_npu_nums:
            module[blk_idx].register_forward_hook(hook.offload_block_to_memory)  
