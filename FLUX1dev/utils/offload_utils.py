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

from mindiesd import CacheConfig

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
        onload_device: str = "npu",
        block_on_npu_nums: int = 2,
        cache_config: CacheConfig = None
    ) -> None:
        self.model = model
        self.onload_device = onload_device
        self.block_nums = len(self.model)
        self.events = []
        for _ in range(self.block_nums):
            self.events.append(torch.npu.Event())
        self.h2d_stream = torch.npu.Stream()
        self.d2h_stream = torch.npu.Stream()
        self.block_on_npu_nums = block_on_npu_nums
        self.cache_config = cache_config
        self.use_cache = False

        if self.cache_config is not None:
            assert self.cache_config.method == "dit_block_cache"
            assert self.cache_config.blocks_count == self.block_nums
            self.use_cache = self.check_cache_config()
            if self.use_cache:
                self.set_cache_states()

    def set_cache_states(self):
        self._cur_step = 0
        self.steps_count = self.cache_config.steps_count
        self.block_start = self.cache_config.block_start
        self.block_end = self.cache_config.block_end
        self.skip_blocks = self.block_end - self.block_start
    
        self.block_on_npu_nums = min(self.block_on_npu_nums, self.block_start)

        self.skip_steps = []
        for cur_step in range(0, self.steps_count):
            if cur_step >= self.cache_config.step_start:
                diftime = cur_step - self.cache_config.step_start
                if diftime % self.cache_config.step_interval != 0:
                    self.skip_steps.append(cur_step)

    def _counter(self, blk_idx):
        if (blk_idx + 1) == self.block_nums:
            self._cur_step += 1  
            if self._cur_step == self.steps_count:
                self._cur_step = 0  
                
    def check_cache_config(self):
        if self.cache_config.step_start >= self.cache_config.steps_count or \
            self.cache_config.step_end == self.cache_config.step_start:
            return False

        if self.cache_config.block_start >= self.cache_config.blocks_count or \
            self.cache_config.block_end == self.cache_config.block_start:
            return False
        
        if self.cache_config.step_interval == 1:
            return False

        return True

    def get_next_blk_idx(self, cur_blk_idx):
        if self.use_cache and self._cur_step in self.skip_steps:
            next_blk_idx = cur_blk_idx + self.block_on_npu_nums
            if next_blk_idx >= self.block_start and cur_blk_idx < self.block_end:
                next_blk_idx = next_blk_idx + self.skip_blocks
        else:
            next_blk_idx = cur_blk_idx + self.block_on_npu_nums
        return next_blk_idx

    def onload_block_to_device(self, block: torch.nn.Module, input):
        cur_blk_idx = block.index
        next_blk_idx = self.get_next_blk_idx(cur_blk_idx)
        forward_event = torch.npu.Event()
        forward_event.record()

        if next_blk_idx < self.block_nums:
            with torch.npu.stream(self.h2d_stream):
                self.h2d_stream.wait_event(forward_event)
                onload_parameters_to_device(self.model[next_blk_idx])
                onload_buffers_to_device(self.model[next_blk_idx])
                self.events[next_blk_idx].record()

        if self.use_cache:
            self._counter(cur_blk_idx)
        torch.npu.current_stream().wait_event(self.events[cur_blk_idx])

    def offload_block_to_memory(self, block: torch.nn.Module, input, output):
        cur_blk_idx = block.index
        if cur_blk_idx >= self.block_on_npu_nums:
            forward_event = torch.npu.Event()
            forward_event.record()
            with torch.npu.stream(self.d2h_stream):
                self.d2h_stream.wait_event(forward_event)  
                offload_parameters_to_memory(block)
                offload_buffers_to_memory(block)
        torch.npu.current_stream().wait_stream(self.d2h_stream)

    def register_hook(self):
        for idx, block in enumerate(self.model):
            block.index = idx

        with torch.npu.stream(self.h2d_stream):
            for blk_idx in range(self.block_nums):
                self.model[blk_idx].to(self.onload_device)
                
                if blk_idx >= self.block_on_npu_nums:
                    initialize_parameters_on_memory(self.model[blk_idx])
                    initialize_buffers_on_memory(self.model[blk_idx])

        torch.npu.current_stream().wait_stream(self.h2d_stream)

        for blk_idx in range(self.block_nums):
            self.model[blk_idx].register_forward_pre_hook(self.onload_block_to_device)  
            if blk_idx >= self.block_on_npu_nums:
                self.model[blk_idx].register_forward_hook(self.offload_block_to_memory)  



class BlockOffloadHookV2():
    r"""
    Block Level Offload V2
    """

    def __init__(
        self, 
        model,
        onload_device: str = "npu",
        block_on_npu_nums: int = 2,
        cache_config: CacheConfig = None
    ) -> None:
        self.model = model
        self.onload_device = onload_device
        self.block_nums = len(self.model)
        self.events = []
        for _ in range(self.block_nums):
            self.events.append(torch.npu.Event())
        self.h2d_stream = torch.npu.Stream()
        self.d2h_stream = torch.npu.Stream()
        self.block_on_npu_nums = block_on_npu_nums
        self.cache_config = cache_config
        self.use_cache = False

        if self.cache_config is not None:
            assert self.cache_config.method == "dit_block_cache"
            assert self.cache_config.blocks_count == self.block_nums
            self.use_cache = self.check_cache_config()
            if self.use_cache:
                self.set_cache_states()

    def set_cache_states(self):
        self._cur_step = -1
        self.steps_count = self.cache_config.steps_count
        self.block_start = self.cache_config.block_start
        self.block_end = self.cache_config.block_end
        self.skip_blocks = self.block_end - self.block_start
    
        self.block_on_npu_nums = min(self.block_on_npu_nums, self.block_start)

        self.skip_steps = []
        for cur_step in range(0, self.steps_count):
            if cur_step >= self.cache_config.step_start:
                diftime = cur_step - self.cache_config.step_start
                if diftime % self.cache_config.step_interval != 0:
                    self.skip_steps.append(cur_step)
                
    def check_cache_config(self):
        if self.cache_config.step_start >= self.cache_config.steps_count or \
            self.cache_config.step_end == self.cache_config.step_start:
            return False

        if self.cache_config.block_start >= self.cache_config.blocks_count or \
            self.cache_config.block_end == self.cache_config.block_start:
            return False
        
        if self.cache_config.step_interval == 1:
            return False

        return True

    def get_next_blk_idx(self, cur_blk_idx):
        if self.use_cache and self._cur_step in self.skip_steps:
            if cur_blk_idx == 0:
                next_blk_idx = cur_blk_idx + 1
            elif cur_blk_idx % 2 != 0:
                next_blk_idx = cur_blk_idx + 2
            if next_blk_idx >= self.block_start and cur_blk_idx < self.block_end:
                next_blk_idx = self.block_end
                if self.block_end % 2 == 0:
                    next_blk_idx += 1
        else:
            if cur_blk_idx == 0:
                next_blk_idx = cur_blk_idx + 1
            elif cur_blk_idx % 2 != 0:
                next_blk_idx = cur_blk_idx + 2

        return next_blk_idx

    def onload_block_to_device(self, block: torch.nn.Module, input):
        cur_blk_idx = block.index
        next_blk_idx = self.get_next_blk_idx(cur_blk_idx)
        forward_event = torch.npu.Event()
        forward_event.record()

        if next_blk_idx < self.block_nums:
            with torch.npu.stream(self.h2d_stream):
                self.h2d_stream.wait_event(forward_event)
                onload_parameters_to_device(self.model[next_blk_idx])
                onload_buffers_to_device(self.model[next_blk_idx])
                self.events[next_blk_idx].record()

        torch.npu.current_stream().wait_event(self.events[cur_blk_idx])

    def offload_block_to_memory(self, block: torch.nn.Module, input, output):
        forward_event = torch.npu.Event()
        forward_event.record()
        with torch.npu.stream(self.d2h_stream):
            self.d2h_stream.wait_event(forward_event)  
            offload_parameters_to_memory(block)
            offload_buffers_to_memory(block)
        torch.npu.current_stream().wait_stream(self.d2h_stream)

    def count_step(self, block: torch.nn.Module, input):
        self._cur_step += 1  
        if self._cur_step == self.steps_count:
            self._cur_step = -1  

    def register_hook(self):
        for idx, block in enumerate(self.model):
            block.index = idx

        with torch.npu.stream(self.h2d_stream):
            for blk_idx in range(self.block_nums):
                self.model[blk_idx].to(self.onload_device)
                
                if blk_idx % 2 != 0:
                    # blk_idx为奇数的都需要offload, 为偶数的常驻npu
                    initialize_parameters_on_memory(self.model[blk_idx])
                    initialize_buffers_on_memory(self.model[blk_idx])

        torch.npu.current_stream().wait_stream(self.h2d_stream)

        for blk_idx in range(self.block_nums):
            if blk_idx == 0:
                self.model[blk_idx].register_forward_pre_hook(self.onload_block_to_device)  
            elif blk_idx % 2 != 0:
                self.model[blk_idx].register_forward_pre_hook(self.onload_block_to_device) 

            if blk_idx % 2 != 0:
                self.model[blk_idx].register_forward_hook(self.offload_block_to_memory)  

        if self.use_cache:
            # self.model[self.block_nums-1].register_forward_hook(self.count_step)  
            self.model[0].register_forward_pre_hook(self.count_step) 