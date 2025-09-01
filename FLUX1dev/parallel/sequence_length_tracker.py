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

_GLOBAL_SEQ = dict()
_SPLIT_SEQ = dict()
_SPLIT_STATES = dict()

def set_global_seq(tensor_name="all", seq_dim=1):
    _GLOBAL_SEQ[tensor_name] = seq_dim

def get_global_seq(tensor_name="all"):
    return _GLOBAL_SEQ[tensor_name]

def get_split_seq_list(tensor_name="all"):
    return _SPLIT_SEQ[tensor_name]

def get_split_states(tensor_name="all"):
    return _SPLIT_STATES[tensor_name]

def set_split_seq(tensor_name="all", world_size=2):
    global_seq = get_global_seq(tensor_name)
    if global_seq % world_size == 0:
        split_seq = global_seq // world_size
        split_list = [split_seq for _ in range(world_size)]
        _SPLIT_SEQ[tensor_name] = split_list
        _SPLIT_STATES[tensor_name] = True
    else:
        split_seq1 = global_seq // world_size + 1
        split_seq2 = global_seq - split_seq1 * (world_size - 1)
        split_list = [split_seq1 if i != (world_size - 1) else split_seq2 for i in range(world_size)]
        _SPLIT_SEQ[tensor_name] = split_list
        _SPLIT_STATES[tensor_name] = False