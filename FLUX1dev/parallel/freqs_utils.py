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

import torch
from .sequence_length_tracker import get_split_states, get_split_seq_list

def get_rotary_emb_sp_equal(image_rotary_emb, encoder_seq, world_size=1):
    cos, sin = image_rotary_emb
    cos_encoder_hidden_states = cos[:encoder_seq, ]
    cos_hidden_states = cos[encoder_seq:, ]
    sin_encoder_hidden_states = sin[:encoder_seq, ]
    sin_hidden_states = sin[encoder_seq:, ]
    
    cos_encoder_hidden_states_split, \
        cos_hidden_states_split, \
            sin_encoder_hidden_states_split, \
                sin_hidden_states_split = [i.chunk(world_size, dim=0) for i in 
                                           (cos_encoder_hidden_states, 
                                            cos_hidden_states,
                                            sin_encoder_hidden_states,
                                            sin_hidden_states)]
    new_cos_list = []
    for i in range(world_size):
        new_cos_list.append(cos_encoder_hidden_states_split[i])
        new_cos_list.append(cos_hidden_states_split[i])

    new_sin_list = []
    for i in range(world_size):
        new_sin_list.append(sin_encoder_hidden_states_split[i])
        new_sin_list.append(sin_hidden_states_split[i])
    cos_split = torch.cat(new_cos_list, dim=0).contiguous()
    sin_split = torch.cat(new_sin_list, dim=0).contiguous()
    image_rotary_emb_split = [cos_split, sin_split]
    return image_rotary_emb_split

def get_rotary_emb_sp_unequal(image_rotary_emb, encoder_seq, world_size=1):
    cos, sin = image_rotary_emb
    cos_encoder_hidden_states = cos[:encoder_seq, ]
    cos_hidden_states = cos[encoder_seq:, ]
    sin_encoder_hidden_states = sin[:encoder_seq, ]
    sin_hidden_states = sin[encoder_seq:, ]
    
    split_size_list = get_split_seq_list("img")

    cos_encoder_hidden_states_split = cos_encoder_hidden_states.chunk(world_size, dim=0)
    sin_encoder_hidden_states_split = sin_encoder_hidden_states.chunk(world_size, dim=0)
    cos_hidden_states_split = [t.contiguous() for t in torch.split(cos_hidden_states, split_size_list, dim=0)]
    sin_hidden_states_split = [t.contiguous() for t in torch.split(sin_hidden_states, split_size_list, dim=0)]

    new_cos_list = []
    for i in range(world_size):
        new_cos_list.append(cos_encoder_hidden_states_split[i])
        new_cos_list.append(cos_hidden_states_split[i])

    new_sin_list = []
    for i in range(world_size):
        new_sin_list.append(sin_encoder_hidden_states_split[i])
        new_sin_list.append(sin_hidden_states_split[i])

    cos_split = torch.cat(new_cos_list, dim=0).contiguous()
    sin_split = torch.cat(new_sin_list, dim=0).contiguous()
    image_rotary_emb_split = [cos_split, sin_split]
    return image_rotary_emb_split

def get_rotary_emb_sp(image_rotary_emb, encoder_seq, world_size=1):
    img_split_state = get_split_states("img")
    if img_split_state:
        image_rotary_emb_sp = get_rotary_emb_sp_equal(image_rotary_emb, encoder_seq, world_size)
    else:
        image_rotary_emb_sp = get_rotary_emb_sp_unequal(image_rotary_emb, encoder_seq, world_size)
    return image_rotary_emb_sp