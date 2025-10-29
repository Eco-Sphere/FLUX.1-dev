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

from typing import Optional

import torch

from diffusers.models.attention_processor import Attention
from mindiesd import attention_forward
from mindiesd import rotary_position_embedding

from .sequence_length_tracker import get_global_seq
from .comm import all_to_all_single_4D
def apply_rotary_emb_mindiesd(x, freqs_cis, head_first=True):
    cos, sin = freqs_cis
    if head_first:
        cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
        sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
    else:
        cos = cos.reshape(1, cos.shape[0], 1, cos.shape[1])
        sin = sin.reshape(1, sin.shape[0], 1, sin.shape[1])
    cos, sin = cos.to(x.device), sin.to(x.device)

    return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=head_first, fused=True)

def apply_fa(query, key, value, attention_mask):
    batch_size = query.shape[0]
    heads = query.shape[-2]
    head_dim = query.shape[-1]
    
    hidden_states = attention_forward(query, key, value, opt_mode="manual", attn_mask=attention_mask, 
                                      op_type="fused_attn_score", layout="BSND")
    return hidden_states.reshape(batch_size, -1, head_dim * heads)

class FluxAttnProcessor2_0(): 
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    
    def __init__(self, parallel_args):
        self.group = parallel_args["ulysses"]["group"]
        self.world_size = parallel_args["ulysses"]["world_size"]
        self.rank = parallel_args["ulysses"]["rank"]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        if encoder_hidden_states is not None: # 双流transformer block,需要先cat再调FA
            batch_size = encoder_hidden_states.shape[0]
            # `sample` projections.
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            inner_dim = value.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            query = all_to_all_single_4D(
                query, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="img",
                async_op=False)
            
            value = all_to_all_single_4D(
                value, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="img",
                async_op=False)

            key = all_to_all_single_4D(
                key, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="img",
                async_op=False)


            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            encoder_hidden_states_value_proj = all_to_all_single_4D(
                encoder_hidden_states_value_proj, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="txt",
                async_op=False)

            encoder_hidden_states_key_proj = all_to_all_single_4D(
                encoder_hidden_states_key_proj, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="txt",
                async_op=False)
            
            encoder_hidden_states_query_proj = all_to_all_single_4D(
                encoder_hidden_states_query_proj, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="txt",
                async_op=False)
            
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2) 
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2) 
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2) 
            if image_rotary_emb is not None: 
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb)
            query = query.transpose(2, 1).contiguous()
            if image_rotary_emb is not None: 
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb)
            key = key.transpose(2, 1).contiguous()
            value = value.transpose(2, 1).contiguous()

        else:
            batch_size = hidden_states.shape[0]
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            inner_dim = value.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # torch.Size([1, 24, 4800, 128])
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            query = all_to_all_single_4D(
                query, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="all",
                async_op=False)

            key = all_to_all_single_4D(
                key, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="all",
                async_op=False)
            
            value = all_to_all_single_4D(
                value, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=2,
                gather_dim=1,
                tensor_name="all",
                async_op=False)

            if image_rotary_emb is not None: 
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb)
            query = query.transpose(2, 1).contiguous()
            
            if image_rotary_emb is not None: 
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb)
            key = key.transpose(2, 1).contiguous()

            value = value.transpose(2, 1).contiguous()

        hidden_states = attention_forward(query, key, value, opt_mode="manual", attn_mask=attention_mask, 
                                        op_type="fused_attn_score", layout="BSND")
        hidden_states = hidden_states.transpose(2, 1).contiguous()

        if encoder_hidden_states is not None:
            text_seq = get_global_seq(tensor_name="txt")

            encoder_hidden_states, hidden_states = (
                hidden_states[:, :, :text_seq, :],
                hidden_states[:, :, text_seq:, :],
            )
            
            hidden_states = all_to_all_single_4D(
                hidden_states, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=1,
                gather_dim=2,
                tensor_name="img",
                async_op=False)
            
            encoder_hidden_states = all_to_all_single_4D(
                encoder_hidden_states, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=1,
                gather_dim=2,
                tensor_name="txt",
                async_op=False)
            
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = encoder_hidden_states.to(query.dtype)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        
        else:
            hidden_states = all_to_all_single_4D(
                hidden_states, 
                group=self.group, 
                world_size=self.world_size,
                scatter_dim=1,
                gather_dim=2,
                tensor_name="all")
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            return hidden_states   