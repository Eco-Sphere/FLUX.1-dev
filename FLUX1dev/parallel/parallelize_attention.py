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
import os
import torch

from diffusers.models.attention_processor import Attention
from mindiesd import attention_forward

from .sequence_length_tracker import get_global_seq
from .comm import all_to_all_single_4D

from FLUX1dev.layers import apply_rotary_emb, apply_rotary_emb_mindiesd


class FluxSingleAttnProcessor2_0():
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, parallel_args):
        self.group = parallel_args["ulysses"]["group"]
        self.world_size = parallel_args["ulysses"]["world_size"]
        self.rank = parallel_args["ulysses"]["rank"]

        self.use_la = bool(int(os.environ.get("ENABLE_LA", 0)))
        self.use_fa_quant = bool(int(os.environ.get("USE_FA_QUANT", 0)))
        self.use_fuse_rope = bool(int(os.environ.get("ROPE_FUSE", 0)))
        self.use_fuse_rmsnorm = bool(int(os.environ.get("RMSNORM_FUSE", 0)))
        self.comm_async = bool(int(os.environ.get("COMM_OVERLAP", 0)))

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if self.comm_async:
            return self.forward_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
        else:
            return self.forward_non_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

    def forward_non_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = value.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # torch.Size([1, 24, 4800, 128])
        if attn.norm_q is not None:
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)
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
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)

        query = query.transpose(2, 1).contiguous()

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb)
            else:
                key = apply_rotary_emb(key, image_rotary_emb)
        key = key.transpose(2, 1).contiguous()

        value = value.transpose(2, 1).contiguous()

        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

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

    def forward_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        value = attn.to_v(hidden_states)
        inner_dim = value.shape[-1]
        head_dim = inner_dim // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_work, value_func = all_to_all_single_4D(
            value,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="all",
            async_op=True)

        query = attn.to_q(hidden_states)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # torch.Size([1, 24, 4800, 128])
        if attn.norm_q is not None:
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        query_work, query_func = all_to_all_single_4D(
            query,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="all",
            async_op=True)

        key = attn.to_k(hidden_states)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)
        key_work, key_func = all_to_all_single_4D(
            key,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="all",
            async_op=True)

        value_work.wait()
        value = value_func()
        query_work.wait()
        query = query_func()
        key_work.wait()
        key = key_func()

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
        query = query.transpose(2, 1).contiguous()

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb)
            else:
                key = apply_rotary_emb(key, image_rotary_emb)
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()

        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

        hidden_states_work, hidden_states_func = all_to_all_single_4D(
            hidden_states,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=1,
            gather_dim=2,
            tensor_name="all",
            async_op=True)

        def output_func():
            hidden_states = hidden_states_func()
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            return hidden_states

        return hidden_states_work, output_func


class FluxAttnProcessor2_0():
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, parallel_args):
        self.group = parallel_args["ulysses"]["group"]
        self.world_size = parallel_args["ulysses"]["world_size"]
        self.rank = parallel_args["ulysses"]["rank"]

        self.use_la = bool(int(os.environ.get("ENABLE_LA", 0)))
        self.use_fa_quant = bool(int(os.environ.get("USE_FA_QUANT", 0)))
        self.use_fuse_rope = bool(int(os.environ.get("ROPE_FUSE", 0)))
        self.use_fuse_rmsnorm = bool(int(os.environ.get("RMSNORM_FUSE", 0)))
        self.comm_async = bool(int(os.environ.get("COMM_OVERLAP", 0)))

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:
        if self.comm_async:
            return self.forward_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                pre_encoder_query,
                pre_encoder_key,
                pre_encoder_value,
                cal_encoder_qkv
            )
        else:
            return self.forward_non_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                pre_encoder_query,
                pre_encoder_key,
                pre_encoder_value,
                cal_encoder_qkv
            )

    def forward_non_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:

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
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)

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
        if cal_encoder_qkv:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_query_proj = pre_encoder_query
            encoder_hidden_states_key_proj = pre_encoder_key
            encoder_hidden_states_value_proj = pre_encoder_value

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
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj,
                                                                 if_fused=self.use_fuse_rmsnorm)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj,
                                                               if_fused=self.use_fuse_rmsnorm)

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
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb, layout="BNSD")
            else:
                query = apply_rotary_emb(query, image_rotary_emb, layout="BNSD")

        query = query.transpose(2, 1).contiguous()
        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb, layout="BNSD")
            else:
                key = apply_rotary_emb(key, image_rotary_emb, layout="BNSD")
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()

        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

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

    def forward_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:

        batch_size, text_seq = encoder_hidden_states.shape[:2]
        value = attn.to_v(hidden_states)
        inner_dim = value.shape[-1]
        head_dim = inner_dim // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_work, value_func = all_to_all_single_4D(
            value,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        query_work, query_func = all_to_all_single_4D(
            query,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        key = attn.to_k(hidden_states)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)
        key_work, key_func = all_to_all_single_4D(
            key,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        if cal_encoder_qkv:
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_value_proj = pre_encoder_value

        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_value_work, encoder_value_func = all_to_all_single_4D(
            encoder_hidden_states_value_proj,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="txt",
            async_op=True)

        # `context` projections.
        if cal_encoder_qkv:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_query_proj = pre_encoder_query
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj,
                                                                 if_fused=self.use_fuse_rmsnorm)

        encoder__query_work, encoder_query_func = all_to_all_single_4D(
            encoder_hidden_states_query_proj,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="txt",
            async_op=True)

        if cal_encoder_qkv:
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_key_proj = pre_encoder_key
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj,
                                                               if_fused=self.use_fuse_rmsnorm)

        encoder_key_work, encoder_key_func = all_to_all_single_4D(
            encoder_hidden_states_key_proj,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="txt",
            async_op=True)

        value_work.wait()
        value = value_func()
        key_work.wait()
        key = key_func()
        query_work.wait()
        query = query_func()

        encoder_value_work.wait()
        encoder_hidden_states_value_proj = encoder_value_func()
        encoder__query_work.wait()
        encoder_hidden_states_query_proj = encoder_query_func()
        encoder_key_work.wait()
        encoder_hidden_states_key_proj = encoder_key_func()

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb, layout="BNSD")
            else:
                query = apply_rotary_emb(query, image_rotary_emb, layout="BNSD")

        query = query.transpose(2, 1).contiguous()
        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb, layout="BNSD")
            else:
                key = apply_rotary_emb(key, image_rotary_emb, layout="BNSD")
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()
        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

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


class FluxAttnProcessor2_0_TxtNonSplit():
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, parallel_args):
        self.group = parallel_args["ulysses"]["group"]
        self.world_size = parallel_args["ulysses"]["world_size"]
        self.rank = parallel_args["ulysses"]["rank"]

        self.use_la = bool(int(os.environ.get("ENABLE_LA", 0)))
        self.use_fa_quant = bool(int(os.environ.get("USE_FA_QUANT", 0)))
        self.use_fuse_rope = bool(int(os.environ.get("ROPE_FUSE", 0)))
        self.use_fuse_rmsnorm = bool(int(os.environ.get("RMSNORM_FUSE", 0)))
        self.comm_async = bool(int(os.environ.get("COMM_OVERLAP", 0)))

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:
        if self.comm_async:
            return self.forward_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                pre_encoder_query,
                pre_encoder_key,
                pre_encoder_value,
                cal_encoder_qkv
            )
        else:
            return self.forward_non_overlap(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                pre_encoder_query,
                pre_encoder_key,
                pre_encoder_value,
                cal_encoder_qkv
            )

    def forward_non_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:

        batch_size, text_seq = encoder_hidden_states.shape[:2]
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
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)

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
        if cal_encoder_qkv:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_query_proj = pre_encoder_query
            encoder_hidden_states_key_proj = pre_encoder_key
            encoder_hidden_states_value_proj = pre_encoder_value

        encoder_hidden_states_query_proj = torch.chunk(
            encoder_hidden_states_query_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = torch.chunk(
            encoder_hidden_states_key_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = torch.chunk(
            encoder_hidden_states_value_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj,
                                                                 if_fused=self.use_fuse_rmsnorm)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj,
                                                               if_fused=self.use_fuse_rmsnorm)

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb, layout="BNSD")
            else:
                query = apply_rotary_emb(query, image_rotary_emb, layout="BNSD")

        query = query.transpose(2, 1).contiguous()
        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb, layout="BNSD")
            else:
                key = apply_rotary_emb(key, image_rotary_emb, layout="BNSD")
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()

        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

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

        encoder_hidden_states_shape = encoder_hidden_states.shape
        encoder_hidden_states_full = torch.empty([self.world_size, *encoder_hidden_states_shape],
                                                 dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
        torch.distributed.all_gather_into_tensor(encoder_hidden_states_full, encoder_hidden_states.contiguous(),
                                                 group=self.group)
        encoder_hidden_states = encoder_hidden_states_full.permute(1, 0, 2, 3, 4).reshape(
            encoder_hidden_states_shape[0], -1, encoder_hidden_states_shape[2], encoder_hidden_states_shape[3])

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

    def forward_overlap(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            pre_encoder_query: Optional[torch.Tensor] = None,
            pre_encoder_key: Optional[torch.Tensor] = None,
            pre_encoder_value: Optional[torch.Tensor] = None,
            cal_encoder_qkv: bool = True
    ) -> torch.FloatTensor:

        batch_size, text_seq = encoder_hidden_states.shape[:2]
        value = attn.to_v(hidden_states)
        inner_dim = value.shape[-1]
        head_dim = inner_dim // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_work, value_func = all_to_all_single_4D(
            value,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        query = attn.to_q(hidden_states)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query, if_fused=self.use_fuse_rmsnorm)
        query_work, query_func = all_to_all_single_4D(
            query,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        key = attn.to_k(hidden_states)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            key = attn.norm_k(key, if_fused=self.use_fuse_rmsnorm)
        key_work, key_func = all_to_all_single_4D(
            key,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=2,
            gather_dim=1,
            tensor_name="img",
            async_op=True)

        # `context` projections.
        if cal_encoder_qkv:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_query_proj = pre_encoder_query
            encoder_hidden_states_key_proj = pre_encoder_key
            encoder_hidden_states_value_proj = pre_encoder_value

        encoder_hidden_states_query_proj = torch.chunk(
            encoder_hidden_states_query_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = torch.chunk(
            encoder_hidden_states_key_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = torch.chunk(
            encoder_hidden_states_value_proj,
            self.world_size, dim=2)[self.rank]
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, text_seq, -1, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj,
                                                                 if_fused=self.use_fuse_rmsnorm)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj,
                                                               if_fused=self.use_fuse_rmsnorm)

        value_work.wait()
        value = value_func()
        query_work.wait()
        query = query_func()
        key_work.wait()
        key = key_func()

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)

        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                query = apply_rotary_emb_mindiesd(query, image_rotary_emb, layout="BNSD")
            else:
                query = apply_rotary_emb(query, image_rotary_emb, layout="BNSD")

        query = query.transpose(2, 1).contiguous()
        if image_rotary_emb is not None:
            if self.use_fuse_rope:
                key = apply_rotary_emb_mindiesd(key, image_rotary_emb, layout="BNSD")
            else:
                key = apply_rotary_emb(key, image_rotary_emb, layout="BNSD")
        key = key.transpose(2, 1).contiguous()
        value = value.transpose(2, 1).contiguous()

        hidden_states = attn.apply_fa(query, key, value, attention_mask, use_la=self.use_la,
                                      use_fa_quant=self.use_fa_quant, world_size=self.world_size)
        hidden_states = hidden_states.transpose(2, 1).contiguous()

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :, :text_seq, :],
            hidden_states[:, :, text_seq:, :],
        )

        encoder_hidden_states_shape = encoder_hidden_states.shape
        encoder_hidden_states_full = torch.empty([self.world_size, *encoder_hidden_states_shape],
                                                 dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
        torch.distributed.all_gather_into_tensor(encoder_hidden_states_full, encoder_hidden_states.contiguous(),
                                                 group=self.group)

        hidden_states_work, hidden_states_func = all_to_all_single_4D(
            hidden_states,
            group=self.group,
            world_size=self.world_size,
            scatter_dim=1,
            gather_dim=2,
            tensor_name="img",
            async_op=True)
        encoder_hidden_states = encoder_hidden_states_full.permute(1, 0, 2, 3, 4).reshape(
            encoder_hidden_states_shape[0], -1, encoder_hidden_states_shape[2], encoder_hidden_states_shape[3])
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states_work.wait()
        hidden_states = hidden_states_func()
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, encoder_hidden_states
