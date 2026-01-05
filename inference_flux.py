#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import argparse
import csv
import json
import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import T5EncoderModel

from mindiesd import CacheAgent, CacheConfig

from FLUX1dev import BlockOffloadHookV2
from FLUX1dev import FluxPipeline, parallelize_transformer
from FLUX1dev import get_local_rank, get_world_size, initialize_torch_distributed
from FLUX1dev.utils import check_prompts_valid, check_param_valid, check_dir_safety, check_file_safety

torch_npu.npu.set_compile_mode(jit_compile=False)
if bool(os.environ.get("USE_NZ", 0)):
    torch.npu.config.allow_internal_format=True
else:
    torch.npu.config.allow_internal_format=False


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./flux", help="Path to the flux model directory")
    parser.add_argument("--save_path", type=str, default="./res", help="ouput image path")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu", help="NPU")
    parser.add_argument("--prompt_path", type=str, default="./prompts.txt", help="input prompt text path")
    parser.add_argument("--prompt_type", choices=["plain", "parti", "hpsv2"], default="plain", help="specify infer prompt type")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="specify image number every prompt generate")
    parser.add_argument("--max_num_prompt", type=int, default=0, help="limit the prompt number[0 indicates no limit]")
    parser.add_argument("--info_file_save_path", type=str, default="./image_info.json", help="path to save image info")
    parser.add_argument("--width", type=int, default=1024, help='Image size width')
    parser.add_argument("--height", type=int, default=1024, help='Image size height')  
    parser.add_argument("--infer_steps", type=int, default=50, help="Inference steps") 
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--use_cache", action="store_true", help="turn on dit cache or not")
    parser.add_argument("--batch_size", type=int, default=1, help="prompt batch size")
    # ======================== Cpu offload config ========================
    parser.add_argument("--cpu_offload", action="store_true", help="when use 32g device, turn on cpu offload.")
    # ======================== Parallel config ========================
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tensor_parallel",
        action="store_true",
        help="turn on tensor parallel or not."
    )
    group.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="turn on sequence parallel or not."
    )
    # ======================== Quant config ========================
    parser.add_argument("--use_quant", action="store_true", help="turn on quant or not")
    parser.add_argument("--quant_type", choices=["w8a16", "w8a8_dynamic"], default="w8a8_dynamic", help="specify quant type")
    # ======================== Test config ========================
    parser.add_argument("--prompt", type=str, default="Beautiful illustration of The ocean. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper", help="default prompt")
    return parser.parse_args()


def _transpose_to_nz(model):
    torch.npu.config.allow_internal_format=True
    if not hasattr(model, "named_modules"):
        return
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.data.device.type == "cpu":
                module.weight.data = module.weight.data.to("npu")
            try:
                weight = torch_npu.npu_format_cast(module.weight.data.contiguous(), 29)
                module.weight.data = weight
            except Exception as e:
                logger.warning(f"Failed to transpose {name} to NZ, skipping: {e}")


def transfer_nd_to_nz(pipe):
    for attr in dir(pipe):
        if attr.startswith("_") or not hasattr(pipe, attr):
            continue
        if hasattr(getattr(pipe, attr), "named_modules"):
            _transpose_to_nz(getattr(pipe, attr))


def init_cv_parallel(cp_level):
    """
    Initialize computer vision parallel processing based on the specified level.
    
    Args:
        cp_level (int): The level of parallel processing to enable.
                        - 0: No parallel processing
                        - 1: Enable double stream
                        - 2: Enable double stream and attention double stream
    """
    if cp_level == 1:
        from FLUX1dev.models import init_double_stream
        init_double_stream()
        print("CV parallel level 1 enabled")
    elif cp_level == 2:
        from FLUX1dev.models import init_double_stream
        from FLUX1dev.layers import init_attn_double_stream
        init_double_stream()
        init_attn_double_stream()
        print("CV parallel level 2 enabled")
    elif cp_level == 0:
        print("CV parallel disabled")
    else:
        print(f"Invalid CV parallel level '{cp_level}'. Valid levels are 0, 1, or 2. No parallel processing enabled")


def initialize_pipeline(args):
    if bool(int(os.environ.get("FAST_GELU", 0))):
        from FLUX1dev.layers import enable_fast_gelu
        enable_fast_gelu()

    if args.tensor_parallel or args.sequence_parallel:
        local_rank = get_local_rank()
        world_size = get_world_size()
        if args.tensor_parallel and world_size != 2:
            raise ValueError(f"When enable tensor parallel, number of NPUs should be equal to 2.")
        initialize_torch_distributed(local_rank, world_size)
        device = torch.device(f"npu:{local_rank}")
    else:
        torch.npu.set_device(args.device_id)
        device = torch.device(f"npu:{args.device_id}")

    check_dir_safety(args.path)
    T5_model_path = os.path.join(args.path, "text_encoder_2")
    T5_model = T5EncoderModel.from_pretrained(T5_model_path).to(torch.bfloat16)

    if args.tensor_parallel:
        from FLUX1dev import replace_tp_from_pretrain, replace_tp_extract_init_dict
        FluxPipeline.from_pretrained = classmethod(replace_tp_from_pretrain)
        FluxPipeline.extract_init_dict = classmethod(replace_tp_extract_init_dict)

    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16, local_files_only=True)
    
    if args.sequence_parallel:
        pipe.transformer.pos_embed.enable_seq_parallel()
        
    if args.use_cache:
        d_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=19,
            steps_count=args.infer_steps,
            step_start=18,
            step_interval=2,
            block_start=5,
            block_end=13,
        )
        d_stream_agent = CacheAgent(d_stream_config)
        pipe.transformer.d_stream_agent = d_stream_agent
        s_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=38,
            steps_count=args.infer_steps,
            step_start=18,
            step_interval=2,
            block_start=1,
            block_end=23,
        )
        s_stream_agent = CacheAgent(s_stream_config)
        pipe.transformer.s_stream_agent = s_stream_agent
    else:
        d_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=19,
            steps_count=args.infer_steps,
            step_start=args.infer_steps,
            step_interval=2,
            block_start=18,
            block_end=18,
        )
        d_stream_agent = CacheAgent(d_stream_config)
        pipe.transformer.d_stream_agent = d_stream_agent
        s_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=38,
            steps_count=args.infer_steps,
            step_start=args.infer_steps,
            step_interval=2,
            block_start=37,
            block_end=37,
        )
        s_stream_agent = CacheAgent(s_stream_config)
        pipe.transformer.s_stream_agent = s_stream_agent

    if args.tensor_parallel:
        import deepspeed
        T5_model = deepspeed.init_inference(
            T5_model,
            tensor_parallel={"tp_size": world_size},
        )
        T5_model.module.to("cpu")
        pipe.to(f"npu:{local_rank}")
        pipe.text_encoder_2.to("cpu")
        pipe.text_encoder_2 = T5_model.module.to(f"npu:{local_rank}")
    else:
        if args.sequence_parallel:
            parallel_args = {
                "ulysses":{
                    "world_size": world_size,
                    "rank": local_rank,
                    "group": None
                }
            }
            pipe = parallelize_transformer(pipe, parallel_args)

        if args.use_quant:
            from mindiesd import quantize
            quant_config_path = os.path.join(args.path, f"quant_weights_{args.quant_type}/quant_model_description_{args.quant_type}.json")
            pipe.transformer = quantize(pipe.transformer, quant_config_path, timestep_config=None, dtype=torch.bfloat16)
            pipe.to(device)
        else:
            if not args.cpu_offload:
                pipe.to(device)
            else:
                original_transformer = pipe.transformer
                pipe.transformer = None
                pipe.to(device)
                pipe.transformer = original_transformer

                transformer_block_hook = BlockOffloadHookV2(
                    pipe.transformer.transformer_blocks,
                    onload_device=device,
                    block_on_npu_nums=2,
                    cache_config=d_stream_config
                )
                transformer_block_hook.register_hook()

                single_transformer_block_hook = BlockOffloadHookV2(
                    pipe.transformer.single_transformer_blocks,
                    onload_device=device,
                    block_on_npu_nums=2,
                    cache_config=s_stream_config
                )
                single_transformer_block_hook.register_hook()

                for name, module in pipe.transformer.named_children():
                    if name not in ["transformer_blocks", "single_transformer_blocks"]:
                        module.to(device) 

    if bool(os.environ.get("USE_NZ", 0)):
        transfer_nd_to_nz(pipe)

    cp_level = int(os.environ.get("CV_PARALLEL_LEVEL", 0))
    init_cv_parallel(cp_level)
    return pipe

def set_seed(seed):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
def infer(args):
    set_seed(args.seed)

    pipe = initialize_pipeline(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, mode=0o640)
    check_dir_safety(args.save_path)
    check_param_valid(args.height, args.width, args.infer_steps)
    
    total_nums = 6
    warmup_nums = 3
    time_consume = 0.0
    
    for i in range(total_nums):
        prompts = [args.prompt]
        torch.npu.synchronize()
        start_time = time.time()
        image = pipe(
            prompts,
            height=args.width,
            width=args.height,
            guidance_scale=3.5,
            num_inference_steps=args.infer_steps,
            max_sequence_length=512,
            use_cache=args.use_cache,
        )        
        torch.npu.synchronize()
        end_time = time.time() - start_time
        print(f"The inference time of the {i} image is: {end_time}")

        if i > (warmup_nums - 1):
            time_consume += end_time

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                image_save_path = os.path.join(args.save_path, f"{i}.png")
                image[0][0].save(image_save_path)
        else:
            image_save_path = os.path.join(args.save_path, f"{i}.png")
            image[0][0].save(image_save_path)
        
    print(f"Average inference time is: {time_consume / (total_nums - warmup_nums)}")

    return


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)