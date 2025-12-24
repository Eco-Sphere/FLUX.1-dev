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

import torch
import torch_npu

from torch_npu.contrib import transfer_to_npu

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import M4ProcessorConfig, W8A8DynamicQuantConfig, \
    W8A8DynamicProcessorConfig, SaveProcessorConfig

from FLUX1dev import FluxPipeline
from FLUX1dev.utils import check_prompts_valid, check_param_valid, check_dir_safety, check_file_safety
from FLUX1dev.quant.dump_utils import InputCapture, DumperManager, get_disable_layer_names, to_device

from prompt_loader import PromptLoader
from FLUX1dev.quant.flux_adapter import FluxAdapter
from transformers import T5EncoderModel
from mindiesd import CacheAgent, CacheConfig

torch_npu.npu.set_compile_mode(jit_compile=False)

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./flux", help="Path to the flux model directory")
    parser.add_argument("--calib_dataset_path", type=str, default="./calib_dataset", help="Path to the flux model directory")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--data_type", choices=["bfloat16", "float16", "float32"], default="bfloat16", help="specify infer prompt type")
    parser.add_argument("--prompt_path", type=str, default="./prompts.txt", help="input prompt text path")
    parser.add_argument("--prompt_type", choices=["plain", "parti", "hpsv2"], default="plain", help="specify infer prompt type")
    parser.add_argument("--width", type=int, default=1024, help='Image size width')
    parser.add_argument("--height", type=int, default=1024, help='Image size height')  
    parser.add_argument("--infer_steps", type=int, default=50, help="Inference steps") 
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--use_calib_data", action="store_true", help="use calib data or not")
    parser.add_argument("--calib_data_nums", type=int, default=5, help="specify calib data nums for quant")
    parser.add_argument("--quant_type", choices=["w8a16", "w8a8_dynamic"], default="w8a8_dynamic", help="specify quant type")
    # ======================== Cpu offload config ========================
    parser.add_argument("--cpu_offload", action="store_true", help="when use 32g device, turn on cpu offload.")
    return parser.parse_args()


def get_prompts(args):
    check_file_safety(args.prompt_path)
    prompt_loader = PromptLoader(args.prompt_path,
                                args.prompt_type,
                                batch_size=1,
                                num_images_per_prompt=1,
                                max_num_prompts=0)
    return prompt_loader


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)


def initialize_pipeline(args):
    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")

    check_dir_safety(args.path)
    T5_model_path = os.path.join(args.path, "text_encoder_2")
    T5_model = T5EncoderModel.from_pretrained(T5_model_path).to(torch.bfloat16)

    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16, local_files_only=True)
        
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

    if not args.cpu_offload:
        pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()

    return pipe


def get_calib_dataset(args, pipe, model):
    calib_dataset_path = args.calib_dataset_path
    if not os.path.exists(calib_dataset_path):
        os.makedirs(calib_dataset_path, mode=0o640)
    check_dir_safety(calib_dataset_path)

    dumper_manager = DumperManager(model, capture_mode="args")
    
    prompt_loader = get_prompts(args)
    check_param_valid(args.height, args.width, args.infer_steps)
    for infer_num, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']

        check_prompts_valid(prompts)
        print(prompts)
        if infer_num < args.calib_data_nums:
            image = pipe(
                prompts,
                height=args.width,
                width=args.height,
                guidance_scale=3.5,
                num_inference_steps=args.infer_steps,
                max_sequence_length=512,
                use_cache=False,
            )
    calib_dataset = InputCapture.get_all()
    dumper_manager.save(os.path.join(calib_dataset_path, f'dit_input_data.pth'))     
    return calib_dataset

def quant(args):
    set_seed(args)
    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    pipe = initialize_pipeline(args)
    data_type = dtype_map[args.data_type]

    save_path = os.path.join(args.path, f"quant_weights_{args.quant_type}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o640)
    check_dir_safety(save_path)
    model = pipe.transformer

    if args.use_calib_data:
        calib_dataset = get_calib_dataset(args, pipe, model)
        calib_dataset = to_device(calib_dataset, device, depth=0)
    else:
        calib_dataset = None

    if args.quant_type == "w8a16":
        disable_names = []
        quant_config = QuantConfig(
            a_bit=16,
            w_bit=8,
            disable_names=disable_names,
            dev_type='npu',
            dev_id=args.device_id,
            w_method='MinMax',
            pr=1.0,
            w_sym=True,
            mm_tensor=False
        )
        calibrator = Calibrator(model, quant_config, calib_data=None)
        # 执行PTQ量化校准
        calibrator.run()
        # save_path路径下会生成quant_model_weight_w8a16.safetensors、quant_model_description_w8a16.json
        calibrator.save(
            output_path=save_path,
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None)

    else:
        session_cfg = SessionConfig(
            processor_cfg_map={
            "m4": M4ProcessorConfig(),
            "w8a8_dynamic": W8A8DynamicProcessorConfig(
                cfg=W8A8DynamicQuantConfig(
                    act_method='minmax'
                ),
                disable_names=get_disable_layer_names(
                    model,
                    layer_include='*',
                    layer_exclude='*net.2*',
                ),

            ),
            "save": SaveProcessorConfig(
                output_path=save_path,
                safetensors_name=None,
                json_name=None,
                save_type=['safe_tensor'],
                part_file_size=None
            )
        },
        calib_data=calib_dataset,
        device='npu',
        dev_id=args.device_id,
        )
        # pydantic库自带的数据类型校验
        session_cfg.model_validate(session_cfg)

        # 量化配置
        if not hasattr(model, 'config'):
            from types import SimpleNamespace
            model.config = SimpleNamespace()
        model.config.torch_dtype = data_type
        model.config.model_type = "flux"
        model.config.num_layers = len(model.transformer_blocks)
        model.config.single_num_layers = len(model.single_transformer_blocks)
        # 执行PTQ量化校准
        # save_path路径下会生成quant_model_description_w8a8_dynamic.json、quant_model_weight_w8a8_dynamic.safetensors
        quant_model(model, session_cfg)

    return

if __name__ == "__main__":
    inference_args = parse_arguments()
    quant(inference_args)