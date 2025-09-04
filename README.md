---
pipeline_tag: text-to-image
frameworks:
  - PyTorch
library_name: openmind
language:
  - en
hardwares:
  - NPU
license: apache-2.0
---
# 模型推理指导  

## 一、模型简介

Flux.1-DEV是一种文本到图像的扩散模型，能够在给定文本输入的情况下生成相符的图像。

本模型使用的优化手段如下：
等价优化：FA、ROPE、RMSnorm、TP并行（32G机器可选）
算法优化：FA、ROPE、RMSnorm、DiTCache、TP并行（32G机器可选）

## 二、环境准备

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- 支持卡数：支持的卡数为1或2
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

### 2.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 2.4 Torch_npu安装
安装pytorch框架 版本2.1.0
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 2.5 安装gcc、g++
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```

### 2.6 下载本仓库
```shell
   git clone https://modelers.cn/MindIE/FLUX.1-dev.git
```

### 2.7 安装所需依赖
```shell
pip install -r requirements.txt
```

## 三、模型权重

### 3.1 权重下载
Flux.1-DEV权重下载地址
```shell
https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main
```

### 3.2 配置文件说明
修改权重配置文件：
```bash
vi ${model_path}/model_index.json
````
做如下修改：
```json
{
  "_class_name": "FluxPipeline",
  "_diffusers_version": "0.30.0.dev0",
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "text_encoder_2": [
    "transformers",
    "T5EncoderModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "tokenizer_2": [
    "transformers",
    "T5TokenizerFast"
  ],
  "transformer": [
    "FLUX1dev",
    "FluxTransformer2DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

## 四、模型推理

### 4.1 Atlas-800I-A2-64g单卡推理性能测试
1. 设置权重路径：
```bash
export model_path="your local flux model path"
```

2. 执行命令：
```shell
# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2

python inference_flux.py \
       --path ${model_path} \
       --save_path "./res" \
       --device_id 0 \
       --device "npu" \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --device_type "A2-64g" \
       --batch_size 1
```
参数说明：
- path: Flux本地模型权重路径，默认读取当前文件夹下的flux文件夹
- save_path: 保存图像路径，默认当前文件夹下的res文件夹
- device_id: 推理设备ID，默认值设置为0
- device: 推理设备类型，默认为npu
- prompt_path: 用于图像生成的文字描述提示的列表文件路径
- width: 图像生成的宽度，默认1024
- height: 图像生成的高度，默认1024
- infer_steps: Flux图像推理步数，默认值为50
- seed: 设置随机种子，默认值为42
- use_cache: 是否开启dit cache近似优化
- device_type: device类型，有A2-32g-single、A2-32g-dual、A2-64g三个选项
- batch_size: 指定prompt的batch size，默认为1，大于1时以list形式送入pipeline

### 4.1 Atlas-800I-A2-64g双卡推理性能测试
1. 设置权重路径：
```bash
export model_path="your local flux model path"
```

2. 执行命令：
```shell
# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2

ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=20095 --nproc_per_node=2 inference_flux.py \
       --path ${model_path} \
       --save_path "./res" \
       --device "npu" \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --device_type "A2-64g" \
       --batch_size 1 \
       --ulysses-degree 2
```
参数说明：
- path: Flux本地模型权重路径，默认读取当前文件夹下的flux文件夹
- save_path: 保存图像路径，默认当前文件夹下的res文件夹
- device: 推理设备类型，默认为npu
- prompt_path: 用于图像生成的文字描述提示的列表文件路径
- width: 图像生成的宽度，默认1024
- height: 图像生成的高度，默认1024
- infer_steps: Flux图像推理步数，默认值为50
- seed: 设置随机种子，默认值为42
- use_cache: 是否开启dit cache近似优化
- device_type: device类型，有A2-32g-single、A2-32g-dual、A2-64g三个选项
- batch_size: 指定prompt的batch size，默认为1，大于1时以list形式送入pipeline
- ulysses-degree: 指定ulysses并行度

### 4.3 Atlas-800I-A2-32g单卡推理性能测试
1. 设置权重路径：
```bash
export model_path="your local flux model path"
```

2. 执行命令：
```shell
# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2

python inference_flux.py \
       --path ${model_path} \
       --save_path "./res" \
       --device_id 0 \
       --device "npu" \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --device_type "A2-32g-single"
```
参数说明参照Atlas-800I-A2-64g参数说明

### 4.4 Atlas-800I-A2-32g双卡推理性能测试
1. 设置权重路径：
```bash
export model_path="your local flux model path"
```

2.执行命令进行权重切分
```shell
python3 tpsplit_weight.py --path ${model_path}
```
备注：权重切分成功后，会在模型权重目录生成'transformer_0'与'transformer_1'两个文件夹，两个文件夹下内容与初始transformer文件夹文件相同，但大小不同，执行du -sh，大小应为15G

3.修改transformer_0与transformer_1下的config文件，添加is_tp变量：
```json
{
  "_class_name": "FluxTransformer2DModel",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "../checkpoints/flux-dev/transformer",
  "attention_head_dim": 128,
  "guidance_embeds": true,
  "in_channels": 64,
  "joint_attention_dim": 4096,
  "num_attention_heads": 24,
  "num_layers": 19,
  "num_single_layers": 38,
  "patch_size": 1,
  "pooled_projection_dim": 768,
  "is_tp": true
}
```

4. 执行命令：
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py --device_type "A2-32g-dual" --path ${model_path} --prompt_path "./prompts.txt" --width 1024 --height 1024 --infer_steps 50 --seed 42 --use_cache
```
参数说明：
- ASCEND_RT_VISIBLE_DEVICES: shell环境变量，用以绑定推理时实际使用的NPU
- mast_port:master节点端口号，torch_run命令变量设置
- nproc_per_node:分布式推理使用的NPU数量，设置为2
其余参数说明参照Atlas-800I-A2-64g参数说明

### 4.5 精度测试
#### 4.5.1 ClipScore测试
1.准备模型与数据集
```shell
# 下载Parti数据集
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate

# 下载clip模型
# 安装git-lfs
apt install git-lfs
git lfs install

git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```
也可手动下载[clip模型](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)权重

2.推理Parti数据集，生成图像
```shell
# 单卡64G Flux 等价优化推理
python inference_flux.py \
       --path ${model_path} \
       --save_path "./clipscore_res_wocache" \
       --device_id 0 \
       --device "npu" \
       --prompt_path "./PartiPrompts.tsv" \
       --prompt_type "parti" \
       --num_images_per_prompt 4 \
       --info_file_save_path "./clip_info_wocache.json" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --device_type "A2-64g"
# 单卡64G Flux 近似优化推理
python inference_flux.py \
       --path ${model_path} \
       --save_path "./clipscore_res_wcache" \
       --device_id 0 \
       --device "npu" \
       --prompt_path "./PartiPrompts.tsv" \
       --prompt_type "parti" \
       --num_images_per_prompt 4 \
       --info_file_save_path "./clip_info_wcache.json" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --device_type "A2-64g"
# 双卡32G Flux等价优化推理
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py --device_type "A2-32g-dual" --path ${model_path} --prompt_path "./PartiPrompts.tsv" --prompt_type "parti" --num_images_per_prompt 4 --info_file_save_path "./clip_info_wocache.json" --width 1024 --height 1024 --infer_steps 50 --seed 42
# 双卡32G Flux近似优化推理
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py --device_type "A2-32g-dual" --path ${model_path} --prompt_path "./PartiPrompts.tsv" --prompt_type "parti" --num_images_per_prompt 4 --info_file_save_path "./clip_info_wcache.json" --width 1024 --height 1024 --infer_steps 50 --seed 42 --use_cache
```
3.执行推理脚本计算clipscore
```shell
# 等价优化
python clipscore.py \
       --device="cpu" \
       --image_info="clip_info_wocache.json" \
       --model_name="ViT-H-14" \
       --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
# 近似优化
python clipscore.py \
       --device="cpu" \
       --image_info="clip_info_wcache.json" \
       --model_name="ViT-H-14" \
       --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```
参数说明
- device: 推理设备，默认使用cpu做计算。
- image_info: 上一步生成的json文件。
- model_name: Clip模型名称。
- model_weights_path: Clip模型权重文件路径。

#### 4.5.2 Hpsv2精度测试
1.准备模型与数据集

[hpsv2数据集获取](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json)
```shell
# 下载权重
wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
```
2.执行hpsv2数据集，生成图像
```shell
#单卡64G Flux等价优化推理
python inference_flux.py \
       --path ${model_path} \
       --save_path "./hpsv2_res_wocache" \
       --device_id 0 \
       --device "npu" \
       --prompt_type "hpsv2" \
       --num_images_per_prompt 1 \
       --info_file_save_path "./hpsv2_info_wocache.json" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --device_type "A2-64g"
#单卡64G Flux近似优化推理
python inference_flux.py \
       --path ${model_path} \
       --save_path "./hpsv2_res_wcache" \
       --device_id 0 \
       --device "npu" \
       --prompt_type "hpsv2" \
       --num_images_per_prompt 1 \
       --info_file_save_path "./hpsv2_info_wcache.json" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --device_type "A2-64g"
# 双卡32G Flux等价优化推理
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py --device_type "A2-32g-dual" --path ${model_path} --prompt_type "hpsv2" --num_images_per_prompt 1 --info_file_save_path "./hpsv2_info_wocache.json" --width 1024 --height 1024 --infer_steps 50 --seed 42
# 双卡32G Flux近似优化推理
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py --device_type "A2-32g-dual" --path ${model_path} --prompt_type "hpsv2" --num_images_per_prompt 1 --info_file_save_path "./hpsv2_info_wocache.json" --width 1024 --height 1024 --infer_steps 50 --seed 42 --use_cache
```
3.执行推理脚本计算hpsv2
```shell
python hpsv2_score.py \
       --image_info="hpsv2_info_wocache.json" \
       --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
       --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```
- image_info: 上一步生成的json文件。
- HPSv2_checkpoint: HPSv2模型权重文件路径。
- clip_checkpointh: Clip模型权重文件路径。

## 五、推理结果参考
### Flux.1-DEV性能数据
| 硬件形态  | cpu规格 | batch size | 分辨率 |迭代次数 | 优化手段 | 性能 | 采样器 | 备注 |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Atlas 800I A2(8*64G) | 64核(arm) |  1  | 1024*1024 |  50  | with DiTCache |  20.4s   | FlowMatchEuler | 单卡运行 |
| Atlas 800I A2(8*32G) | 64核(arm) |  1  | 1024*1024 |  50  | with DiTCache |  24.6s   | FlowMatchEuler | 双卡运行 |

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。
```python
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
import os
import argparse
import csv
import json
import time

import torch
import torch_npu

from torch_npu.contrib import transfer_to_npu

from mindiesd import CacheAgent, CacheConfig
from FLUX1dev import BlockOffloadHook
from FLUX1dev import FluxPipeline, parallelize_transformer
from FLUX1dev import get_local_rank, get_world_size, initialize_torch_distributed
from FLUX1dev.utils import check_prompts_valid, check_param_valid, check_dir_safety, check_file_safety
from transformers import T5EncoderModel

torch_npu.npu.set_compile_mode(jit_compile=False)


class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int = 1,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.check_input_isvalid(batch_size, num_images_per_prompt, max_num_prompts)
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(max_num_prompts)
        else:
            print("This operation is not supported!")

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration
        
        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret
    
    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, max_num_prompts: int):
        with open('hpsv2_benchmark_prompts.json', 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))

    def check_input_isvalid(self, batch_size, num_images_per_prompt, max_num_prompts):
        if batch_size <= 0:
            raise ValueError(f"Param batch_size invalid, expected positive value, but get {batch_size}")
        if num_images_per_prompt <= 0:
            raise ValueError(f"Param num_images_per_prompt invalid, expected positive value, but get {num_images_per_prompt}")
        if max_num_prompts < 0:
            raise ValueError(f"Param max_num_prompts invalid, expected greater than or equal to 0, \
                                 but get {max_num_prompts}")


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
    parser.add_argument("--device_type", choices=["A2-32g", "A2-64g"], default="A2-64g", help="specify device type")
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
    # parser.add_argument(
    #     "--tensor_parallel",
    #     action="store_true",
    #     help="turn on tensor parallel or not.",
    # )
    # parser.add_argument(
    #     "--sequence_parallel",
    #     action="store_true",
    #     help="turn on sequence parallel or not.",
    # )
    # ======================== Quant config ========================
    parser.add_argument("--use_quant", action="store_true", help="turn on quant or not")
    parser.add_argument("--quant_type", choices=["w8a16", "w8a8_dynamic"], default="w8a8_dynamic", help="specify quant type")
    return parser.parse_args()


def initialize_pipeline(args):
    if args.tensor_parallel or args.sequence_parallel:
        local_rank = get_local_rank()
        world_size = get_world_size()
        if world_size != 2:
            raise ValueError(f"number of NPUs should be equal to 2.")
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

        if args.device_type == "A2-64g":
            pipe.to(device)
        else:
            if args.use_quant:
                from mindiesd import quantize
                quant_config_path = os.path.join(args.path, f"quant_weights_{args.quant_type}/quant_model_description_{args.quant_type}.json")
                pipe.transformer = quantize(pipe.transformer, quant_config_path, timestep_config=None, dtype=torch.bfloat16)
                pipe.to(device)
            else:
                # pipe.enable_model_cpu_offload()
                original_transformer = pipe.transformer
                pipe.transformer = None
                pipe.to(device)
                pipe.transformer = original_transformer

                transformer_block_hook = BlockOffloadHook(
                    pipe.transformer.transformer_blocks,
                    onload_device=device,
                    block_on_npu_nums=2,
                    cache_config=d_stream_config
                )
                transformer_block_hook.register_hook()
                for name, module in pipe.transformer.named_children():
                    if name != "transformer_blocks":
                        module.to(device)
    return pipe

def infer(args):
    pipe = initialize_pipeline(args)

    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, mode=0o640)
    check_dir_safety(args.save_path)

    infer_num = 0
    time_consume = 0
    current_prompt = None
    image_info = []
    check_file_safety(args.prompt_path)
    prompt_loader = PromptLoader(args.prompt_path,
                                args.prompt_type,
                                args.batch_size,
                                args.num_images_per_prompt,
                                args.max_num_prompt)
    check_param_valid(args.height, args.width, args.infer_steps)
    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        check_prompts_valid(prompts)

        print(f"[{infer_num+n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size
        if infer_num > 3:
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

        if infer_num > 3:
            torch.npu.synchronize()
            end_time = time.time() - start_time
            time_consume += end_time

        for j in range(n_prompts):
            image_save_path = os.path.join(args.save_path, f"{save_names[j]}.png")
            image[0][j].save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)
    
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if os.path.exists(args.info_file_save_path):
                os.remove(args.info_file_save_path)

            with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
                json.dump(image_info, f)
    else:
        if os.path.exists(args.info_file_save_path):
            os.remove(args.info_file_save_path)

        with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
            json.dump(image_info, f)

    image_time_count = len(prompt_loader) - 3
    print(f"flux pipeline time is:{time_consume/image_time_count}")

    return


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)


```