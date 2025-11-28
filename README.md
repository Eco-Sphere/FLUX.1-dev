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
- device_type: device类型，有A2-32g、A2-64g两个选项
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
       --sequence_parallel
```
参数说明：
- ASCEND_RT_VISIBLE_DEVICES: shell环境变量，用以绑定推理时实际使用的NPU
- mast_port: master节点端口号，torch_run命令变量设置
- nproc_per_node: 分布式推理使用的NPU数量，设置为2
- path: Flux本地模型权重路径，默认读取当前文件夹下的flux文件夹
- save_path: 保存图像路径，默认当前文件夹下的res文件夹
- device: 推理设备类型，默认为npu
- prompt_path: 用于图像生成的文字描述提示的列表文件路径
- width: 图像生成的宽度，默认1024
- height: 图像生成的高度，默认1024
- infer_steps: Flux图像推理步数，默认值为50
- seed: 设置随机种子，默认值为42
- use_cache: 是否开启dit cache近似优化
- device_type: device类型，有A2-32g、A2-64g两个选项
- batch_size: 指定prompt的batch size，默认为1，大于1时以list形式送入pipeline
- sequence_parallel: 指定开启双芯SP并行

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
       --device_type "A2-32g"
```
参数说明参照Atlas-800I-A2-64g参数说明

### 4.4 Atlas-800I-A2-32g双卡TP并行推理性能测试
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
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2002 --nproc_per_node=2 inference_flux.py \
       --path ${model_path} \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --use_cache \
       --tensor_parallel

```
参数说明：
- tensor_parallel: 指定开启双芯TP并行
其余参数说明参照Atlas-800I-A2-64g参数说明


### 4.5 Atlas-800I-A2-32g单卡w8a16量化推理性能测试
#### 4.5.1 安装量化工具msModelSlim
参考[官方README](https://gitee.com/ascend/msit/tree/master/msmodelslim)
1.git clone下载msit仓代码
2.进入到msit/msmodelslim的目录 cd msit/msmodelslim；并在进入的msmodelslim目录下，运行安装脚本 bash install.sh
#### 4.5.2 w8a16量化
执行命令：
```shell
# 指定量化类型
export quant_type="w8a16"

python quant.py \
       --path ${model_path} \
       --device_id 7 \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --device_type "A2-32g" \
       --quant_type ${quant_type}
```
参数说明：
- quant_type: 量化类型，有w8a16、w8a8_dynamic两个选项
其余参数说明参照Atlas-800I-A2-64g参数说明
备注：量化成功后，会在模型权重目录生成'quant_weights_w8a16'一个文件夹，文件夹下包含两个文件'quant_model_description_w8a16.json'和'quant_model_weight_w8a16.safetensors'

#### 4.5.3 安装量化模型推理工具NNAL神经网络加速库和torch_atb
1. 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

2. 安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x Ascend-cann-nnal_<version>_linux-<arch>.run
# 默认路径安装:
./Ascend-cann-nnal_<version>_linux-<arch>.run --install --torch_atb
# 配置环境变量:
source ${HOME}/Ascend/nnal/atb/set_env.sh
```

#### 4.5.4 Atlas-800I-A2-32g单卡w8a16量化推理性能测试
执行命令：
```shell
# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2
# 指定量化类型
export quant_type="w8a16"

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
       --device_type "A2-32g" \
       --use_quant \
       --quant_type ${quant_type}

```
参数说明：
- use_quant: 指定使用量化模型
- quant_type: 量化类型，有w8a16、w8a8_dynamic两个选项
其余参数说明参照Atlas-800I-A2-64g参数说明

### 4.6 Atlas-800I-A2-32g单卡w8a8_dynamic量化推理性能测试
#### 4.6.1 安装量化工具msModelSlim
参照Atlas-800I-A2-32g单卡w8a16安装量化工具msModelSlim说明

#### 4.6.2 w8a8_dynamic量化
执行命令：
```shell
export quant_type="w8a8_dynamic"
python quant.py \
       --path ${model_path} \
       --device_id 7 \
       --prompt_path "./prompts.txt" \
       --width 1024 \
       --height 1024 \
       --infer_steps 50 \
       --seed 42 \
       --device_type "A2-32g" \
       --use_calib_data \
       --quant_type ${quant_type}
```
参数说明：
- quant_type: 量化类型，有w8a16、w8a8_dynamic两个选项
其余参数说明参照Atlas-800I-A2-64g参数说明
备注：量化成功后，会在模型权重目录生成'quant_weights_w8a8_dynamic'一个文件夹，文件夹下包含两个文件'quant_model_description_w8a8_dynamic.json'和'quant_model_weight_w8a8_dynamic.safetensors'

#### 4.6.3 安装量化模型推理工具NNAL神经网络加速库和torch_atb
参照Atlas-800I-A2-32g单卡w8a16安装量化模型推理工具NNAL神经网络加速库和torch_atb说明

#### 4.6.4 Atlas-800I-A2-32g单卡w8a8_dynamic量化推理性能测试
执行命令：
```shell
# 在环境中导入以下环境变量提高推理性能
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2
# 指定量化类型
export quant_type="w8a8_dynamic"

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
       --device_type "A2-32g" \
       --use_quant \
       --quant_type ${quant_type}

```
参数说明：
- use_quant: 指定使用量化模型
- quant_type: 量化类型，有w8a16、w8a8_dynamic两个选项
其余参数说明参照Atlas-800I-A2-64g参数说明

### 4.7 精度测试
#### 4.7.1 ClipScore测试
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

#### 4.7.2 Hpsv2精度测试
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
```
3.执行推理脚本计算hpsv2
```shell
import logging
import os
import sys
import glob
import shutil
import argparse
import platform
import subprocess
import pkg_resources
from typing import Optional
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

os.environ["SOURCE_DATE_EPOCH"] = "0"
MINDIE_SD_VERSION_DEFAULT = "2.3.0"
VERSION_ENV = "MINDIE_SD_VERSION_OVERRIDE"


def get_mindiesd_version():
    mindiesd_version = ""
    version = os.environ.get(VERSION_ENV, None)
    if version:
        print(f"MINDIE_SD_VERSION_OVERRIDE is: {version}")
        mindiesd_version = mindiesd_version + version
    else:
        print(f"MINDIE_SD_VERSION_DEFAULT is: {MINDIE_SD_VERSION_DEFAULT}")
        mindiesd_version = mindiesd_version + MINDIE_SD_VERSION_DEFAULT

    return mindiesd_version

def copy_so_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    so_files = [f for f in os.listdir(src_dir) if f.endswith('.so')]
    if not so_files:
        logging.warning(f"No .so files found in {src_dir}")
        return
    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        subprocess.check_call(['/bin/cp', src_file, dest_file])
        logging.info(f"Copied {src_file} to {dest_file}")


class CustomBuildPy(_build_py):
    def run(self):
        # 1. 进入 build 目录执行构建脚本
        print(">>> Running build.sh to compile shared libraries...")
        build_script_path = os.path.join(os.path.abspath(os.getcwd()), 'build')
        subprocess.check_call(['bash', './build.sh'], cwd=build_script_path)
        # 2. 把生成的 .so 文件移动到 Python 包目录 mindiesd/plugin
        source_dir = os.path.join(build_script_path, 'build')
        destination_dir = os.path.join(os.path.abspath(os.getcwd()), 'mindiesd', 'plugin')
        copy_so_files(source_dir, destination_dir)
        
        # 3. 继续默认 build_py 流程
        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # 标记为二进制 wheel，否则会生成 py3-none-any
        self.root_is_pure = False


if __name__ == "__main__":
    requirements = ["torch", "torch_npu"]
    mindie_sd_version = get_mindiesd_version()

    setup(
        name="mindie_sd",
        version=mindie_sd_version,
        author="ascend",
        description="build wheel for mindie sd",
        setup_requires=[],
        install_requires=requirements,
        zip_safe=False,
        python_requires=">=3.10",
        include_package_data=True,
        packages=find_packages(),
        package_data={
            "": [
                "*.so",  
                "ops/**/*"
            ]
        },
        cmdclass={
            "build_py": CustomBuildPy,
            "bdist_wheel": BDistWheel
        }
    )

```