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
```shell
#ifndef PYTORCH_NPU_HELPER_HPP_
#define PYTORCH_NPU_HELPER_HPP_

#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/aclnn_base.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <fstream>
#include <functional>
#include <type_traits>
#include <vector>
#include <string_view>
#include <utility>
#include <tuple>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

#define NPU_NAME_SPACE at_npu::native

using AclOpExecutor = struct aclOpExecutor;
using AclTensor = struct aclTensor;
using AclScalar = struct aclScalar;
using AclIntArray = struct aclIntArray;
using AclFloatArray = struct aclFloatArray;
using AclBoolArray = struct aclBoolArray;
using AclTensorList = struct aclTensorList;

template<typename T = void>
using FunctionPtr = T*;
constexpr int K_HASH_BUF_SIZE = 8192;
constexpr int K_HASH_BUF_MAX_SIZE = K_HASH_BUF_SIZE + 1024;
constexpr int64_t ACL_TENSOR_MAX_DIM_FOR_FORMAT = 5;
constexpr int64_t DIM_NUM_3D = 3;
constexpr int64_t DIM_NUM_4D = 4;
constexpr int64_t DIM_NUM_5D = 5;
extern thread_local char g_hashBuf[K_HASH_BUF_SIZE];
extern thread_local int g_hashOffset;

template <std::string_view const& ApiName>
constexpr auto GetWorkspaceSizeApiName()
{
    constexpr std::string_view suffix = "GetWorkspaceSize";
    std::array<char, ApiName.size() + suffix.size() + 1> buf{};
    size_t idx = 0;
    for (; idx < ApiName.size(); ++idx) {
        buf[idx] = ApiName[idx];
    }
    for (size_t j = 0; j < suffix.size(); ++j) {
        buf[idx + j] = suffix[j];
    }
    buf[idx + suffix.size()] = '\0';
    return buf;
}

inline std::vector<std::string> SplitStr(std::string s, const std::string &del)
{
    int end = s.find(del);
    std::vector<std::string> path_list;
    while (end != -1) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}

inline bool IsFileExist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

inline std::string RealPath(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPathBuf[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPathBuf) == nullptr) {
        return "";
    }
    return std::string(realPathBuf);
}
inline std::vector<std::string> ProcessPathList(const std::string& pathStr)
{
    return SplitStr(pathStr, ":");
}
inline void AppendLibPathSuffix(std::vector<std::string>& pathList)
{
    for (auto &it : pathList) {
        it = it + "/op_api/lib/";
    }
}
std::vector<std::string> ProcessCustomLibPath(const char* ascendCustomOppPath);
inline std::vector<std::string> ProcessCustomLibPath(const char* ascendCustomOppPath)
{
    std::string ascendCustomOppPathStr(ascendCustomOppPath);
    auto customLibPathList = ProcessPathList(ascendCustomOppPathStr);
    if (customLibPathList.empty()) {
        return std::vector<std::string>();
    }
    AppendLibPathSuffix(customLibPathList);
    return customLibPathList;
}
inline std::vector<std::string> GetCustomLibPath()
{
    const char *ascendCustomOppPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (ascendCustomOppPath == nullptr) {
        ASCEND_LOGW("ASCEND_CUSTOM_OPP_PATH is not exists");
        return std::vector<std::string>();
    }
    return ProcessCustomLibPath(ascendCustomOppPath);
}
std::vector<std::string> ProcessVendorsList(const std::string& vendorsPath, const std::string& line);
inline std::vector<std::string> ProcessVendorsList(const std::string& vendorsPath, const std::string& line)
{
    auto defaultVendorsList = SplitStr(line, ",");
    for (auto &it : defaultVendorsList) {
        it = RealPath(vendorsPath + "/" + it + "/op_api/lib/");
    }
    return defaultVendorsList;
}

inline std::string GetVendorsConfigFilePath(const std::string& vendorsPath)
{
    return RealPath(vendorsPath + "/config.ini");
}

inline bool ValidateVendorsConfigFile(const std::string& configFile)
{
    if (configFile.empty() || !IsFileExist(configFile)) {
        ASCEND_LOGW("config.ini is not exists or the path length is more than %d", PATH_MAX);
        return false;
    }
    return true;
}

inline std::string ReadLoadPriorityLine(const std::string& configFile)
{
    std::ifstream ifs(configFile);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    return line;
}

inline std::string ExtractLoadPriorityValue(const std::string& line)
{
    std::string head = "load_priority=";
    std::string result = line;
    if (result.find(head) == 0) {
        result.erase(0, head.length());
    }
    return result;
}
std::vector<std::string> ParseVendorsConfig(const std::string& vendorsPath);
inline std::vector<std::string> ParseVendorsConfig(const std::string& vendorsPath)
{
    std::string vendorsConfigFile = GetVendorsConfigFilePath(vendorsPath);
    if (!ValidateVendorsConfigFile(vendorsConfigFile)) {
        return {};
    }
    std::string line = ReadLoadPriorityLine(vendorsConfigFile);
    std::string priorityValue = ExtractLoadPriorityValue(line);
    return ProcessVendorsList(vendorsPath, priorityValue);
}
inline std::vector<std::string> GetDefaultCustomLibPath()
{
    const char *ascendOppPath = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> defaultVendorsList;
    if (ascendOppPath == nullptr) {
        ASCEND_LOGW("ASCEND_OPP_PATH is not exists");
        return std::vector<std::string>();
    }
    std::string vendorsPath(ascendOppPath);
    vendorsPath = vendorsPath + "/vendors";
    return ParseVendorsConfig(vendorsPath);
}

const std::vector<std::string> g_customLibPath = GetCustomLibPath();
const std::vector<std::string> g_defaultCustomLibPath = GetDefaultCustomLibPath();

constexpr aclDataType K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
    ACL_UINT8,
    ACL_INT8,
    ACL_INT16,
    ACL_INT32,
    ACL_INT64,
    ACL_FLOAT16,
    ACL_FLOAT,
    ACL_DOUBLE,
    ACL_DT_UNDEFINED,
    ACL_COMPLEX64,
    ACL_COMPLEX128,
    ACL_BOOL,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_BF16,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED
};

template<typename T>
inline bool CheckDataPointer(const T* data)
{
    if (data == nullptr) {
        TORCH_CHECK(false, "memcpy failed: source data is null pointer");
        return false;
    }
    return true;
}
inline bool CheckDataSize(size_t size)
{
    if (size == 0) {
        TORCH_CHECK(false, "memcpy failed: copy size is 0 (no data to copy)");
        return false;
    }
    return true;
}
inline bool CheckBufferSpace(size_t size)
{
    if (g_hashOffset + size > K_HASH_BUF_SIZE) {
        g_hashOffset = K_HASH_BUF_MAX_SIZE;
        TORCH_CHECK(false, "memcpy failed: buffer overflow");
        return false;
    }
    return true;
}
template<typename T>
inline bool ValidateMemcpyParams(const T* data, size_t size)
{
    return CheckDataPointer(data) && CheckDataSize(size) && CheckBufferSpace(size);
}

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

template<typename T = void>
inline void *GetOpApiFuncAddrInLib(T *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}
inline bool IsCustomLibPathEmpty()
{
    return g_customLibPath.empty();
}

inline std::string GetCustomOpApiLibPath(const std::string& libPath)
{
    return RealPath(libPath + "/" + GetCustOpApiLibName());
}

inline void* LoadCustomOpApiHandler(const std::string& custOpApiLib)
{
    if (custOpApiLib.empty()) {
        return nullptr;
    }
    return GetOpApiLibHandler(custOpApiLib.c_str());
}

inline void* FindFuncInCustomLibPath(const char* apiName, const std::string& libPath)
{
    auto custOpApiLib = GetCustomOpApiLibPath(libPath);
    auto custOpApiHandler = LoadCustomOpApiHandler(custOpApiLib);
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            ASCEND_LOGI("%s is found in %s.", apiName, custOpApiLib.c_str());
            return funcAddr;
        }
    }
    return nullptr;
}
inline bool ShouldSearchCustomLib()
{
    return !IsCustomLibPathEmpty();
}
inline void* SearchCustomLibPaths(const char* apiName)
{
    for (const auto &libPath : g_customLibPath) {
        void* funcAddr = FindFuncInCustomLibPath(apiName, libPath);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return nullptr;
}

inline void LogCustomLibNotFound(const char* apiName)
{
    ASCEND_LOGI("%s is not in custom lib.", apiName);
}
inline void* FindFuncInCustomLib(const char* apiName)
{
    if (!ShouldSearchCustomLib()) {
        return nullptr;
    }
    
    void* result = SearchCustomLibPaths(apiName);
    if (result == nullptr) {
        LogCustomLibNotFound(apiName);
    }
    return result;
}
inline bool IsDefaultCustomLibPathEmpty()
{
    return g_defaultCustomLibPath.empty();
}
inline std::string GetDefaultCustomOpApiLibPath(const std::string& libPath)
{
    return RealPath(libPath + "/" + GetCustOpApiLibName());
}
inline void* LoadDefaultCustomOpApiHandler(const std::string& defaultCustOpApiLib)
{
    if (defaultCustOpApiLib.empty()) {
        return nullptr;
    }
    return GetOpApiLibHandler(defaultCustOpApiLib.c_str());
}
inline void* FindFuncInDefaultLibPath(const char* apiName, const std::string& libPath)
{
    auto defaultCustOpApiLib = GetDefaultCustomOpApiLibPath(libPath);
    auto custOpApiHandler = LoadDefaultCustomOpApiHandler(defaultCustOpApiLib);
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            ASCEND_LOGI("%s is found in %s.", apiName, defaultCustOpApiLib.c_str());
            return funcAddr;
        }
    }
    return nullptr;
}
inline bool ShouldSearchDefaultLib()
{
    return !IsDefaultCustomLibPathEmpty();
}
inline void* SearchDefaultLibPaths(const char* apiName)
{
    for (const auto &libPath : g_defaultCustomLibPath) {
        void* funcAddr = FindFuncInDefaultLibPath(apiName, libPath);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return nullptr;
}
inline void LogDefaultLibNotFound(const char* apiName)
{
    ASCEND_LOGI("%s is not in default custom lib.", apiName);
}
inline void* FindFuncInDefaultLib(const char* apiName)
{
    if (!ShouldSearchDefaultLib()) {
        return nullptr;
    }
    void* result = SearchDefaultLibPaths(apiName);
    if (result == nullptr) {
        LogDefaultLibNotFound(apiName);
    }
    return result;
}
inline void* GetFuncFromDefaultLib(const char* apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}
inline void *GetOpApiFuncAddr(const char *apiName)
{
    void* funcAddr = FindFuncInCustomLib(apiName);
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    funcAddr = FindFuncInDefaultLib(apiName);
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    return GetFuncFromDefaultLib(apiName);
}
c10::Scalar CreateScalarFromDouble(const at::Tensor* tensor);
c10::Scalar CreateScalarFromLong(const at::Tensor* tensor);
c10::Scalar CreateScalarFromFloat(const at::Tensor* tensor);
c10::Scalar CreateScalarFromInt(const at::Tensor* tensor);
c10::Scalar CreateScalarFromHalf(const at::Tensor* tensor);
c10::Scalar CreateScalarFromBool(const at::Tensor* tensor);
c10::Scalar CreateScalarFromComplexDouble(const at::Tensor* tensor);
c10::Scalar CreateScalarFromComplexFloat(const at::Tensor* tensor);
c10::Scalar CreateScalarFromBFloat16(const at::Tensor* tensor);
inline c10::Scalar CreateScalarFromDouble(const at::Tensor* aclInput)
{
    double value = *(double *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromLong(const at::Tensor* aclInput)
{
    int64_t value = *(int64_t *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromFloat(const at::Tensor* aclInput)
{
    float value = *(float *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromInt(const at::Tensor* aclInput)
{
    int value = *(int *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromHalf(const at::Tensor* aclInput)
{
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromBool(const at::Tensor* aclInput)
{
    int8_t value = *(int8_t *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromComplexDouble(const at::Tensor* aclInput)
{
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromComplexFloat(const at::Tensor* aclInput)
{
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromBFloat16(const at::Tensor* aclInput)
{
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    return c10::Scalar(value);
}
c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    const at::Tensor *aclInput = &tensor;
    switch (aclInput->scalar_type()) {
        case at::ScalarType::Double:
            return CreateScalarFromDouble(aclInput);
        case at::ScalarType::Long:
            return CreateScalarFromLong(aclInput);
        case at::ScalarType::Float:
            return CreateScalarFromFloat(aclInput);
        case at::ScalarType::Int:
            return CreateScalarFromInt(aclInput);
        case at::ScalarType::Half:
            return CreateScalarFromHalf(aclInput);
        case at::ScalarType::Bool:
            return CreateScalarFromBool(aclInput);
        case at::ScalarType::ComplexDouble:
            return CreateScalarFromComplexDouble(aclInput);
        case at::ScalarType::ComplexFloat:
            return CreateScalarFromComplexFloat(aclInput);
        case at::ScalarType::BFloat16:
            return CreateScalarFromBFloat16(aclInput);
        default:
            return c10::Scalar();
    }
}
inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpuTensor)
{
    at::Tensor cpuPinMemTensor = cpuTensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex),
                              cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpuScalar, at::ScalarType scalarDataType)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpuScalar).to(scalarDataType));
}

AclTensor *ConvertType(const at::Tensor &atTensor)
{
    if (!atTensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalarDataType = atTensor.scalar_type();
    aclDataType aclType = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarDataType)];
    TORCH_CHECK(aclType != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalarDataType)) + " has not been supported")
    c10::SmallVector<int64_t, ACL_TENSOR_MAX_DIM_FOR_FORMAT> storageDims;
    // if aclType is ACL_STRING, storageDims is empty.
    auto itemSize = atTensor.itemsize();
    if (itemSize == 0) {
        AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
        return nullptr;
    }
    if (aclType != ACL_STRING) {
        storageDims.push_back(atTensor.storage().nbytes() / itemSize);
    }

    const auto dimNum = atTensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case DIM_NUM_3D:
            format = ACL_FORMAT_NCL;
            break;
        case DIM_NUM_4D:
            format = ACL_FORMAT_NCHW;
            break;
        case DIM_NUM_5D:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }

    if (atTensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(atTensor);
        at::Tensor aclInput = CopyScalarToDevice(expScalar, scalarDataType);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), aclType,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto aclTensorObj =
        aclCreateTensor(atTensor.sizes().data(), atTensor.sizes().size(), aclType, atTensor.strides().data(),
                        atTensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(atTensor.storage().data()));
    return aclTensorObj;
}

AclScalar *ConvertType(const at::Scalar &atScalar)
{
    at::ScalarType scalarDataType = atScalar.type();
    aclDataType aclType = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarDataType)];
    TORCH_CHECK(aclType != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalarDataType)) + " has not been supported")
    AclScalar *aclScalarObj = nullptr;
    switch (scalarDataType) {
        case at::ScalarType::Double: {
            double value = atScalar.toDouble();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = atScalar.toLong();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = atScalar.toBool();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = atScalar.toComplexDouble();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        default:
            aclScalarObj = nullptr;
            break;
    }
    return aclScalarObj;
}

inline AclIntArray *ConvertType(const at::IntArrayRef &atArray)
{
    auto array = aclCreateIntArray(atArray.data(), atArray.size());
    return array;
}

template <std::size_t N> inline AclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline AclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline AclTensorList *ConvertType(const at::TensorList &atTensorList)
{
    std::vector<const AclTensor *> tensorTist(atTensorList.size());
    for (size_t i = 0; i < atTensorList.size(); i++) {
        tensorTist[i] = ConvertType(atTensorList[i]);
    }
    auto aclTensorList = aclCreateTensorList(tensorTist.data(), tensorTist.size());
    return aclTensorList;
}

inline AclTensor *ConvertType(const c10::optional<at::Tensor> &optTensor)
{
    if (optTensor.has_value() && optTensor.value().defined()) {
        return ConvertType(optTensor.value());
    }
    return nullptr;
}

inline AclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &optArray)
{
    if (optArray.has_value()) {
        return ConvertType(optArray.value());
    }
    return nullptr;
}

inline AclScalar *ConvertType(const c10::optional<at::Scalar> &optScalar)
{
    if (optScalar.has_value()) {
        return ConvertType(optScalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarType)];
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(AclTensor *p)
{
    aclDestroyTensor(p);
}

inline void Release(AclScalar *p)
{
    aclDestroyScalar(p);
}

inline void Release(AclIntArray *p)
{
    aclDestroyIntArray(p);
}

inline void Release(AclBoolArray *p)
{
    aclDestroyBoolArray(p);
}

inline void Release(AclTensorList *p)
{
    aclDestroyTensorList(p);
}

template <typename T> void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I> void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple> void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I> auto Call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto Call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return Call(f, t, std::make_index_sequence<size>{});
}

uint64_t CalcHashId();
using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = void (*)(void *, bool);
using ReleaseHugeMem = void (*)(void *, bool);

static void ValidateApiAddresses(
    const void* getWorkspaceSizeFuncAddr, 
    const void* opApiFuncAddr,
    std::string_view apiName,
    std::string_view workspaceSizeApiStr)
{
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        apiName.data(), " or ", workspaceSizeApiStr.data(), " not in ", GetOpApiLibName(),
        ", or ", GetOpApiLibName(), " not found."
    );
}

static void InitHugeMem(const void* initMemAddr)
{
    using InitHugeMemFunc = int (*)(FunctionPtr<>, bool);
    InitHugeMemFunc initMemFunc = reinterpret_cast<InitHugeMemFunc>(initMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
}

template <std::string_view const& ApiName, typename GetWorkspaceSizeFuncType, typename... Args>
static auto PrepareParamsAndCalcWorkspaceSize(
    uint64_t* workspaceSizeAddr, AclOpExecutor** executorAddr,
    GetWorkspaceSizeFuncType getWorkspaceSizeFuncAddr, Args&&... args)
{
    auto convertedParams = ConvertTypes(std::forward<Args>(args)..., workspaceSizeAddr, executorAddr);
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr);
    auto workspaceStatus = Call(getWorkspaceSizeFunc, convertedParams);

    TORCH_CHECK(
        workspaceStatus == 0,
        "call ", ApiName.data(), " failed, detail:", aclGetRecentErrMsg()
    );

    return convertedParams;
}

static void* AllocateWorkspace(uint64_t workspaceSize, at::Tensor& workspaceTensor)
{
    if (workspaceSize == 0) {
        return nullptr;
    }

    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    workspaceTensor = at::empty({workspaceSize}, options.dtype(c10::kByte));
    return const_cast<void*>(workspaceTensor.storage().data());
}

template <typename ReleaseMemAddrType>
static void ReleaseHugeMemResource(ReleaseMemAddrType releaseMemAddr)
{
    using ReleaseHugeMemFunc = void (*)(FunctionPtr<>, bool);
    ReleaseHugeMemFunc releaseMemFunc = reinterpret_cast<ReleaseHugeMemFunc>(releaseMemAddr);
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
}

template <
    std::string_view const& ApiName, 
    typename ConvertedParamsType, 
    typename OpApiFuncAddrType, 
    typename ReleaseMemAddrType
    >
static void RunOpCommand(
    aclrtStream aclStreamObj, void* workspaceAddr, uint64_t workspaceSize,
    AclOpExecutor* executor, ConvertedParamsType convertedParams, OpApiFuncAddrType opApiFuncAddr,
    ReleaseMemAddrType releaseMemAddr)
{
    auto aclCall = [=]() -> int {
        using OpApiFunc = int (*)(FunctionPtr<>, uint64_t, AclOpExecutor*, const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto apiRet = opApiFunc(workspaceAddr, workspaceSize, executor, aclStreamObj);

        TORCH_CHECK(
            apiRet == 0,
            "call ", ApiName.data(), " failed, detail:", aclGetRecentErrMsg()
        );

        ReleaseConvertTypes(convertedParams);
        ReleaseHugeMemResource(releaseMemAddr);
        return apiRet;
    };

    at_npu::native::OpCommand cmd;
    cmd.Name(ApiName.data());
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
}

template <typename UnInitMemAddrType>
static void UnInitHugeMem(UnInitMemAddrType unInitMemAddr)
{
    using UnInitHugeMemFunc = void (*)(FunctionPtr<>, bool);
    UnInitHugeMemFunc unInitMemFunc = reinterpret_cast<UnInitHugeMemFunc>(unInitMemAddr);
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
}

template <std::string_view const& ApiName, typename... Args>
void EXEC_NPU_CMD(Args&&... args)
{
    static constexpr auto workspaceSizeApiStr = GetWorkspaceSizeApiName<ApiName>();
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiStr.data());
    static const auto opApiFuncAddr = GetOpApiFuncAddr(ApiName.data());
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");

    ValidateApiAddresses(
        getWorkspaceSizeFuncAddr, 
        opApiFuncAddr, 
        ApiName, 
        std::string_view(workspaceSizeApiStr.data(), workspaceSizeApiStr.size())
    );

    InitHugeMem(initMemAddr);
    uint64_t workspaceSize = 0;
    AclOpExecutor* executor = nullptr;
    auto convertedParams = PrepareParamsAndCalcWorkspaceSize<ApiName>(
        &workspaceSize, 
        &executor, 
        getWorkspaceSizeFuncAddr,
        std::forward<Args>(args)...
    );
    at::Tensor workspaceTensor;
    void* workspaceAddr = AllocateWorkspace(workspaceSize, workspaceTensor);
    auto aclStreamObj = c10_npu::getCurrentNPUStream().stream(false);

    RunOpCommand<ApiName>(
        aclStreamObj, workspaceAddr, workspaceSize, executor,
        convertedParams, opApiFuncAddr, releaseMemAddr);

    UnInitHugeMem(unInitMemAddr);
}

#endif // PYTORCH_NPU_HELPER_HPP_
```