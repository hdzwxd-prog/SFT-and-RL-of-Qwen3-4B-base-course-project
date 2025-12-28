# Qwen3-4B-Base 微调项目

基于 LLaMA Factory 的 Qwen3-4B-Base 模型微调项目，使用 Tulu-3-sft-personas-instruction-following 数据集。

## 项目结构

```
MLS_project/
├── 核心脚本/
│   ├── download_model.py              # 模型下载脚本
│   ├── download_dataset.py            # 数据集下载脚本
│   ├── download_preference_dataset.py  # 偏好数据集下载脚本
│   ├── download_all.py                # 一键下载脚本
│   ├── train.py                        # 训练主脚本
│   ├── run_eval.py                     # 评估脚本包装器（自动管理输出目录）
│   ├── evaluate_ifeval_with_llamafactory.py  # IFEval评估脚本
│   ├── evaluate_model.py               # 通用模型评估脚本
│   └── visualize_evaluation.py         # 评估结果可视化脚本
│
├── 配置文件/
│   ├── train_config.yaml               # SFT训练配置
│   ├── dpo_config.yaml                 # DPO训练配置
│   ├── eval_config.yaml                # 评估配置
│   ├── eval_config_sft.yaml            # SFT评估配置
│   ├── eval_config_dpo.yaml            # DPO评估配置
│   └── data/dataset_info.json          # 数据集信息配置
│
├── 文档/
│   ├── README.md                       # 项目主文档（本文件）
│   ├── PROJECT_REPORT.md               # 项目完整报告
│   ├── MLSYS 项目一：任务概述.md       # 任务概述
│   ├── EVALUATION.md                   # 评估指南（包含通用评估和IFEval评估）
│   ├── RL_TRAINING.md                  # 强化学习训练指南
│   └── TROUBLESHOOTING.md              # 故障排除指南
│
├── 数据与模型/
│   ├── Qwen3-4B-Base/                  # 基础模型目录（下载后）
│   ├── tulu-3-sft-personas-instruction-following/  # SFT数据集（下载后）
│   └── saves/                          # 训练检查点保存目录
│       └── qwen3-4b-base/lora/
│           ├── sft/                    # SFT训练结果
│           └── dpo/                    # DPO训练结果
│
├── 评估结果/
│   ├── evaluation_results_sft_checkpoint-2532/  # SFT模型评估结果
│   ├── evaluation_results_dpo/                 # DPO模型评估结果
│   └── instruction_following_evaluation/       # IFEval评估结果
│
└── MLS_project_stage1/                 # 阶段一：基础模型评估
    ├── eval_stage1_v3.py               # Stage1评估主脚本
    ├── eval_stage1_checkpoint.py       # Stage1评估检查点脚本
    ├── dataset_cache/                  # 数据集缓存
    └── detailed_eval_results/         # 详细评估结果
```

## 环境要求

- Python >= 3.9 (推荐 3.10)
- CUDA >= 11.6 (推荐 12.2)
- 2x NVIDIA A6000 GPU (48GB 显存)

## 快速开始

### 前置要求

1. **硬件**: 2x NVIDIA A6000 GPU (48GB 显存)
2. **软件**: 
   - Linux 系统
   - Python 3.9+ (推荐 3.10)
   - CUDA 11.6+ (推荐 12.2)

### 快速开始步骤

1. **创建和激活虚拟环境**（见下方详细说明）
2. **安装依赖**（见下方详细说明）
3. **下载模型和数据集**: `python download_all.py`
4. **开始训练**: `python train.py` 或 `llamafactory-cli train train_config.yaml`

详细步骤请参考下方各章节。

---

## 安装步骤

### 1. 创建和激活虚拟环境

为了避免依赖冲突，强烈建议使用虚拟环境。有两种方式：

#### 方式一：使用 venv（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

#### 方式二：使用 conda

```bash
# 创建conda环境
conda create -n qwen3-ft python=3.10 -y

# 激活conda环境
conda activate qwen3-ft
```

**注意**: 激活虚拟环境后，命令行提示符前会显示环境名称（如 `(venv)` 或 `(qwen3-ft)`），表示环境已激活。

### 2. 安装依赖

在激活的虚拟环境中安装依赖：

```bash
# 升级pip（推荐）
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

或者直接使用pip安装LLaMA Factory（推荐，更简单）：

```bash
# 使用pip直接安装（推荐，无需从GitHub克隆）
pip install llamafactory[torch,metrics] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果网络无法访问GitHub，或者想从源码安装：

```bash
# 先安装基础依赖
pip install -r requirements.txt

# 然后从源码安装LLaMA Factory（需要访问GitHub）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

**注意**: 如果使用conda，建议先用conda安装PyTorch（带CUDA支持）：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

然后再安装其他依赖：

```bash
pip install -r requirements.txt
```

### 3. 登录 HuggingFace（如果需要）

某些模型和数据集需要登录 HuggingFace：

```bash
huggingface-cli login
```

### 4. 检查环境

运行环境检查脚本（确保在激活的虚拟环境中）：

```bash
python check_setup.py
```

这会检查Python版本、依赖包、CUDA、配置文件和模型/数据集是否就绪。

### 5. 下载模型和数据集

#### 方式一：一键下载（推荐）

```bash
python download_all.py
```

#### 方式二：分别下载

```bash
# 下载模型（自动选择最快镜像）
python download_model.py

# 下载数据集（自动选择最快镜像）
python download_dataset.py
```

**自动镜像选择功能**：
- 脚本会自动测试多个国内镜像站点（HF-Mirror、ModelScope、Gitee AI等）
- 选择延迟最低的可用镜像进行下载，大幅提升下载速度
- 如果所有镜像都不可用，会自动回退到官方源

**禁用自动镜像选择**（使用官方源）：
```bash
python download_model.py --no-mirror
python download_dataset.py --no-mirror
```

**注意**: 下载脚本会自动检查文件是否已存在。如果已存在且完整，会跳过下载。支持断点续传，下载中断后重新运行会自动继续。

下载完成后，当前目录下会生成：
- `Qwen3-4B-Base/` - 模型文件
- `tulu-3-sft-personas-instruction-following/` - 数据集文件

## 训练配置

训练配置位于 `train_config.yaml`，主要参数：

- **模型**: Qwen3-4B-Base
- **方法**: LoRA微调（rank=16, alpha=32）
- **训练设备**: 2x A6000 (自动使用DDP)
- **批次大小**: per_device_train_batch_size=2, gradient_accumulation_steps=8
- **学习率**: 2.0e-4 (cosine调度)
- **训练轮数**: 3 epochs
- **序列长度**: 32768 tokens

### 调整配置

如需修改训练参数，编辑 `train_config.yaml` 文件。

对于2张A6000 GPU的配置建议：
- **LoRA**: rank=16-32, alpha=32-64
- **Batch Size**: per_device=1-4, gradient_accumulation=4-16
- **序列长度**: 根据显存调整（A6000支持较长序列）
- **混合精度**: bf16 (推荐) 或 fp16

## 开始训练

**重要**: 确保已激活虚拟环境后再开始训练！

### 方式一：使用训练脚本

```bash
# 确保虚拟环境已激活
python train.py
```

### 方式二：直接使用 llamafactory-cli

```bash
# 确保虚拟环境已激活
llamafactory-cli train train_config.yaml
```

### 多GPU训练

LLaMA Factory会自动检测并使用多GPU（DDP）。确保环境变量正确：

```bash
# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 开始训练
llamafactory-cli train train_config.yaml
```

## 训练输出

训练过程中的检查点和日志保存在：
- **模型检查点**: `./saves/qwen3-4b-base/lora/sft/`
- **日志**: 训练过程中的日志会显示在终端

## 使用虚拟环境

每次使用项目时，都需要先激活虚拟环境：

```bash
# 使用venv创建的虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 使用conda创建的虚拟环境
conda activate qwen3-ft
```

退出虚拟环境：

```bash
# venv
deactivate

# conda
conda deactivate
```

## 注意事项

1. **虚拟环境**: 每次运行训练或脚本前，确保已激活虚拟环境

2. **显存使用**: 2张A6000 (48GB) 足够训练4B模型，如果显存不足，可以：
   - 减小 `per_device_train_batch_size`
   - 增大 `gradient_accumulation_steps`
   - 减小 `cutoff_len`
   - 启用 `gradient_checkpointing` (已启用)

3. **数据集格式**: 确保 `data/dataset_info.json` 中的路径正确指向数据集目录

4. **模型路径**: 确保 `train_config.yaml` 中的 `model_name_or_path` 指向正确的模型目录

5. **中断恢复**: 训练中断后，重新运行相同命令会自动从检查点恢复

6. **环境管理**: 如果不小心退出了虚拟环境，记得重新激活后再运行命令

## 评估和推理

### 模型评估

训练完成后，可以使用评估脚本对模型进行全面评估。详细评估指南请参考 [EVALUATION.md](./EVALUATION.md)。

#### 快速评估

**方式一：使用 LLaMA Factory 进行评估（推荐）**

1. 编辑 `eval_config.yaml`，设置要评估的checkpoint路径
2. 运行评估：`llamafactory-cli train eval_config.yaml`
3. 可视化结果：`python visualize_evaluation.py --results_dir ./evaluation_results`

**方式二：IFEval指令遵循评估**

```bash
# 评估SFT和DPO模型（使用lm_eval计算评分）
python evaluate_ifeval_with_llamafactory.py --use_lm_eval
```

**方式三：使用评估脚本包装器（自动管理输出目录）**

```bash
# 自动根据checkpoint生成唯一的输出目录，避免覆盖
python run_eval.py eval_config.yaml
```

详细说明请参考 [EVALUATION.md](./EVALUATION.md)。

### 推理

可以使用以下命令进行交互式推理：

```bash
llamafactory-cli chat train_config.yaml
```

或导出合并后的模型：

```bash
llamafactory-cli export train_config.yaml
```

## 参考资源

- [LLaMA Factory 官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3 模型页面](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [Tulu-3 数据集页面](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)

## 故障排除

常见问题及解决方案请参考 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)。

### 快速问题排查

- **下载失败或速度慢**: 脚本会自动选择最快的国内镜像，如需禁用使用 `--no-mirror` 参数
- **显存不足 (OOM)**: 减小batch size、增大gradient accumulation、减小序列长度、使用QLoRA
- **多GPU训练问题**: 确认 `CUDA_VISIBLE_DEVICES` 设置正确
- **虚拟环境问题**: 确保已激活虚拟环境，如果损坏可重新创建

详细故障排除指南请参考 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)。

## 强化学习训练（RL Training）

完成SFT训练后，可以使用DPO方法进行强化学习训练，进一步提高模型的指令遵循能力。

### 快速开始

```bash
# 方式1: 使用快速开始脚本（推荐）
bash quick_start_rl.sh

# 方式2: 手动执行
# 1. 下载偏好数据集
python download_preference_dataset.py ultrafeedback

# 2. 配置训练参数（编辑 dpo_config.yaml）
# 主要修改 adapter_name_or_path 为您的SFT checkpoint路径

# 3. 开始DPO训练
llamafactory-cli train dpo_config.yaml
```

### 详细文档

完整的RL训练指南请参考：[RL_TRAINING.md](./RL_TRAINING.md)

### 推荐的数据集

- **UltraFeedback** (英文，推荐): 约6.4万条高质量偏好对
- **Orca DPO Pairs** (英文): 约10万条偏好对
- **DPO En-Zh Mixed** (中英文): 约2万条混合偏好对

### 配置文件

- `dpo_config.yaml`: DPO训练配置（推荐方法）
- `kto_config.yaml`: KTO训练配置（备选方案）

## 许可证

请遵循原模型和数据集的使用许可。

