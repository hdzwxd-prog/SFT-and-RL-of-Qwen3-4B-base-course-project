# 强化学习训练指南 (RL Training Guide)

本文档介绍如何在SFT（监督微调）的基础上进行强化学习训练，以提高模型的指令遵循能力。

## 目录

1. [概述](#概述)
2. [推荐方法](#推荐方法)
3. [数据集选择](#数据集选择)
4. [训练步骤](#训练步骤)
5. [配置说明](#配置说明)
6. [常见问题](#常见问题)

---

## 概述

强化学习（RL）训练是LLM微调流程中的重要环节，通常用于：

- **提高指令遵循能力**：让模型更好地理解和执行用户指令
- **改善输出质量**：生成更符合人类偏好的回答
- **对齐人类价值观**：使模型输出更加安全、有用、无害

### RL训练方法对比

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **DPO** | 简单高效，无需奖励模型，训练稳定 | 需要chosen/rejected对 | ⭐⭐⭐⭐⭐ 推荐 |
| **KTO** | 只需标注desirable/undesirable，数据准备更简单 | 效果可能略逊于DPO | ⭐⭐⭐⭐ |
| **PPO** | 理论上更灵活 | 需要训练奖励模型，实现复杂，训练不稳定 | ⭐⭐⭐ |

**建议：优先使用DPO方法，简单高效，效果优秀。**

---

## 推荐方法：DPO (Direct Preference Optimization)

DPO是目前最流行的RL训练方法，具有以下优势：

1. **无需奖励模型**：直接优化偏好，比PPO更简单
2. **训练稳定**：使用参考模型约束，避免过度优化
3. **效果优秀**：在多个基准测试中表现出色
4. **资源效率**：显存占用相对较小

---

## 数据集选择

### 推荐的偏好数据集

#### 1. UltraFeedback（⭐ 强烈推荐）

- **数据集ID**: `HuggingFaceH4/ultrafeedback_binarized`
- **语言**: 英文
- **大小**: 约6.4万条偏好对
- **特点**: 高质量、多样化的指令遵循偏好对
- **适用场景**: 英文指令遵循任务

#### 2. Orca DPO Pairs

- **数据集ID**: `Intel/orca_dpo_pairs`
- **语言**: 英文
- **大小**: 约10万条偏好对
- **特点**: Intel发布的Orca系列偏好数据
- **适用场景**: 英文指令遵循任务

#### 3. DPO En-Zh Mixed

- **数据集ID**: `hiyouga/DPO-En-Zh-20k`
- **语言**: 中英文混合
- **大小**: 约2万条偏好对
- **特点**: 支持中英文指令
- **适用场景**: 需要支持中文的指令遵循任务

#### 4. COIG-P

- **数据集ID**: `m-a-p/COIG-P`
- **语言**: 中文
- **大小**: 约1万条偏好对
- **特点**: 中文偏好数据集
- **适用场景**: 中文指令遵循任务

---

## 训练步骤

### 步骤 1: 下载偏好数据集

使用提供的脚本下载偏好数据集：

```bash
# 查看可用数据集
python download_preference_dataset.py --list

# 下载UltraFeedback数据集（推荐）
python download_preference_dataset.py ultrafeedback

# 或下载其他数据集
python download_preference_dataset.py orca_dpo
python download_preference_dataset.py dpo_en_zh
```

**注意**：数据集会自动缓存到本地，无需手动下载。

### 步骤 2: 检查SFT checkpoint

确保您已经完成了SFT训练，并有可用的checkpoint：

```bash
# 查看可用的SFT checkpoint
ls -lh saves/qwen3-4b-base/lora/sft/checkpoint-*

# 建议使用最新的checkpoint，例如：
# saves/qwen3-4b-base/lora/sft/checkpoint-2000
```

### 步骤 3: 配置DPO训练

编辑 `dpo_config.yaml` 文件，主要修改以下参数：

1. **`adapter_name_or_path`**: 设置为您的SFT checkpoint路径
   ```yaml
   adapter_name_or_path: ./saves/qwen3-4b-base/lora/sft/checkpoint-2000
   ```

2. **`dataset`**: 选择使用的偏好数据集（已在 `data/dataset_info.json` 中配置）
   ```yaml
   dataset: ultrafeedback  # 或其他数据集名称
   ```

3. **`output_dir`**: 设置DPO训练的输出目录
   ```yaml
   output_dir: ./saves/qwen3-4b-base/lora/dpo
   ```

### 步骤 4: 运行DPO训练

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 运行DPO训练
llamafactory-cli train dpo_config.yaml
```

训练过程会显示：
- 训练损失（preference loss）
- KL散度（与参考模型的差异）
- 奖励分数（chosen和rejected的平均logits差异）

### 步骤 5: 监控训练

训练过程中可以观察：

1. **损失曲线**：preference loss应该逐渐下降
2. **KL散度**：应该保持在合理范围（通常<10），过大表示过度偏离参考模型
3. **奖励差距**：chosen和rejected的奖励差距应该逐渐增大

训练完成后，损失曲线图会保存在 `output_dir` 目录中。

### 步骤 6: 评估模型

训练完成后，使用评估脚本测试模型性能：

```bash
# 使用评估配置测试DPO后的模型
# 先修改 eval_config.yaml 中的 adapter_name_or_path 为DPO checkpoint
# adapter_name_or_path: ./saves/qwen3-4b-base/lora/dpo/checkpoint-1875

llamafactory-cli train eval_config.yaml
```

或使用交互式对话测试：

```bash
llamafactory-cli chat \
  --model_name_or_path ./Qwen3-4B-Base \
  --adapter_name_or_path ./saves/qwen3-4b-base/lora/dpo/checkpoint-1875 \
  --template qwen3
```

---

## 配置说明

### 关键超参数说明

#### DPO特定参数

- **`pref_beta`** (默认: 0.1)
  - 控制KL散度约束的强度
  - 较大值（0.5）：更强约束，更保守
  - 较小值（0.1）：更灵活，允许更大改变
  - **推荐**: 0.1（对于4B模型）

- **`pref_loss`** (默认: sigmoid)
  - DPO损失函数类型
  - `sigmoid`: 标准DPO损失（推荐）
  - `hinge`: 对outlier更鲁棒
  - `orpo`: 不需要参考模型，但效果可能略差
  - **推荐**: sigmoid

- **`pref_ftx`** (默认: 0.0)
  - 混合SFT损失的权重
  - 0.0: 纯DPO损失（推荐）
  - 0.1: 加入少量SFT损失辅助

#### 训练参数

- **`learning_rate`** (推荐: 5.0e-6)
  - DPO训练使用较小的学习率
  - 范围: 1.0e-6 到 5.0e-6
  - 过大会导致训练不稳定

- **`num_train_epochs`** (推荐: 2.0-3.0)
  - DPO训练通常需要较少的epoch
  - 过多epoch可能导致过拟合

- **`per_device_train_batch_size`** (推荐: 1)
  - DPO训练通常使用较小的batch size
  - 因为需要同时计算chosen和rejected的logits

- **`gradient_accumulation_steps`** (推荐: 8)
  - 通过梯度累积增大有效batch size
  - 有效batch size = batch_size × accumulation_steps × GPU数

---

## 常见问题

### Q1: DPO训练需要多少显存？

**A**: 对于4B模型（Qwen3-4B-Base），使用LoRA+bf16，2张A6000（48GB）GPU足够：
- 模型：~8GB（每GPU）
- 参考模型：~8GB（每GPU，但可以共享）
- 训练状态：~4GB
- 总计：约20-25GB每GPU

如果显存不足，可以：
- 减小 `lora_rank`（如从16减到8）
- 减小 `cutoff_len`
- 增大 `gradient_accumulation_steps` 并减小 `per_device_train_batch_size`

### Q2: DPO训练需要多长时间？

**A**: 取决于数据集大小和训练配置：
- UltraFeedback（64k样本）：约3-6小时（2×A6000，3 epochs）
- Orca DPO（100k样本）：约5-8小时
- 较小的数据集（10k样本）：约1-2小时

### Q3: 如何选择合适的beta值？

**A**: 
- **0.1**: 适用于大多数情况，推荐起始值
- **0.2-0.5**: 如果发现模型过度偏离，可以增大
- **<0.1**: 不推荐，可能导致训练不稳定

### Q4: DPO训练后模型效果变差了怎么办？

**A**: 可能的原因和解决方案：

1. **过拟合**：减少 `num_train_epochs` 或增加 `pref_beta`
2. **学习率过大**：减小 `learning_rate`（如从5e-6减到1e-6）
3. **数据质量问题**：检查偏好数据集的质量
4. **checkpoint选择**：尝试不同的checkpoint（不一定是最新的最好）

### Q5: 可以使用自己的偏好数据吗？

**A**: 可以！需要：

1. 准备偏好数据（JSON格式）：
   ```json
   [
     {
       "instruction": "用户指令",
       "chosen": "优质回答",
       "rejected": "劣质回答"
     }
   ]
   ```

2. 在 `data/dataset_info.json` 中添加配置：
   ```json
   "my_preference_dataset": {
     "file_name": "my_preference_data.json",
     "ranking": true,
     "columns": {
       "prompt": "instruction",
       "chosen": "chosen",
       "rejected": "rejected"
     }
   }
   ```

3. 在配置文件中使用：
   ```yaml
   dataset: my_preference_dataset
   ```

### Q6: DPO vs KTO 如何选择？

**A**: 
- **DPO**：需要chosen/rejected对，效果通常更好，**推荐优先使用**
- **KTO**：只需标注desirable/undesirable，数据准备更简单，但效果可能略差

如果您的数据只有单一回答（没有明确的chosen/rejected对），可以考虑KTO。

---

## 训练流程总结

```
1. SFT训练 (已完成)
   ↓
2. 下载偏好数据集
   python download_preference_dataset.py ultrafeedback
   ↓
3. 配置DPO训练
   编辑 dpo_config.yaml
   ↓
4. 运行DPO训练
   llamafactory-cli train dpo_config.yaml
   ↓
5. 评估DPO模型
   llamafactory-cli train eval_config.yaml
   ↓
6. 部署和使用
   llamafactory-cli chat ...
```

---

## 参考资料

- [LLaMA Factory文档](https://github.com/hiyouga/LLaMA-Factory)
- [DPO论文](https://arxiv.org/abs/2305.18290)
- [KTO论文](https://arxiv.org/abs/2402.01306)
- [UltraFeedback数据集](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

---

## 下一步

完成DPO训练后，您可以：

1. **继续优化**：尝试不同的超参数或数据集
2. **多轮RL训练**：在DPO基础上再进行一轮DPO训练
3. **部署模型**：将训练好的模型部署到生产环境
4. **评估指标**：使用更详细的评估指标（如人工评估）来评估模型性能

