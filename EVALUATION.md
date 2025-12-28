# 模型评估指南

本文档介绍如何使用评估脚本对微调后的模型进行全面评估，包括通用评估和IFEval指令遵循评估。

## 目录

1. [快速开始](#快速开始)
2. [通用模型评估](#通用模型评估)
3. [IFEval指令遵循评估](#ifeval指令遵循评估)
4. [评估指标说明](#评估指标说明)
5. [故障排除](#故障排除)

## 快速开始

### 步骤 1: 准备评估配置

编辑 `eval_config.yaml`，设置要评估的 checkpoint 路径：

```yaml
adapter_name_or_path: ./saves/qwen3-4b-base/lora/sft/checkpoint-2000  # 修改为你的checkpoint路径
```

### 步骤 2: 运行评估

```bash
llamafactory-cli train eval_config.yaml
```

### 步骤 3: 可视化结果

```bash
python visualize_evaluation.py --results_dir ./evaluation_results
```

## 评估指标说明

评估脚本会计算以下指标：

1. **BLEU Score**: 衡量生成文本与参考文本的n-gram重叠度
   - 范围: 0-100，越高越好
   - 通常 >20 表示较好的性能

2. **ROUGE Score**: 衡量生成文本与参考文本的重叠度
   - **ROUGE-1**: 单个词级别的重叠
   - **ROUGE-2**: 两个词级别的重叠
   - **ROUGE-L**: 最长公共子序列
   - 范围: 0-100，越高越好

3. **长度统计**: 
   - 平均预测长度 vs 平均参考长度
   - 长度比例（理想值为1.0）

4. **词汇多样性**: 
   - 衡量生成文本的词汇丰富程度
   - 值越高表示词汇越丰富

## 输出文件说明

评估完成后，会在输出目录生成以下文件：

### 评估指标
- `eval_results.json`: JSON格式的评估指标
- `metrics.json`: 详细的指标数据（使用自定义脚本时）

### 预测结果
- `generated_predictions.jsonl`: 每行的格式为：
  ```json
  {"prompt": "用户输入", "predict": "模型预测", "label": "参考答案"}
  ```

### 可视化图表
- `evaluation_metrics.png`: 包含多个子图的综合指标可视化
  - BLEU和ROUGE分数柱状图
  - 长度分布直方图
  - 长度比例分布
  - 指标汇总表

### 示例展示
- `examples.json`: JSON格式的示例
- `examples.md`: Markdown格式的示例（更易读）

## 使用自定义评估脚本

如果你想更精细地控制评估过程，可以使用 `evaluate_model.py`：

```bash
python evaluate_model.py \
  --checkpoint ./saves/qwen3-4b-base/lora/sft/checkpoint-2000 \
  --base_model ./Qwen3-4B-Base \
  --dataset tulu3_sft_personas \
  --max_samples 100 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --output_dir ./evaluation_results
```

### 参数说明

- `--checkpoint`: LoRA checkpoint 路径（必需）
- `--base_model`: 基础模型路径（默认: ./Qwen3-4B-Base）
- `--dataset`: 测试数据集名称（默认: tulu3_sft_personas）
- `--max_samples`: 最大评估样本数（默认: 100）
- `--max_new_tokens`: 生成的最大token数（默认: 512）
- `--temperature`: 生成温度，控制随机性（默认: 0.7）
  - 较低值（0.1-0.5）: 更确定、保守的输出
  - 较高值（0.7-1.0）: 更随机、创造性的输出
- `--output_dir`: 结果保存目录（默认: ./evaluation_results）

## 依赖安装

评估脚本需要额外的依赖：

```bash
pip install matplotlib seaborn rouge-score nltk
```

如果遇到问题，可以：

```bash
pip install -r requirements.txt
```

## 故障排除

### 问题1: 模型加载失败

如果直接加载模型失败，建议使用 LLaMA Factory 的方式进行评估（方式一）。

### 问题2: 依赖缺失

确保安装了所有必需的依赖：
```bash
pip install matplotlib seaborn rouge-score nltk
```

### 问题3: 内存不足

如果遇到内存不足，可以：
- 减少 `--max_samples` 参数
- 使用较小的 batch size（在配置文件中设置 `per_device_eval_batch_size: 2`）
- 使用量化模型

### 问题4: 生成质量不佳

如果生成的文本质量不理想：
- 尝试调整 `temperature` 参数（0.3-0.9）
- 检查 checkpoint 是否训练充分
- 确认数据集格式是否正确

## 示例输出

评估完成后，你会看到类似以下的输出：

```
============================================================
评估指标摘要
============================================================
BLEU Score:        25.34%
ROUGE-1 F1:        42.18%
ROUGE-2 F1:        18.92%
ROUGE-L F1:        38.76%
平均预测长度:      156.3 words
平均参考长度:      142.1 words
长度比例:          1.10
============================================================
```

可视化图表会保存在 `evaluation_results` 目录下，可以直接查看。

---

## 通用模型评估

### 使用 LLaMA Factory 进行评估（推荐）

这是最简单和推荐的方式：

1. **编辑评估配置**：
   ```bash
   # 编辑 eval_config.yaml，设置要评估的checkpoint路径
   # adapter_name_or_path: ./saves/qwen3-4b-base/lora/sft/checkpoint-2000
   ```

2. **运行评估**：
   ```bash
   llamafactory-cli train eval_config.yaml
   ```
   
   这会在 `evaluation_results` 目录下生成：
   - `eval_results.json` - 评估指标
   - `generated_predictions.jsonl` - 预测结果

3. **可视化评估结果**：
   ```bash
   python visualize_evaluation.py --results_dir ./evaluation_results
   ```
   
   这会生成：
   - `comprehensive_metrics.png` - 综合指标可视化图表
   - `evaluation_metrics.png` - 详细指标分析图表
   - `predictions_analysis.png` - 预测结果分析图表
   - `examples.json` / `examples.md` - 示例展示

### 使用自定义评估脚本

如果你想更精细地控制评估过程，可以使用 `evaluate_model.py`：

```bash
python evaluate_model.py \
  --checkpoint ./saves/qwen3-4b-base/lora/sft/checkpoint-2000 \
  --base_model ./Qwen3-4B-Base \
  --dataset tulu3_sft_personas \
  --max_samples 100 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --output_dir ./evaluation_results
```

---

## IFEval指令遵循评估

IFEval（Instruction-Following Evaluation）是专门用于评估模型指令遵循能力的基准测试。由于LLaMA Factory不直接支持IFEval的特殊评估指标，我们提供了专门的评估脚本。

### 数据集信息

**IFEval (Instruction-Following Evaluation)**
- **来源**: `google/IFEval` on HuggingFace
- **描述**: 包含约500个可验证指令，用于评估指令遵循能力
- **论文**: [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)
- **仓库**: https://github.com/google-research/google-research/tree/master/instruction_following_eval

IFEval数据集测试各种指令类型，如：
- 字数约束（"write in more than 400 words"）
- 关键词要求（"mention the keyword of AI at least 3 times"）
- 格式要求（"Do not use any commas"）
- 重复要求（"repeat the request above word for word"）

### 使用方法

#### 基本使用

```bash
# 评估SFT和DPO模型（默认）
python evaluate_ifeval_with_llamafactory.py
```

#### 高级选项

```bash
# 只评估SFT模型
python evaluate_ifeval_with_llamafactory.py --eval_sft_only

# 只评估DPO模型
python evaluate_ifeval_with_llamafactory.py --eval_dpo_only

# 使用lm_eval计算IFEval评分（推荐，与Stage1方法一致）
python evaluate_ifeval_with_llamafactory.py --use_lm_eval

# 限制评估样本数（加快测试）
python evaluate_ifeval_with_llamafactory.py --max_samples 20

# 指定不同的模型路径
python evaluate_ifeval_with_llamafactory.py \
  --sft_checkpoint ./saves/qwen3-4b-base/lora/sft/checkpoint-2532 \
  --dpo_checkpoint ./saves/qwen3-4b-base/lora/dpo/checkpoint-1875 \
  --base_model ./Qwen3-4B-Base
```

### 参数说明

- `--sft_checkpoint`: SFT checkpoint路径（默认: `./saves/qwen3-4b-base/lora/sft/checkpoint-2532`）
- `--dpo_checkpoint`: DPO checkpoint路径（默认: `./saves/qwen3-4b-base/lora/dpo/checkpoint-1875`）
- `--base_model`: 基础模型路径（默认: `./Qwen3-4B-Base`）
- `--cache_dir`: 数据集缓存目录（默认: `./MLS_project_stage1/dataset_cache`）
- `--output_dir`: 输出目录（默认: `./instruction_following_evaluation`）
- `--max_samples`: 最大评估样本数（默认: 100，设为None评估全部）
- `--use_lm_eval`: 使用lm_eval计算IFEval评分（推荐）
- `--eval_sft_only`: 只评估SFT模型
- `--eval_dpo_only`: 只评估DPO模型

### 输出文件

评估完成后，输出目录包含：

1. **sft_ifeval_predictions.json** - SFT模型的预测结果
   - 格式：JSON数组，每个元素包含 `key`, `prompt`, `prediction`, `instruction_id_list`, `kwargs`

2. **dpo_ifeval_predictions.json** - DPO模型的预测结果
   - 格式：同上

3. **ifeval_summary.json** - 评估摘要（包含IFEval评分，如果使用--use_lm_eval）
   - 包含：样本数、平均预测长度、IFEval评分等

4. **sft_ifeval_lm_eval_results.json** - SFT模型的完整lm_eval评估结果（如果使用--use_lm_eval）
   - 包含完整的评估结果，包括 `prompt_level_strict_acc` 分数

5. **dpo_ifeval_lm_eval_results.json** - DPO模型的完整lm_eval评估结果（如果使用--use_lm_eval）

6. **ifeval_score_comparison.png** - IFEval评分对比图（如果使用--use_lm_eval）

### 获取IFEval评分

要获得与Stage1相同的IFEval评分（`prompt_level_strict_acc`），请使用 `--use_lm_eval` 参数：

```bash
python evaluate_ifeval_with_llamafactory.py --use_lm_eval
```

评分会保存在：
- `ifeval_summary.json` 中的 `ifeval_score` 字段（0-1之间的浮点数）
- `*_ifeval_lm_eval_results.json` 中的 `results.ifeval.prompt_level_strict_acc,none` 字段

### 注意事项

1. **数据集缓存**: 脚本会自动使用Stage1的缓存数据集，无需重新下载
2. **显存使用**: 如果显存不足，可以减少 `--max_samples` 或分别评估SFT和DPO
3. **评估时间**: IFEval数据集有约500个样本，完整评估可能需要一些时间
4. **模型格式**: 确保模型路径正确，脚本会先加载基础模型，然后加载PEFT适配器

---

## 评估指标说明

### BLEU Score

衡量生成文本与参考文本的n-gram重叠度
- 范围: 0-100，越高越好
- 通常 >20 表示较好的性能
- BLEU-4 是最常用的版本

### ROUGE Score

衡量生成文本与参考文本的重叠度
- **ROUGE-1**: 单个词级别的重叠
- **ROUGE-2**: 两个词序列的重叠
- **ROUGE-L**: 最长公共子序列
- 范围: 0-100，越高越好

### IFEval Score

专门用于评估指令遵循能力的指标
- **prompt_level_strict_acc**: 严格准确率，评估模型是否完全遵循指令中的所有约束
- 范围: 0-1，越高越好
- 这是IFEval的核心指标，与Stage1评估使用的方法一致

### 长度统计

- 平均预测长度 vs 平均参考长度
- 长度比例（理想值为1.0）
- 长度分布分析

### 词汇多样性

- 衡量生成文本的词汇丰富程度
- 值越高表示词汇越丰富

---

## 故障排除

### 通用评估问题

#### 问题1: 模型加载失败

如果直接加载模型失败，建议使用 LLaMA Factory 的方式进行评估：

```bash
llamafactory-cli train eval_config.yaml
```

#### 问题2: 依赖缺失

确保安装了所有必需的依赖：
```bash
pip install matplotlib seaborn rouge-score nltk lm-eval
```

#### 问题3: 内存不足

如果遇到内存不足，可以：
- 减少 `--max_samples` 参数
- 使用较小的 batch size（在配置文件中设置 `per_device_eval_batch_size: 2`）
- 使用量化模型

#### 问题4: 生成质量不佳

如果生成的文本质量不理想：
- 尝试调整 `temperature` 参数（0.3-0.9）
- 检查 checkpoint 是否训练充分
- 确认数据集格式是否正确

### IFEval评估问题

#### 问题1: 数据集加载失败

```bash
# 检查缓存目录是否存在
ls -la ./MLS_project_stage1/dataset_cache/hub/datasets--google--IFEval

# 如果不存在，需要先运行Stage1的评估脚本下载数据集
```

#### 问题2: 模型加载失败

```bash
# 检查模型路径是否正确
ls -la ./Qwen3-4B-Base
ls -la ./saves/qwen3-4b-base/lora/sft/checkpoint-2532
```

#### 问题3: 显存不足

```bash
# 减少评估样本数
python evaluate_ifeval_with_llamafactory.py --max_samples 50

# 或者只评估一个模型
python evaluate_ifeval_with_llamafactory.py --eval_sft_only
```

#### 问题4: lm_eval评估失败

如果使用 `--use_lm_eval` 时遇到问题：
- 确保已安装 `lm-eval`: `pip install lm-eval`
- 检查PEFT兼容性，`evaluate_ifeval_with_llamafactory.py` 已解决PEFT兼容性问题
- 如果仍有问题，可以不使用 `--use_lm_eval`，只生成预测结果

---

## 评估结果对比

### SFT vs DPO 模型对比

评估完成后，可以对比SFT和DPO模型的表现：

1. **查看评分文件**：
   - `ifeval_summary.json` 包含两个模型的评分对比
   - `evaluation_results_sft_checkpoint-2532/` 和 `evaluation_results_dpo/` 目录包含各自的评估结果

2. **可视化对比**：
   - IFEval评分对比图（如果使用--use_lm_eval）
   - 综合指标对比图

3. **详细分析**：
   - 参考 `PROJECT_REPORT.md` 中的详细对比分析

---

## 评估结果管理

### 评估结果目录结构

评估结果会保存在以下目录：
```
evaluation_results_dpo/                    # DPO模型评估结果
evaluation_results_sft_checkpoint-2532/   # SFT模型评估结果
```

### 避免覆盖现有结果

**方法1：使用 run_eval.py（推荐）**

`run_eval.py` 脚本会自动根据adapter路径生成唯一的output_dir，避免覆盖：

```bash
# 评估SFT模型
python run_eval.py eval_config_sft.yaml

# 或直接使用主配置文件
python run_eval.py eval_config.yaml
```

**方法2：手动指定output_dir**

编辑配置文件，手动指定不同的output_dir：

```yaml
adapter_name_or_path: ./saves/qwen3-4b-base/lora/sft/checkpoint-2532
output_dir: ./evaluation_results_sft_checkpoint-2532  # 手动指定，避免覆盖
```

### 评估结果文件说明

每个评估结果目录包含：

- `eval_results.json` - 评估集指标（BLEU, ROUGE等）
- `predict_results.json` - 预测集指标
- `generated_predictions.jsonl` - 完整预测结果
- `examples.md` - 预测示例（Markdown格式）
- `examples.json` - 预测示例（JSON格式）
- `comprehensive_metrics.png` - 综合指标可视化图表
- `evaluation_metrics.png` - 详细指标分析图表
- `predictions_analysis.png` - 预测结果分析图表

### 注意事项

1. **避免覆盖**：每次评估前确保使用不同的output_dir
2. **检查配置**：运行评估前检查adapter_path是否正确
3. **备份结果**：重要评估结果建议备份
4. **磁盘空间**：评估结果可能占用较大空间，注意磁盘使用情况

