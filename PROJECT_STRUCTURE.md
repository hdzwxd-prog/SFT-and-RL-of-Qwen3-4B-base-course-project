# 项目结构说明

本文档说明整理后的项目文件结构和功能模块。

## 核心脚本

### 下载脚本

- **download_model.py** - 下载Qwen3-4B-Base模型
  - 支持自动选择最快的国内镜像
  - 支持断点续传
  - 自动验证文件完整性

- **download_dataset.py** - 下载Tulu-3-sft-personas-instruction-following数据集
  - 支持自动镜像选择
  - 支持断点续传

- **download_preference_dataset.py** - 下载偏好数据集（用于DPO训练）
  - 支持UltraFeedback、Orca DPO Pairs等数据集
  - 支持自动镜像选择

- **download_all.py** - 一键下载模型和数据集
  - 调用download_model.py和download_dataset.py
  - 统一管理下载流程

### 训练脚本

- **train.py** - 训练主脚本
  - 封装llamafactory-cli调用
  - 提供统一的训练接口

### 评估脚本

- **evaluate_ifeval_with_llamafactory.py** - IFEval指令遵循评估
  - 直接使用transformers和peft加载模型，避免lm_eval的PEFT兼容性问题
  - 支持使用lm_eval计算IFEval评分（与Stage1方法一致）
  - 支持SFT和DPO模型对比评估
  - 生成预测结果和可视化图表

- **evaluate_model.py** - 通用模型评估脚本
  - 支持BLEU、ROUGE等指标计算
  - 支持自定义评估参数
  - 生成可视化图表和示例

- **visualize_evaluation.py** - 评估结果可视化
  - 生成综合指标图表
  - 生成预测分析图表
  - 保存评估示例

- **run_eval.py** - 评估脚本包装器
  - 自动根据checkpoint路径生成唯一的输出目录
  - 避免覆盖现有评估结果
  - 简化评估流程

## 配置文件

### 训练配置

- **train_config.yaml** - SFT训练配置
  - LoRA微调配置
  - 优化器配置
  - 训练超参数

- **dpo_config.yaml** - DPO训练配置
  - DPO特定参数
  - 偏好数据集配置
  - 训练超参数

### 评估配置

- **eval_config.yaml** - 主评估配置
- **eval_config_sft.yaml** - SFT评估专用配置
- **eval_config_dpo.yaml** - DPO评估专用配置

### 数据集配置

- **data/dataset_info.json** - 数据集信息配置
  - 数据集路径
  - 数据格式
  - 数据集元信息

## 文档

### 主要文档

- **README.md** - 项目主文档
  - 项目概述
  - 快速开始
  - 安装步骤
  - 训练指南
  - 评估指南（简要）
  - 故障排除（简要）

- **PROJECT_REPORT.md** - 项目完整报告
  - 阶段一：基础模型评估
  - 阶段二：SFT训练
  - 阶段三：DPO训练
  - 综合分析与总结
  - 详细评估结果和对比分析

- **MLSYS 项目一：任务概述.md** - 任务概述文档

### 功能文档

- **EVALUATION.md** - 评估指南
  - 通用模型评估（BLEU/ROUGE）
  - IFEval指令遵循评估
  - 评估指标说明
  - 评估结果管理
  - 故障排除

- **RL_TRAINING.md** - 强化学习训练指南
  - DPO方法介绍
  - 数据集选择
  - 训练步骤
  - 配置说明

- **TROUBLESHOOTING.md** - 故障排除指南
  - 环境配置问题
  - 训练相关问题
  - 评估相关问题
  - 模型加载问题
  - 调试技巧

## 数据目录

### 模型和数据集

- **Qwen3-4B-Base/** - 基础模型目录（下载后生成）
- **tulu-3-sft-personas-instruction-following/** - SFT数据集目录（下载后生成）

### 训练结果

- **saves/qwen3-4b-base/lora/**
  - **sft/** - SFT训练检查点
  - **dpo/** - DPO训练检查点

### 评估结果

- **evaluation_results_sft_checkpoint-2532/** - SFT模型评估结果
- **evaluation_results_dpo/** - DPO模型评估结果
- **instruction_following_evaluation/** - IFEval评估结果

## Stage1目录

**MLS_project_stage1/** - 阶段一：基础模型评估

### 核心脚本

- **eval_stage1_v3.py** - Stage1评估主脚本
  - 评估基础模型在多个基准测试上的表现
  - 生成详细评估结果

- **eval_stage1_checkpoint.py** - Stage1评估检查点脚本
  - 支持断点续传
  - 支持增量评估

### 数据修复脚本（历史记录）

以下脚本用于修复数据集下载问题，已不再使用，但保留作为历史记录：

- **fix_mmlu_subsets.py** - 修复MMLU子集下载
- **fix_mmlu_retry.py** - MMLU重试下载
- **fix_missing_others.py** - 修复其他缺失数据集
- **fix_final_missing.py** - 修复最终缺失数据集
- **full_download_verify.py** - 全量下载验证
- **test.py** - 测试脚本
- **test1.md** - 测试记录（Jupyter notebook导出）

### 评估结果

- **dataset_cache/** - 数据集缓存
- **detailed_eval_results/** - 详细评估结果
  - summary_report.json - 汇总报告
  - capability_radar.png - 能力雷达图
  - log_*.json - 各任务的详细日志
- **eval_results/** - 评估结果
  - summary.json - 汇总结果
  - capability_radar.png - 能力雷达图

## 已删除的文件

以下文件已在整理过程中删除或合并：

### 已删除的文档

- **IFEVAL_EVAL_GUIDE.md** - 已合并到EVALUATION.md
- **INSTRUCTION_FOLLOWING_EVAL.md** - 已合并到EVALUATION.md
- **DEBUGGING_GUIDE.md** - 已合并到TROUBLESHOOTING.md
- **QUICKSTART.md** - 已合并到README.md
- **PROJECT_SUMMARY.md** - 内容已整合到PROJECT_REPORT.md
- **report_review.md** - 内容已整合到PROJECT_REPORT.md
- **EVALUATION_GUIDE.md** - 已合并到EVALUATION.md

### 已删除的脚本

- **evaluate_instruction_following.py** - 功能已被evaluate_ifeval_with_llamafactory.py替代，且解决了PEFT兼容性问题

## 项目工作流程

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载模型和数据集
python download_all.py

# 下载偏好数据集（用于DPO训练）
python download_preference_dataset.py ultrafeedback
```

### 3. 阶段一：基础模型评估

```bash
cd MLS_project_stage1
python eval_stage1_v3.py
```

### 4. 阶段二：SFT训练

```bash
# 返回项目根目录
cd ..

# 开始SFT训练
python train.py
# 或
llamafactory-cli train train_config.yaml
```

### 5. 阶段三：DPO训练

```bash
# 编辑dpo_config.yaml，设置SFT checkpoint路径
# 开始DPO训练
llamafactory-cli train dpo_config.yaml
```

### 6. 模型评估

```bash
# 通用评估
llamafactory-cli train eval_config.yaml
python visualize_evaluation.py --results_dir ./evaluation_results_sft_checkpoint-2532

# IFEval评估
python evaluate_ifeval_with_llamafactory.py --use_lm_eval
```

## 文件命名规范

### 脚本文件

- 使用小写字母和下划线：`download_model.py`
- 功能明确：文件名应清楚表达脚本功能

### 配置文件

- 使用小写字母和下划线：`train_config.yaml`
- 类型明确：`*_config.yaml` 表示配置文件

### 文档文件

- 使用大写字母和下划线：`README.md`, `EVALUATION.md`
- 功能明确：文件名应清楚表达文档内容

## 维护建议

1. **定期清理**：删除不再使用的临时文件和脚本
2. **文档更新**：及时更新文档以反映代码变化
3. **版本控制**：使用git管理代码和配置
4. **备份重要结果**：定期备份训练检查点和评估结果

