# 模型评估总结报告

生成时间: 2025-12-27 16:56:34

---

## 评估指标

### 评估集指标

- **bleu-4**: 32.00%
- **rouge-1**: 38.09%
- **rouge-2**: 19.00%
- **rouge-l**: 20.49%

### 预测集指标

- **bleu-4**: 33.05%
- **rouge-1**: 38.42%
- **rouge-2**: 19.02%
- **rouge-l**: 20.52%

## 预测统计

- **总样本数**: 100

### 文本长度统计

- **预测平均长度**: 341.1 词
- **参考平均长度**: 240.0 词
- **平均长度比例**: 5.16

## 输出文件

评估生成的文件包括:

- `comprehensive_metrics.png`: 综合指标可视化
- `predictions_analysis.png`: 预测结果分析
- `evaluation_summary.md`: 本报告
- `examples.md`: 预测示例（如果存在）
- `examples.json`: 预测示例JSON格式（如果存在）
- `generated_predictions.jsonl`: 完整预测结果
- `eval_results.json`: 评估指标JSON

## 训练过程

DPO训练过程图表请查看:
- `saves/qwen3-4b-base/lora/dpo/training_loss.png`: 训练损失曲线
- `saves/qwen3-4b-base/lora/dpo/training_rewards_accuracies.png`: 奖励和准确率曲线

