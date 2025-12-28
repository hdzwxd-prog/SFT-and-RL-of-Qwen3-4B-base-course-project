#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本
使用 LLaMA Factory 对微调后的模型进行全面评估
包括测试集评估、示例展示和可视化分析
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

# Set English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')


def load_model_and_tokenizer(checkpoint_path: str, base_model_path: str):
    """加载模型和tokenizer"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print(f"加载基础模型: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"加载LoRA适配器: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用 LLaMA Factory 的方式加载...")
        # 如果直接加载失败，可以使用 LLaMA Factory 的 API
        return None, None


def generate_predictions(model, tokenizer, test_dataset, max_samples: int = 100, 
                        max_new_tokens: int = 512, temperature: float = 0.7):
    """生成预测结果"""
    print(f"\n开始生成预测，最多处理 {max_samples} 个样本...")
    
    predictions = []
    references = []
    inputs = []
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(tqdm(test_dataset[:max_samples], desc="生成预测")):
            # 提取输入消息
            messages = example.get('messages', [])
            if not messages:
                continue
            
            # 构建输入（只包含用户消息，不包括助手回复）
            input_messages = messages[:-1] if len(messages) > 1 else messages
            
            # 获取参考输出（最后一个助手的回复）
            reference = messages[-1]['content'] if messages[-1]['role'] == 'assistant' else ""
            
            # 使用模板格式化输入
            try:
                # 构建提示文本
                prompt_parts = []
                for msg in input_messages:
                    if msg['role'] == 'user':
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'system':
                        prompt_parts.append(f"System: {msg['content']}")
                
                prompt = "\n\n".join(prompt_parts) + "\n\nAssistant: "
                
                # Tokenize 输入
                inputs_tensor = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs_tensor = {k: v.to(model.device) for k, v in inputs_tensor.items()}
                
                # 生成回复
                outputs = model.generate(
                    **inputs_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # 解码生成结果
                generated_text = tokenizer.decode(outputs[0][inputs_tensor['input_ids'].shape[1]:], skip_special_tokens=True)
                
                predictions.append(generated_text)
                references.append(reference)
                inputs.append(prompt)
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
    
    return predictions, references, inputs


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算评估指标"""
    print("\n计算评估指标...")
    
    metrics = {}
    
    # 1. BLEU 分数
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method3
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if ref.strip():
                score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)
                bleu_scores.append(score)
        
        metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
        metrics['bleu_std'] = np.std(bleu_scores) if bleu_scores else 0.0
    except ImportError:
        print("警告: nltk 未安装，跳过 BLEU 计算")
        metrics['bleu'] = 0.0
    
    # 2. ROUGE 分数
    try:
        from rouge_score import rouge_scorer
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            scores = rouge_scorer_obj.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        metrics['rouge1'] = np.mean(rouge_scores['rouge1'])
        metrics['rouge2'] = np.mean(rouge_scores['rouge2'])
        metrics['rougeL'] = np.mean(rouge_scores['rougeL'])
    except ImportError:
        print("警告: rouge_score 未安装，跳过 ROUGE 计算")
        metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
    
    # 3. 长度统计
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    metrics['avg_pred_length'] = np.mean(pred_lengths)
    metrics['avg_ref_length'] = np.mean(ref_lengths)
    metrics['length_ratio'] = metrics['avg_pred_length'] / metrics['avg_ref_length'] if metrics['avg_ref_length'] > 0 else 0.0
    
    # 4. 词汇多样性（唯一词比例）
    all_pred_words = []
    all_ref_words = []
    for pred, ref in zip(predictions, references):
        all_pred_words.extend(pred.split())
        all_ref_words.extend(ref.split())
    
    metrics['pred_vocab_diversity'] = len(set(all_pred_words)) / len(all_pred_words) if all_pred_words else 0.0
    metrics['ref_vocab_diversity'] = len(set(all_ref_words)) / len(all_ref_words) if all_ref_words else 0.0
    
    return metrics


def visualize_results(metrics: Dict[str, float], predictions: List[str], 
                     references: List[str], output_dir: str):
    """创建可视化图表"""
    print("\n生成可视化图表...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # BLEU 和 ROUGE 分数
    ax1 = axes[0, 0]
    metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    metric_values = [
        metrics.get('bleu', 0) * 100,
        metrics.get('rouge1', 0) * 100,
        metrics.get('rouge2', 0) * 100,
        metrics.get('rougeL', 0) * 100
    ]
    bars = ax1.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('BLEU and ROUGE Scores', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 长度分布
    ax2 = axes[0, 1]
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    ax2.hist([pred_lengths, ref_lengths], bins=30, alpha=0.7, 
             label=['Predicted', 'Reference'], color=['#3498db', '#e74c3c'])
    ax2.set_xlabel('Length (words)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Response Length Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 长度比例分布
    ax3 = axes[1, 0]
    length_ratios = [len(p.split())/len(r.split()) if len(r.split()) > 0 else 0 
                     for p, r in zip(predictions, references)]
    ax3.hist(length_ratios, bins=30, color='#9b59b6', alpha=0.7)
    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Ratio (1.0)')
    ax3.set_xlabel('Length Ratio (Predicted / Reference)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Length Ratio Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 词汇多样性对比
    ax4 = axes[1, 1]
    diversity_data = [metrics.get('pred_vocab_diversity', 0) * 100, 
                     metrics.get('ref_vocab_diversity', 0) * 100]
    bars = ax4.bar(['Predicted', 'Reference'], diversity_data, 
                   color=['#3498db', '#e74c3c'])
    ax4.set_ylabel('Vocabulary Diversity (%)', fontsize=12)
    ax4.set_title('Vocabulary Diversity Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 指标图表已保存: {output_dir}/evaluation_metrics.png")
    
    # 2. 指标汇总表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['BLEU Score', f"{metrics.get('bleu', 0)*100:.2f}%"],
        ['ROUGE-1 F1', f"{metrics.get('rouge1', 0)*100:.2f}%"],
        ['ROUGE-2 F1', f"{metrics.get('rouge2', 0)*100:.2f}%"],
        ['ROUGE-L F1', f"{metrics.get('rougeL', 0)*100:.2f}%"],
        ['Avg Predicted Length', f"{metrics.get('avg_pred_length', 0):.1f} words"],
        ['Avg Reference Length', f"{metrics.get('avg_ref_length', 0):.1f} words"],
        ['Length Ratio', f"{metrics.get('length_ratio', 0):.2f}"],
        ['Predicted Vocab Diversity', f"{metrics.get('pred_vocab_diversity', 0)*100:.2f}%"],
        ['Reference Vocab Diversity', f"{metrics.get('ref_vocab_diversity', 0)*100:.2f}%"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Evaluation Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'metrics_summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 指标汇总表已保存: {output_dir}/metrics_summary_table.png")
    
    plt.close('all')


def save_examples(predictions: List[str], references: List[str], inputs: List[str],
                 output_dir: str, num_examples: int = 20):
    """保存示例输入输出"""
    print(f"\n保存 {num_examples} 个示例...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    examples = []
    for i in range(min(num_examples, len(predictions))):
        examples.append({
            'index': i + 1,
            'input': inputs[i],
            'reference': references[i],
            'prediction': predictions[i],
            'reference_length': len(references[i].split()),
            'prediction_length': len(predictions[i].split()),
        })
    
    # 保存为 JSON
    with open(os.path.join(output_dir, 'examples.json'), 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    # 保存为 Markdown（更易读）
    md_content = "# Model Evaluation Examples\n\n"
    md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"
    
    for ex in examples:
        md_content += f"## Example {ex['index']}\n\n"
        md_content += "### Input:\n```\n"
        md_content += ex['input'] + "\n```\n\n"
        md_content += "### Reference:\n```\n"
        md_content += ex['reference'] + "\n```\n\n"
        md_content += "### Prediction:\n```\n"
        md_content += ex['prediction'] + "\n```\n\n"
        md_content += f"**Length:** Reference: {ex['reference_length']} words, "
        md_content += f"Prediction: {ex['prediction_length']} words\n\n"
        md_content += "---\n\n"
    
    with open(os.path.join(output_dir, 'examples.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ 示例已保存: {output_dir}/examples.json 和 {output_dir}/examples.md")


def main():
    parser = argparse.ArgumentParser(description='评估微调后的模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径 (例如: saves/qwen3-4b-base/lora/sft/checkpoint-2000)')
    parser.add_argument('--base_model', type=str, default='./Qwen3-4B-Base',
                       help='基础模型路径')
    parser.add_argument('--dataset', type=str, default='tulu3_sft_personas',
                       help='测试数据集名称')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='最大评估样本数')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("模型评估脚本")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"基础模型: {args.base_model}")
    print(f"数据集: {args.dataset}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 加载测试数据集
    print("\n加载测试数据集...")
    try:
        # 尝试从本地加载
        dataset = load_dataset('./tulu-3-sft-personas-instruction-following')
        test_dataset = dataset.get('train', dataset[list(dataset.keys())[0]])
        # 如果数据集很大，可以取一部分作为测试集
        if len(test_dataset) > args.max_samples:
            test_dataset = test_dataset.shuffle(seed=42).select(range(args.max_samples))
        print(f"✓ 加载了 {len(test_dataset)} 个测试样本")
    except Exception as e:
        print(f"从本地加载数据集失败: {e}")
        print("尝试从 HuggingFace 加载...")
        dataset = load_dataset('allenai/tulu-3-sft-personas-instruction-following')
        test_dataset = dataset.get('train', dataset[list(dataset.keys())[0]])
        if len(test_dataset) > args.max_samples:
            test_dataset = test_dataset.shuffle(seed=42).select(range(args.max_samples))
        print(f"✓ 加载了 {len(test_dataset)} 个测试样本")
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.base_model)
    
    if model is None or tokenizer is None:
        print("\n⚠ 无法直接加载模型，请使用 LLaMA Factory 的推理功能")
        print("建议使用以下命令进行推理：")
        print(f"  llamafactory-cli chat --model_name_or_path {args.base_model} \\")
        print(f"    --adapter_name_or_path {args.checkpoint} \\")
        print(f"    --template qwen3")
        print("\n或者使用训练配置进行评估：")
        print("  在 train_config.yaml 中设置 do_eval: true")
        return
    
    # 生成预测
    predictions, references, inputs = generate_predictions(
        model, tokenizer, test_dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    if not predictions:
        print("⚠ 未生成任何预测结果")
        return
    
    print(f"✓ 生成了 {len(predictions)} 个预测")
    
    # 计算指标
    metrics = compute_metrics(predictions, references)
    
    # 保存指标
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ 指标已保存: {metrics_file}")
    
    # 打印指标摘要
    print("\n" + "=" * 60)
    print("评估指标摘要")
    print("=" * 60)
    print(f"BLEU Score:        {metrics.get('bleu', 0)*100:.2f}%")
    print(f"ROUGE-1 F1:        {metrics.get('rouge1', 0)*100:.2f}%")
    print(f"ROUGE-2 F1:        {metrics.get('rouge2', 0)*100:.2f}%")
    print(f"ROUGE-L F1:        {metrics.get('rougeL', 0)*100:.2f}%")
    print(f"平均预测长度:      {metrics.get('avg_pred_length', 0):.1f} words")
    print(f"平均参考长度:      {metrics.get('avg_ref_length', 0):.1f} words")
    print(f"长度比例:          {metrics.get('length_ratio', 0):.2f}")
    print("=" * 60)
    
    # 创建可视化
    visualize_results(metrics, predictions, references, output_dir)
    
    # 保存示例
    save_examples(predictions, references, inputs, output_dir, num_examples=20)
    
    print(f"\n✓ 评估完成！所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

