#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFEval评估脚本 - 使用LLaMA Factory风格但直接处理IFEval
避免lm_eval的PEFT兼容性问题
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# Default paths
BASE_MODEL_PATH = "./Qwen3-4B-Base"
SFT_CHECKPOINT = "./saves/qwen3-4b-base/lora/sft/checkpoint-2532"
DPO_CHECKPOINT = "./saves/qwen3-4b-base/lora/dpo/checkpoint-1875"
STAGE1_CACHE_DIR = "./MLS_project_stage1/dataset_cache"
OUTPUT_DIR = "./instruction_following_evaluation"


def load_model_with_peft(base_model_path: str, adapter_path: Optional[str], device_map: str = "auto"):
    """Load model with PEFT adapter"""
    print(f"Loading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"Loading PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def load_ifeval_dataset(cache_dir: str):
    """Load IFEval dataset from Stage1 cache"""
    cache_dir_abs = os.path.abspath(cache_dir)
    
    # Set cache directory
    os.environ["HF_DATASETS_CACHE"] = cache_dir_abs
    os.environ["HF_HOME"] = cache_dir_abs
    
    print(f"Loading IFEval dataset from cache: {cache_dir_abs}")
    
    try:
        dataset = load_dataset("google/IFEval", cache_dir=cache_dir_abs, trust_remote_code=True)
        print(f"Dataset loaded: {dataset}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def generate_predictions(model, tokenizer, dataset, max_samples: Optional[int] = None, max_new_tokens: int = 512):
    """Generate predictions on IFEval dataset"""
    predictions = []
    
    # Get the split (usually 'train')
    split = list(dataset.keys())[0]
    eval_data = dataset[split]
    
    if max_samples:
        eval_data = eval_data.select(range(min(max_samples, len(eval_data))))
    
    print(f"Generating predictions for {len(eval_data)} samples...")
    
    for i, item in enumerate(tqdm(eval_data)):
        prompt = item.get('prompt', '')
        
        # Format prompt with chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                # IFEval prompts are usually single-turn
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        if formatted_prompt in generated_text:
            prediction = generated_text[len(formatted_prompt):].strip()
        else:
            # If template formatting changed the prompt, try to extract
            prediction = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        predictions.append({
            'key': item.get('key', i),
            'prompt': prompt,
            'prediction': prediction,
            'instruction_id_list': item.get('instruction_id_list', []),
            'kwargs': item.get('kwargs', [])
        })
    
    return predictions


def evaluate_ifeval_with_lm_eval(base_model_path: str, adapter_path: Optional[str], cache_dir: str, max_samples: Optional[int] = None) -> Optional[Dict]:
    """
    Evaluate IFEval using lm_eval (same method as Stage1)
    Returns the evaluation results dictionary
    """
    try:
        import lm_eval
    except ImportError:
        print("Warning: lm_eval not found. Cannot compute IFEval scores.")
        print("Install with: pip install lm-eval")
        return None
    
    # Set cache directory
    cache_dir_abs = os.path.abspath(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = cache_dir_abs
    os.environ["HF_HOME"] = cache_dir_abs
    
    print(f"\n使用 lm_eval 评估 IFEval...")
    print(f"数据集缓存: {cache_dir_abs}")
    
    # Construct model arguments
    if adapter_path:
        model_args = f"pretrained={base_model_path},peft={adapter_path},device_map=auto"
    else:
        model_args = f"pretrained={base_model_path},device_map=auto,torch_dtype=bfloat16"
    
    try:
        # Use lm_eval.simple_evaluate (same as Stage1)
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=["ifeval"],
            num_fewshot=0,
            batch_size=8,
            log_samples=True,
            limit=max_samples  # Limit samples if specified
        )
        
        return results
    except Exception as e:
        print(f"lm_eval 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_ifeval_score(results: Dict) -> Optional[float]:
    """Extract IFEval prompt_level_strict_acc score from lm_eval results"""
    try:
        if 'results' in results and 'ifeval' in results['results']:
            ifeval_results = results['results']['ifeval']
            # Try to get prompt_level_strict_acc (same as Stage1)
            score = ifeval_results.get("prompt_level_strict_acc,none") or \
                   ifeval_results.get("acc_norm,none") or \
                   ifeval_results.get("acc,none")
            return score
    except Exception as e:
        print(f"提取 IFEval 分数时出错: {e}")
    
    return None


def evaluate_ifeval_predictions(predictions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate IFEval predictions - compute basic statistics
    Note: For actual IFEval scores, use evaluate_ifeval_with_lm_eval()
    """
    print("\n计算基本统计信息...")
    
    stats = {
        'total_samples': len(predictions),
        'avg_prediction_length': np.mean([len(p.get('prediction', '').split()) for p in predictions]),
        'avg_prompt_length': np.mean([len(p.get('prompt', '').split()) for p in predictions])
    }
    
    return stats


def create_comparison_visualization(results: Dict[str, Dict], output_dir: str):
    """Create comparison visualization with IFEval scores"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(results.keys())
    
    # Check if we have IFEval scores
    has_scores = any('ifeval_score' in results[name] for name in model_names)
    
    if has_scores:
        # Create figure with score comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. IFEval Score comparison
        ax1 = axes[0]
        scores = [results[name].get('ifeval_score', 0) * 100 for name in model_names]
        bars = ax1.bar(model_names, scores, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('IFEval Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('IFEval Score Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylim([0, max(scores) * 1.2 if scores else 100])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Sample count
        ax2 = axes[1]
        counts = [results[name].get('total_samples', 0) for name in model_names]
        bars2 = ax2.bar(model_names, counts, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars2, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Evaluation Sample Count', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'ifeval_score_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nIFEval 评分对比图已保存: {output_file}")
        plt.close()
    else:
        # Fallback to basic comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Prediction length comparison
        ax1 = axes[0]
        lengths = [results[name].get('avg_prediction_length', 0) for name in model_names]
        bars = ax1.bar(model_names, lengths, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, lengths):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Average Prediction Length (words)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Prediction Length Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Sample count
        ax2 = axes[1]
        counts = [results[name].get('total_samples', 0) for name in model_names]
        bars2 = ax2.bar(model_names, counts, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars2, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Evaluation Sample Count', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'ifeval_basic_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n基本对比图已保存: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on IFEval dataset')
    parser.add_argument('--sft_checkpoint', type=str, default=SFT_CHECKPOINT,
                       help='Path to SFT checkpoint adapter')
    parser.add_argument('--dpo_checkpoint', type=str, default=DPO_CHECKPOINT,
                       help='Path to DPO checkpoint adapter')
    parser.add_argument('--base_model', type=str, default=BASE_MODEL_PATH,
                       help='Path to base model')
    parser.add_argument('--cache_dir', type=str, default=STAGE1_CACHE_DIR,
                       help='Dataset cache directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for evaluation results')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to evaluate (for faster testing)')
    parser.add_argument('--eval_sft_only', action='store_true',
                       help='Only evaluate SFT model')
    parser.add_argument('--eval_dpo_only', action='store_true',
                       help='Only evaluate DPO model')
    parser.add_argument('--use_lm_eval', action='store_true',
                       help='Use lm_eval to compute IFEval scores (same as Stage1)')
    parser.add_argument('--skip_predictions', action='store_true',
                       help='Skip generating predictions, only run lm_eval evaluation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("IFEval Evaluation (Direct Model Loading)")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"SFT Checkpoint: {args.sft_checkpoint}")
    print(f"DPO Checkpoint: {args.dpo_checkpoint}")
    print(f"Cache Directory: {args.cache_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Samples: {args.max_samples}")
    print("=" * 60)
    
    # Load dataset
    dataset = load_ifeval_dataset(args.cache_dir)
    if dataset is None:
        print("Failed to load IFEval dataset")
        return
    
    results = {}
    
    # Evaluate SFT model
    if not args.eval_dpo_only:
        print("\n" + "=" * 60)
        print("评估 SFT 模型")
        print("=" * 60)
        
        sft_stats = {}
        
        # Generate predictions if not skipping
        if not args.skip_predictions:
            model, tokenizer = load_model_with_peft(args.base_model, args.sft_checkpoint)
            
            sft_predictions = generate_predictions(model, tokenizer, dataset, args.max_samples)
            
            basic_stats = evaluate_ifeval_predictions(sft_predictions)
            sft_stats.update(basic_stats)
            
            # Save predictions
            sft_output_file = os.path.join(args.output_dir, 'sft_ifeval_predictions.json')
            with open(sft_output_file, 'w', encoding='utf-8') as f:
                json.dump(sft_predictions, f, indent=2, ensure_ascii=False)
            print(f"SFT 预测结果已保存: {sft_output_file}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # Use lm_eval to compute IFEval score
        if args.use_lm_eval:
            print("\n使用 lm_eval 计算 IFEval 评分...")
            lm_eval_results = evaluate_ifeval_with_lm_eval(
                args.base_model, 
                args.sft_checkpoint, 
                args.cache_dir,
                args.max_samples
            )
            
            if lm_eval_results:
                # Save full lm_eval results
                sft_lm_eval_file = os.path.join(args.output_dir, 'sft_ifeval_lm_eval_results.json')
                with open(sft_lm_eval_file, 'w', encoding='utf-8') as f:
                    json.dump(lm_eval_results, f, indent=2, ensure_ascii=False)
                print(f"SFT lm_eval 完整结果已保存: {sft_lm_eval_file}")
                
                # Extract score
                score = extract_ifeval_score(lm_eval_results)
                if score is not None:
                    sft_stats['ifeval_score'] = score
                    print(f"SFT IFEval 评分: {score*100:.2f}%")
                else:
                    print("警告: 无法从 lm_eval 结果中提取 IFEval 评分")
        
        results['SFT'] = sft_stats
    
    # Evaluate DPO model
    if not args.eval_sft_only:
        print("\n" + "=" * 60)
        print("评估 DPO 模型")
        print("=" * 60)
        
        dpo_stats = {}
        
        # Generate predictions if not skipping
        if not args.skip_predictions:
            model, tokenizer = load_model_with_peft(args.base_model, args.dpo_checkpoint)
            
            dpo_predictions = generate_predictions(model, tokenizer, dataset, args.max_samples)
            
            basic_stats = evaluate_ifeval_predictions(dpo_predictions)
            dpo_stats.update(basic_stats)
            
            # Save predictions
            dpo_output_file = os.path.join(args.output_dir, 'dpo_ifeval_predictions.json')
            with open(dpo_output_file, 'w', encoding='utf-8') as f:
                json.dump(dpo_predictions, f, indent=2, ensure_ascii=False)
            print(f"DPO 预测结果已保存: {dpo_output_file}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # Use lm_eval to compute IFEval score
        if args.use_lm_eval:
            print("\n使用 lm_eval 计算 IFEval 评分...")
            lm_eval_results = evaluate_ifeval_with_lm_eval(
                args.base_model, 
                args.dpo_checkpoint, 
                args.cache_dir,
                args.max_samples
            )
            
            if lm_eval_results:
                # Save full lm_eval results
                dpo_lm_eval_file = os.path.join(args.output_dir, 'dpo_ifeval_lm_eval_results.json')
                with open(dpo_lm_eval_file, 'w', encoding='utf-8') as f:
                    json.dump(lm_eval_results, f, indent=2, ensure_ascii=False)
                print(f"DPO lm_eval 完整结果已保存: {dpo_lm_eval_file}")
                
                # Extract score
                score = extract_ifeval_score(lm_eval_results)
                if score is not None:
                    dpo_stats['ifeval_score'] = score
                    print(f"DPO IFEval 评分: {score*100:.2f}%")
                else:
                    print("警告: 无法从 lm_eval 结果中提取 IFEval 评分")
        
        results['DPO'] = dpo_stats
    
    # Create visualizations
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Creating Visualizations")
        print("=" * 60)
        create_comparison_visualization(results, args.output_dir)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'ifeval_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    # Print summary
    if results:
        print("\n评估结果摘要:")
        for model_name, stats in results.items():
            print(f"\n{model_name}:")
            if 'ifeval_score' in stats:
                print(f"  IFEval 评分: {stats['ifeval_score']*100:.2f}%")
            print(f"  样本数: {stats.get('total_samples', 'N/A')}")
            if 'avg_prediction_length' in stats:
                print(f"  平均预测长度: {stats['avg_prediction_length']:.1f} 词")
    
    if not args.use_lm_eval:
        print("\n提示: 使用 --use_lm_eval 参数可以获得与 Stage1 相同的 IFEval 评分")
        print("      (prompt_level_strict_acc)")
    
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()

