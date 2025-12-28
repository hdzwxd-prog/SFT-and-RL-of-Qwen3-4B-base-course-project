import os
import json
import subprocess
from datasets import load_dataset
from huggingface_hub import snapshot_download
from datetime import datetime

# ==========================================
# 1. 自动环境配置 (相对路径 + 镜像)
# ==========================================
CURRENT_DIR = os.getcwd()
CACHE_DIR = os.path.join(CURRENT_DIR, "dataset_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 开启镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# 依据 MLSYS 项目说明书定义的 5 大集群 [cite: 30]
DATASET_CONFIGS = {
    "Academic Knowledge": [
        {"name": "MMLU", "path": "cais/mmlu", "config": "all", "split": "test"},
        {"name": "ARC-Challenge", "path": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test"}
    ],
    "Commonsense Reasoning": [
        {"name": "HellaSwag", "path": "Rowan/hellaswag", "config": None, "split": "validation"},
        {"name": "WinoGrande", "path": "winogrande", "config": "winogrande_xl", "split": "validation"},
        {"name": "CSQA", "path": "tau/commonsense_qa", "config": None, "split": "validation"}
    ],
    "Physical Reasoning": [
        {"name": "PIQA", "path": "piqa", "config": None, "split": "validation"}
    ],
    "Math & Logic": [
        {"name": "GSM8K", "path": "openai/gsm8k", "config": "main", "split": "test"}
    ],
    "Instruction Following": [
        {"name": "IFEval", "path": "google/IFEval", "config": None, "split": "train"}
    ]
}

def smart_download(path, config, name):
    """
    智能下载函数：尝试标准加载，失败则切换到快照克隆模式
    """
    try:
        # 尝试标准加载
        return load_dataset(path, config, cache_dir=CACHE_DIR, trust_remote_code=True)
    except Exception as e:
        if "scripts are no longer supported" in str(e) or "piqa.py" in str(e):
            print(f"\n  [检测到脚本限制] 正在为 {name} 启动强制快照下载模式...", end="")
            # 嵌入命令行逻辑的 Python 等价实现
            local_dir = os.path.join(CACHE_DIR, path.split('/')[-1])
            snapshot_download(
                repo_id=path,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                endpoint="https://hf-mirror.com"
            )
            # 下载后再次尝试加载本地路径
            return load_dataset(local_dir, config, cache_dir=CACHE_DIR, trust_remote_code=True)
        else:
            raise e

def run_project1_setup():
    detailed_report = {}
    print(f"[{datetime.now()}] 启动全量自动化下载 (嵌入快照下载逻辑)...")
    
    for cluster, ds_list in DATASET_CONFIGS.items():
        print(f"\n--- 正在处理集群: {cluster} ---")
        detailed_report[cluster] = []
        
        for item in ds_list:
            print(f"  任务: {item['name']} ({item['path']}) ", end="", flush=True)
            try:
                ds = smart_download(item['path'], item['config'], item['name'])
                
                # 验证样本数量
                sample_count = len(ds[item['split']])
                preview = str(ds[item['split']][0])[:100] + "..."
                
                print(f" [成功 ✅] 样本数: {sample_count}")
                detailed_report[cluster].append({
                    "name": item['name'],
                    "status": "Success",
                    "samples": sample_count,
                    "preview": preview
                })
            except Exception as e:
                print(f" [最终失败 ❌] {str(e)[:100]}")
                detailed_report[cluster].append({"name": item['name'], "status": f"Error: {str(e)}"})

    # 保存报告
    with open("detailed_download_report.json", "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, indent=4, ensure_ascii=False)
    print(f"\n[{datetime.now()}] 全量任务完成。报告已保存。")

if __name__ == "__main__":
    # 确保安装了必要库：pip install huggingface_hub datasets
    run_project1_setup()