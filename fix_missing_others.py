import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

# 配置路径 (必须与之前一致)
PROJECT_ROOT = os.getcwd()
CACHE_DIR = os.path.join(PROJECT_ROOT, "dataset_cache")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# 需要补齐的缺失数据集
MISSING_DATASETS = [
    # 学术知识
    {"name": "ARC-Challenge", "path": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test"},
    # 常识
    {"name": "HellaSwag", "path": "Rowan/hellaswag", "config": None, "split": "validation"},
    # 物理 (可能需要快照下载)
    {"name": "PIQA", "path": "piqa", "config": None, "split": "validation"},
    # 指令遵循
    {"name": "IFEval", "path": "google/IFEval", "config": None, "split": "train"}
]

def fix_others():
    print("🚀 开始补齐 ARC, HellaSwag, PIQA, IFEval...")
    print(f"📂 目标缓存: {CACHE_DIR}\n")
    
    for item in MISSING_DATASETS:
        print(f"正在下载: {item['name']} ... ", end="", flush=True)
        try:
            # 尝试常规下载
            load_dataset(item['path'], item['config'], cache_dir=CACHE_DIR, trust_remote_code=True)
            print("✅ 成功")
        except Exception as e:
            # PIQA 特殊处理
            if "piqa" in item['path']:
                print("\n   ⚠️  触发 PIQA 快照修复...", end="")
                try:
                    local_dir = os.path.join(CACHE_DIR, "piqa")
                    snapshot_download(repo_id="piqa", repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False)
                    load_dataset(local_dir, cache_dir=CACHE_DIR, trust_remote_code=True)
                    print("✅ 成功")
                except Exception as e2:
                    print(f"❌ PIQA 最终失败: {e2}")
            else:
                print(f"❌ 失败: {str(e)[:100]}")

if __name__ == "__main__":
    fix_others()