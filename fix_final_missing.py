import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

# 配置路径 (必须与 eval_stage1_v3.py 一致)
PROJECT_ROOT = os.getcwd()
CACHE_DIR = os.path.join(PROJECT_ROOT, "dataset_cache")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# 这里的列表不包含 MMLU，因为 MMLU 你已经修好了
MISSING_TARGETS = [
    {"name": "ARC", "path": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test"},
    # {"name": "HellaSwag", "path": "Rowan/hellaswag", "config": None, "split": "validation"},
    {"name": "IFEval", "path": "google/IFEval", "config": None, "split": "train"},
    {"name": "PIQA", "path": "piqa", "config": None, "split": "validation"}
]

print(f"🚀 启动缺口补全 (ARC/HellaSwag/IFEval/PIQA)...")
print(f"📂 缓存路径: {CACHE_DIR}")

for item in MISSING_TARGETS:
    print(f"\n👉 正在处理: {item['name']} ... ", end="", flush=True)
    try:
        # 特殊处理 PIQA：如果常规下载失败，用快照
        if item['name'] == "PIQA":
            try:
                load_dataset(item['path'], cache_dir=CACHE_DIR, trust_remote_code=True)
            except:
                print("   [切换快照下载]...", end="", flush=True)
                local_dir = os.path.join(CACHE_DIR, "piqa")
                snapshot_download(repo_id="piqa", repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False)
                load_dataset(local_dir, cache_dir=CACHE_DIR, trust_remote_code=True)
        else:
            # 常规下载
            load_dataset(item['path'], item['config'], cache_dir=CACHE_DIR, trust_remote_code=True)
        
        print("✅ 成功")
    except Exception as e:
        print(f"❌ 失败: {str(e)[:100]}")

print("\n🎉 补全任务结束！请继续下一步。")