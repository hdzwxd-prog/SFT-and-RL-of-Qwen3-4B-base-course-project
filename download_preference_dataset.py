#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载偏好数据集用于RL训练（DPO/KTO）
支持多个常用的偏好数据集
"""
import os
import sys
import time
import requests
from datasets import load_dataset
from huggingface_hub import snapshot_download

# 国内主流 Hugging Face 镜像列表
MIRROR_SITES = [
    {
        "name": "HF-Mirror (推荐)",
        "endpoint": "https://hf-mirror.com",
        "test_url": "https://hf-mirror.com/api/models",
        "env_var": "HF_ENDPOINT"
    },
    {
        "name": "ModelScope (阿里魔搭)",
        "endpoint": "https://www.modelscope.cn",
        "test_url": "https://www.modelscope.cn/api/v1/models",
        "env_var": "MODELSCOPE_ENDPOINT",
        "use_modelscope": True
    },
    {
        "name": "Gitee AI",
        "endpoint": "https://ai.gitee.com",
        "test_url": "https://ai.gitee.com",
        "env_var": "HF_ENDPOINT"
    },
    {
        "name": "WiseModel (始智AI)",
        "endpoint": "https://wisemodel.cn",
        "test_url": "https://wisemodel.cn",
        "env_var": "HF_ENDPOINT"
    },
    {
        "name": "官方源 (HuggingFace)",
        "endpoint": "https://huggingface.co",
        "test_url": "https://huggingface.co/api/models",
        "env_var": "HF_ENDPOINT"
    }
]

# 推荐的偏好数据集
RECOMMENDED_DATASETS = {
    "ultrafeedback": {
        "name": "UltraFeedback (英文，推荐)",
        "dataset_id": "llamafactory/ultrafeedback_binarized",
        "description": "高质量的英文偏好数据集，包含约6.4万条偏好对，适用于指令遵循任务（使用LLaMA Factory处理版本）",
        "size": "~64000",
        "language": "英文"
    },
    "orca_dpo": {
        "name": "Orca DPO Pairs (英文)",
        "dataset_id": "Intel/orca_dpo_pairs",
        "description": "Intel发布的Orca DPO偏好数据集，适合指令遵循训练",
        "size": "~100000",
        "language": "英文"
    },
    "dpo_en_zh": {
        "name": "DPO En-Zh Mixed (中英文混合)",
        "dataset_id": "hiyouga/DPO-En-Zh-20k",
        "description": "中英文混合的偏好数据集，包含约2万条偏好对",
        "size": "~20000",
        "language": "中英文"
    },
    "coig_p": {
        "name": "COIG-P (中文)",
        "dataset_id": "m-a-p/COIG-P",
        "description": "中文偏好数据集，适合中文指令遵循任务",
        "size": "~10000",
        "language": "中文"
    }
}

def test_mirror_speed(mirror, timeout=5):
    """测试镜像站点的访问速度"""
    try:
        start_time = time.time()
        response = requests.get(mirror["test_url"], timeout=timeout, allow_redirects=True)
        latency = time.time() - start_time
        if response.status_code in [200, 301, 302]:
            return True, latency
        else:
            return False, float('inf')
    except:
        return False, float('inf')

def find_fastest_mirror(verbose=True):
    """测试所有镜像站点，返回最快的可用镜像"""
    if verbose:
        print("=" * 60)
        print("正在测试镜像站点速度...")
        print("=" * 60)
    
    results = []
    for mirror in MIRROR_SITES:
        if verbose:
            print(f"测试 {mirror['name']} ({mirror['endpoint']})...", end=" ", flush=True)
        success, latency = test_mirror_speed(mirror)
        if success:
            results.append((mirror, latency))
            if verbose:
                print(f"✓ 可用 (延迟: {latency:.2f}秒)")
        else:
            if verbose:
                print("✗ 不可用或超时")
    
    if not results:
        if verbose:
            print("\n⚠ 所有镜像站点都不可用，将使用默认配置")
        return None
    
    results.sort(key=lambda x: x[1])
    best_mirror, best_latency = results[0]
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ 选择最快镜像: {best_mirror['name']}")
        print(f"  地址: {best_mirror['endpoint']}")
        print(f"  延迟: {best_latency:.2f}秒")
        print("=" * 60 + "\n")
    
    return best_mirror

def setup_mirror_environment(mirror):
    """设置环境变量以使用指定的镜像"""
    if mirror is None:
        return
    
    if "env_var" in mirror:
        os.environ[mirror["env_var"]] = mirror["endpoint"]
    
    if mirror.get("use_modelscope", False):
        os.environ["USE_MODELSCOPE_HUB"] = "1"
    else:
        if "USE_MODELSCOPE_HUB" in os.environ:
            del os.environ["USE_MODELSCOPE_HUB"]
        if mirror["endpoint"] != "https://huggingface.co":
            os.environ["HF_ENDPOINT"] = mirror["endpoint"]

def list_available_datasets():
    """列出可用的偏好数据集"""
    print("\n" + "=" * 80)
    print("推荐的偏好数据集（用于DPO/KTO训练）")
    print("=" * 80)
    print(f"{'ID':<15} {'名称':<35} {'语言':<10} {'大小':<15} {'说明'}")
    print("-" * 80)
    for key, info in RECOMMENDED_DATASETS.items():
        print(f"{key:<15} {info['name']:<35} {info['language']:<10} {info['size']:<15} {info['description']}")
    print("=" * 80 + "\n")

def download_preference_dataset(dataset_key="ultrafeedback", auto_mirror=True):
    """
    下载偏好数据集
    
    Args:
        dataset_key: 数据集键名（从RECOMMENDED_DATASETS中选择）
        auto_mirror: 是否自动选择最快镜像
    """
    if dataset_key not in RECOMMENDED_DATASETS:
        print(f"❌ 错误: 未知的数据集键 '{dataset_key}'")
        print("\n可用的数据集键:")
        for key in RECOMMENDED_DATASETS.keys():
            print(f"  - {key}")
        return
    
    dataset_info = RECOMMENDED_DATASETS[dataset_key]
    dataset_id = dataset_info["dataset_id"]
    
    # 自动选择最快镜像
    selected_mirror = None
    if auto_mirror:
        print("正在自动选择最快的下载镜像...\n")
        selected_mirror = find_fastest_mirror(verbose=True)
        if selected_mirror:
            setup_mirror_environment(selected_mirror)
            print(f"✓ 使用镜像: {selected_mirror['name']} ({selected_mirror['endpoint']})\n")
        else:
            print("⚠ 无法连接到任何镜像，将使用默认配置（可能较慢）\n")
    
    print(f"数据集信息:")
    print(f"  名称: {dataset_info['name']}")
    print(f"  ID: {dataset_id}")
    print(f"  语言: {dataset_info['language']}")
    print(f"  大小: {dataset_info['size']} 条偏好对")
    print(f"  说明: {dataset_info['description']}")
    print()
    
    try:
        print(f"正在从 HuggingFace 加载数据集: {dataset_id}")
        print("  （数据集会自动缓存到本地）")
        
        # 加载数据集（会自动缓存）
        dataset = load_dataset(dataset_id)
        
        print(f"\n✓ 数据集加载完成!")
        print(f"\n数据集结构:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条数据")
            if len(split_data) > 0:
                print(f"    列名: {list(split_data[0].keys())}")
        
        print(f"\n✓ 数据集已缓存到本地，可以在配置文件中使用以下ID:")
        print(f"  {dataset_id}")
        print(f"\n提示: 请确保在 data/dataset_info.json 中正确配置了数据集格式")
        
    except Exception as e:
        print(f"\n❌ 下载数据集时出错: {e}")
        print("\n请确保:")
        print("1. 已安装 datasets: pip install datasets")
        print("2. 已安装 huggingface_hub: pip install huggingface_hub")
        print("3. 如果需要，已登录 HuggingFace: huggingface-cli login")
        print("4. 网络连接正常")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            list_available_datasets()
        else:
            dataset_key = sys.argv[1]
            auto_mirror = '--no-mirror' not in sys.argv
            download_preference_dataset(dataset_key, auto_mirror=auto_mirror)
    else:
        print("使用方法:")
        print("  python download_preference_dataset.py <dataset_key>")
        print("\n例如:")
        print("  python download_preference_dataset.py ultrafeedback")
        print("\n查看可用数据集:")
        print("  python download_preference_dataset.py --list")
        print("\n推荐的第一个数据集:")
        print("  python download_preference_dataset.py ultrafeedback")
        list_available_datasets()

