#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Tulu-3-sft-personas-instruction-following 数据集到当前目录
支持自动选择最快的镜像源
"""
import os
import sys
import time
import requests
from datasets import load_dataset

# 国内主流 Hugging Face 镜像列表（与download_model.py保持一致）
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

def check_dataset_completeness(local_dir, dataset_name=None, min_expected_size=10):
    """
    检查数据集是否完整
    
    Args:
        local_dir: 数据集目录路径
        dataset_name: 原始数据集名称（用于获取预期大小）
        min_expected_size: 最小预期数据量（默认至少10条）
    
    Returns:
        (is_complete, actual_size, expected_size): (是否完整, 实际数据量, 预期数据量)
    """
    try:
        # 尝试加载本地数据集
        local_dataset = load_dataset(local_dir)
        
        # 检查每个split的数据量
        total_size = 0
        for split_name, split_data in local_dataset.items():
            split_size = len(split_data)
            total_size += split_size
            print(f"  {split_name}: {split_size} 条数据")
        
        # 如果数据量太少，认为不完整
        if total_size < min_expected_size:
            print(f"⚠ 数据集数据量过少 ({total_size} 条)，可能下载不完整")
            return False, total_size, None
        
        # 如果提供了数据集名称，尝试获取官方数据量进行对比
        expected_size = None
        if dataset_name:
            try:
                # 只获取数据集信息，不下载
                from datasets import get_dataset_config_names, get_dataset_infos
                # 尝试获取数据集信息
                info = load_dataset(dataset_name, split='train', streaming=True)
                # 对于streaming数据集，无法直接获取大小，需要其他方法
                # 先返回True，让下载过程继续验证
            except:
                pass
        
        return True, total_size, expected_size
        
    except Exception as e:
        print(f"  加载数据集失败: {e}")
        return False, 0, None

def download_dataset(dataset_name="allenai/tulu-3-sft-personas-instruction-following", 
                     local_dir="./tulu-3-sft-personas-instruction-following",
                     auto_mirror=True):
    """
    下载数据集到本地目录，支持自动选择最快镜像
    
    Args:
        dataset_name: HuggingFace数据集名称
        local_dir: 本地保存目录
        auto_mirror: 是否自动选择最快镜像（默认True）
    """
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
    
    print(f"开始下载数据集: {dataset_name}")
    print(f"保存位置: {os.path.abspath(local_dir)}")
    
    # 如果目录已存在，检查是否完整
    import sys
    force_download = '--force' in sys.argv
    
    if os.path.exists(local_dir):
        print("检查数据集完整性...")
        is_complete, actual_size, expected_size = check_dataset_completeness(
            local_dir, 
            dataset_name=dataset_name,
            min_expected_size=10  # 至少应该有10条数据
        )
        
        if is_complete:
            print(f"✓ 数据集已完整存在于 {local_dir}，跳过下载")
            print(f"  总数据量: {actual_size} 条")
            return
        else:
            # 数据集不完整，继续下载
            if force_download:
                print(f"⚠ 强制重新下载，将覆盖目录: {local_dir}")
                import shutil
                shutil.rmtree(local_dir)
            else:
                print(f"⚠ 检测到不完整的数据集 (当前: {actual_size} 条)，将继续下载...")
                print(f"  （将自动断点续传，只下载缺失的部分）")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 先使用load_dataset加载数据集（支持断点续传）
            print("正在从HuggingFace加载数据集...")
            print("  （如果数据集已部分下载，将自动断点续传）")
            
            # 如果本地目录存在但不完整，先尝试从本地加载看看能否继续
            # 如果不行，再从远程加载
            dataset = None
            if os.path.exists(local_dir) and not force_download:
                try:
                    # 尝试从本地加载，看看能否继续
                    print("  尝试从本地继续加载...")
                    local_dataset = load_dataset(local_dir)
                    local_size = sum(len(split) for split in local_dataset.values())
                    print(f"  本地已有 {local_size} 条数据，将从远程继续下载...")
                except:
                    pass
            
            # 从远程加载完整数据集（会自动处理断点续传）
            dataset = load_dataset(dataset_name)
            
            # 显示下载的数据集信息
            print(f"\n✓ 数据集加载完成:")
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data)} 条数据")
            
            # 保存到本地（支持断点续传）
            print("\n正在保存数据集到本地...")
            if force_download and os.path.exists(local_dir):
                import shutil
                print(f"  删除旧目录: {local_dir}")
                shutil.rmtree(local_dir)
            
            # 保存数据集（如果目录存在，会合并或覆盖）
            # 注意：save_to_disk会覆盖，所以如果之前不完整，这里会重新保存完整版本
            dataset.save_to_disk(local_dir)
        
            print(f"✓ 数据集下载完成！保存在: {os.path.abspath(local_dir)}")
            
            # 验证下载结果
            print("\n验证下载结果...")
            try:
                verify_dataset = load_dataset(local_dir)
                total_size = sum(len(split) for split in verify_dataset.values())
                print(f"✓ 验证成功，总数据量: {total_size} 条")
                
                if total_size > 0:
                    print("\n数据集信息:")
                    for split_name, split_data in verify_dataset.items():
                        print(f"  {split_name}: {len(split_data)} 条数据")
                        if len(split_data) > 0:
                            print(f"    示例字段: {list(split_data[0].keys())}")
                
                return
            except Exception as ve:
                print(f"⚠ 验证失败: {ve}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    print(f"  重试下载... (第 {retry_count}/{max_retries} 次)\n")
                    continue
                else:
                    raise
            
        except Exception as e:
            error_str = str(e)
            retry_count += 1
            
            # 检查是否是网络错误
            is_network_error = (
                "Connection" in error_str or
                "timeout" in error_str.lower() or
                "Read timed out" in error_str
            )
            
            if is_network_error and retry_count < max_retries:
                print(f"\n⚠ 网络错误，等待后重试... (第 {retry_count}/{max_retries} 次)")
                import time
                wait_time = retry_count * 5
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                print("  继续下载...\n")
                continue
            else:
                print(f"\n❌ 下载数据集时出错: {error_str}")
                print("\n请确保:")
                print("1. 已安装 datasets: pip install datasets")
                print("2. 已安装 huggingface_hub: pip install huggingface_hub")
                print("3. 如果需要，已登录 HuggingFace: huggingface-cli login")
                print("4. 有足够的磁盘空间")
                print("5. 网络连接正常")
                if retry_count >= max_retries:
                    print(f"\n已重试 {max_retries} 次，下载失败")
                raise

if __name__ == "__main__":
    # 检查是否禁用自动镜像选择
    auto_mirror = '--no-mirror' not in sys.argv
    download_dataset(auto_mirror=auto_mirror)

