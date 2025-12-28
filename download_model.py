#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Qwen3-4B-Base 模型到当前目录
支持自动选择最快的镜像源
"""
import os
import time
import requests
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        "use_modelscope": True  # 需要使用ModelScope的API
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
    """
    测试镜像站点的访问速度
    
    Args:
        mirror: 镜像站点配置字典
        timeout: 超时时间（秒）
    
    Returns:
        (success, latency): (是否成功, 延迟时间/秒)
    """
    try:
        start_time = time.time()
        response = requests.get(mirror["test_url"], timeout=timeout, allow_redirects=True)
        latency = time.time() - start_time
        
        if response.status_code in [200, 301, 302]:
            return True, latency
        else:
            return False, float('inf')
    except Exception as e:
        return False, float('inf')

def find_fastest_mirror(verbose=True):
    """
    测试所有镜像站点，返回最快的可用镜像
    
    Args:
        verbose: 是否显示详细测试信息
    
    Returns:
        best_mirror: 最快的镜像配置，如果都不可用则返回None
    """
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
    
    # 按延迟排序，选择最快的
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
    """
    设置环境变量以使用指定的镜像
    
    Args:
        mirror: 镜像配置字典
    """
    if mirror is None:
        return
    
    # 设置环境变量
    if "env_var" in mirror:
        os.environ[mirror["env_var"]] = mirror["endpoint"]
    
    # 特殊处理ModelScope
    if mirror.get("use_modelscope", False):
        os.environ["USE_MODELSCOPE_HUB"] = "1"
        # ModelScope使用不同的模型ID格式
        print("⚠ 注意: ModelScope需要将模型ID转换为ModelScope格式")
        print("  例如: Qwen/Qwen3-4B-Base -> Qwen/Qwen3-4B-Base")
    else:
        # 确保不使用ModelScope
        if "USE_MODELSCOPE_HUB" in os.environ:
            del os.environ["USE_MODELSCOPE_HUB"]
        
        # 设置HF镜像端点
        if mirror["endpoint"] != "https://huggingface.co":
            os.environ["HF_ENDPOINT"] = mirror["endpoint"]
            print(f"✓ 已设置镜像端点: {mirror['endpoint']}")

def check_model_completeness(local_dir):
    """
    检查模型文件是否完整（包括权重文件）
    
    Args:
        local_dir: 模型目录路径
    
    Returns:
        bool: 模型是否完整
    """
    import json
    
    # 检查必要的配置文件
    required_config_files = ["config.json", "tokenizer.json"]
    for file_name in required_config_files:
        file_path = os.path.join(local_dir, file_name)
        if not os.path.exists(file_path):
            return False
    
    # 检查模型权重文件
    # 方法1: 检查是否有index.json（分片模型）
    index_file = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 检查所有权重文件是否存在
            if "weight_map" in index_data:
                # 获取所有需要的文件
                required_files = set(index_data["weight_map"].values())
                missing_files = []
                for file_name in required_files:
                    file_path = os.path.join(local_dir, file_name)
                    if not os.path.exists(file_path):
                        missing_files.append(file_name)
                
                if missing_files:
                    print(f"  检测到缺失的模型权重文件: {len(missing_files)}/{len(required_files)}")
                    print(f"  缺失文件: {', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''}")
                    return False
                return True
        except Exception as e:
            print(f"  读取index.json失败: {e}")
            return False
    
    # 方法2: 检查是否有单个模型文件
    single_model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin"
    ]
    for file_name in single_model_files:
        file_path = os.path.join(local_dir, file_name)
        if os.path.exists(file_path):
            # 检查文件大小是否合理（至少1MB）
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:  # 至少1MB
                return True
    
    # 如果都没有，说明不完整
    print(f"  未找到完整的模型权重文件")
    return False

def download_model(model_name="Qwen/Qwen3-4B-Base", local_dir="./Qwen3-4B-Base", auto_mirror=True):
    """
    下载模型到本地目录
    
    Args:
        model_name: HuggingFace模型名称
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
    
    print(f"开始下载模型: {model_name}")
    print(f"保存位置: {os.path.abspath(local_dir)}")
    
    # 如果目录已存在，检查是否完整
    import sys
    force_download = '--force' in sys.argv
    resume_download = True  # 默认启用断点续传
    
    if os.path.exists(local_dir):
        # 检查模型文件是否完整
        is_complete = check_model_completeness(local_dir)
        
        if is_complete:
            try:
                # 再次验证：尝试加载tokenizer
                tokenizer = AutoTokenizer.from_pretrained(local_dir)
                print(f"✓ 模型已完整存在于 {local_dir}，跳过下载")
                print(f"  Tokenizer词汇表大小: {len(tokenizer)}")
                return
            except Exception as e:
                # tokenizer加载失败，但文件存在，可能是配置问题，继续下载
                print(f"⚠ Tokenizer验证失败，将重新下载: {e}")
                is_complete = False
        
        if not is_complete:
            # 如果加载失败，说明下载不完整
            if force_download:
                # 强制重新下载：删除旧目录
                print(f"⚠ 强制重新下载，删除旧目录: {local_dir}")
                import shutil
                shutil.rmtree(local_dir)
                resume_download = True  # 重新开始下载
            else:
                # 默认使用断点续传，只下载缺失的文件
                print(f"⚠ 检测到不完整的下载，将自动断点续传...")
                print(f"  （snapshot_download会自动检查并只下载缺失的文件）")
                print(f"  （如需强制重新下载，请使用 --force 参数）")
                resume_download = True
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 使用snapshot_download下载完整模型，支持断点续传
            print(f"开始下载/续传模型: {model_name}")
            try:
                result = snapshot_download(
                    repo_id=model_name,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                    # resume_download 参数已废弃，huggingface_hub会自动断点续传
                    # 如果遇到416错误（文件不匹配），需要删除损坏的文件后重试
                )
            except Exception as download_error:
                error_msg = str(download_error)
                # 如果是因为网络问题无法访问，但本地目录已存在，尝试使用本地文件
                if ("cannot be accessed" in error_msg or "Connection" in error_msg or "ProtocolError" in error_msg) and os.path.exists(local_dir):
                    print(f"⚠ 网络连接失败，但检测到本地目录存在，尝试使用本地文件...")
                    # 检查本地文件是否可用
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(local_dir)
                        print(f"✓ 使用本地已存在的模型文件")
                        print(f"✓ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
                        print("✓ 模型验证完成！")
                        return
                    except Exception as local_error:
                        print(f"⚠ 本地文件不完整或损坏: {local_error}")
                        # 继续抛出原始错误，让外层处理
                        raise download_error
                else:
                    # 其他错误直接抛出
                    raise
            
            # 检查返回值，snapshot_download可能返回None（当使用local_dir时）
            # 如果返回None，使用local_dir作为路径
            model_path = result if result else local_dir
            
            # 如果result是None，检查本地目录是否存在
            if result is None:
                if not os.path.exists(local_dir):
                    raise Exception("下载失败：本地目录不存在")
                
                # 检查是否有必要的模型文件
                required_files = ["config.json"]
                existing_files = [f for f in required_files if os.path.exists(os.path.join(local_dir, f))]
                if not existing_files:
                    raise Exception(f"下载不完整：缺少必要文件 {required_files}")
            
            # 检查路径是否存在
            if not os.path.exists(model_path):
                raise Exception(f"模型路径不存在: {model_path}")
            
            print(f"✓ 模型下载完成！保存在: {os.path.abspath(model_path)}")
            
            # 验证下载
            print("验证模型文件...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
            print("✓ 模型下载并验证完成！")
            return
            
        except Exception as e:
            error_str = str(e)
            
            # 检查是否是网络错误
            is_network_error = (
                "Connection" in error_str or
                "Connection reset" in error_str or
                "Connection aborted" in error_str or
                "ProtocolError" in error_str or
                "cannot be accessed" in error_str or
                "timeout" in error_str.lower() or
                "Network" in error_str
            )
            
            # 检查是否是文件大小不一致或416错误
            is_file_error = (
                "416" in error_str or 
                "Range Not Satisfiable" in error_str or
                "Consistency check failed" in error_str or
                "file should be of size" in error_str
            )
            
            # 检查是否是路径相关错误
            is_path_error = (
                "expected str, bytes or os.PathLike" in error_str or
                "not NoneType" in error_str or
                "NoneType" in error_str
            )
            
            # 处理网络错误
            if is_network_error:
                retry_count += 1
                print(f"\n⚠ 检测到网络错误，等待后重试... (第 {retry_count}/{max_retries} 次)")
                print(f"  错误信息: {error_str[:200]}")
                
                if retry_count < max_retries:
                    import time
                    wait_time = retry_count * 5  # 递增等待时间：5秒、10秒、15秒
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    print("  继续下载...\n")
                    continue
                else:
                    print(f"\n❌ 网络错误，重试 {max_retries} 次后仍然失败")
                    print("建议:")
                    print("1. 检查网络连接")
                    print("2. 如果使用代理，确保代理配置正确")
                    print("3. 稍后重试: python download_model.py")
                    print("4. 或使用镜像源（如果可用）")
                    raise
            
            # 处理路径错误（snapshot_download返回None的情况）
            if is_path_error:
                retry_count += 1
                print(f"\n⚠ 检测到路径错误，检查本地文件... (第 {retry_count}/{max_retries} 次)")
                
                # 检查本地目录是否存在
                if os.path.exists(local_dir):
                    # 检查是否有必要的文件
                    required_files = ["config.json"]
                    existing_files = [f for f in required_files if os.path.exists(os.path.join(local_dir, f))]
                    
                    if existing_files:
                        print(f"  发现本地文件，尝试直接使用本地目录...")
                        # 直接使用本地目录验证
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(local_dir)
                            print(f"✓ 使用本地已存在的模型文件")
                            print(f"✓ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
                            print("✓ 模型验证完成！")
                            return
                        except Exception as ve:
                            print(f"  本地文件验证失败: {ve}")
                            if retry_count < max_retries:
                                print("  尝试重新下载...\n")
                                continue
                    else:
                        print(f"  本地目录存在但缺少必要文件，尝试重新下载...")
                        if retry_count < max_retries:
                            continue
                else:
                    print(f"  本地目录不存在，尝试重新下载...")
                    if retry_count < max_retries:
                        continue
                
                if retry_count >= max_retries:
                    print(f"\n❌ 重试 {max_retries} 次后仍然失败")
                    print("建议使用 --force 参数强制重新下载:")
                    print(f"  python download_model.py --force")
                    raise
            
            if is_file_error:
                retry_count += 1
                print(f"\n⚠ 检测到文件损坏或不匹配，自动修复并继续下载... (第 {retry_count}/{max_retries} 次)")
                
                # 从错误信息中提取损坏的文件名
                import re
                damaged_file = None
                
                # 尝试从错误信息中提取文件名
                if "tokenizer.json" in error_str:
                    damaged_file = "tokenizer.json"
                elif "file should be of size" in error_str:
                    # 提取文件名，例如: "file should be of size 7031645 but has size 21209146 (tokenizer.json)"
                    match = re.search(r'\(([^)]+)\)', error_str)
                    if match:
                        damaged_file = match.group(1)
                
                # 删除损坏的文件
                if damaged_file:
                    damaged_path = os.path.join(local_dir, damaged_file)
                    if os.path.exists(damaged_path):
                        file_size = os.path.getsize(damaged_path)
                        print(f"  发现损坏的文件: {damaged_file} (大小: {file_size} bytes)")
                        print(f"  删除损坏的文件，将重新下载此文件...")
                        os.remove(damaged_path)
                
                # 清理缓存目录
                cache_dir = os.path.join(local_dir, ".cache")
                if os.path.exists(cache_dir):
                    import shutil
                    print(f"  清理缓存目录...")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                
                # 如果没找到具体文件，尝试删除所有可能损坏的常见文件
                if not damaged_file:
                    print(f"  清理可能损坏的文件...")
                    common_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
                    for cf in common_files:
                        cf_path = os.path.join(local_dir, cf)
                        if os.path.exists(cf_path):
                            print(f"    删除 {cf}")
                            os.remove(cf_path)
                
                if retry_count < max_retries:
                    print("  继续下载（断点续传）...\n")
                    continue
                else:
                    print(f"\n❌ 重试 {max_retries} 次后仍然失败")
                    print("建议使用 --force 参数强制重新下载:")
                    print(f"  python download_model.py --force")
                    raise
            else:
                # 处理其他错误（包括网络错误和路径错误）
                retry_count += 1
                
                # 检查是否是网络错误
                is_network_error = (
                    "Connection" in error_str or
                    "Connection reset" in error_str or
                    "Connection aborted" in error_str or
                    "ProtocolError" in error_str or
                    "cannot be accessed" in error_str or
                    "timeout" in error_str.lower()
                )
                
                # 检查是否是路径相关错误（NoneType）
                is_path_error = (
                    "expected str, bytes or os.PathLike" in error_str or
                    "not NoneType" in error_str or
                    "NoneType" in error_str
                )
                
                if is_network_error:
                    print(f"\n⚠ 检测到网络错误，等待后重试... (第 {retry_count}/{max_retries} 次)")
                    print(f"  错误信息: {error_str[:200]}")
                    
                    # 如果本地目录存在，尝试使用本地文件
                    if os.path.exists(local_dir):
                        print(f"  检测到本地目录存在，尝试验证本地文件...")
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(local_dir)
                            print(f"✓ 使用本地已存在的模型文件")
                            print(f"✓ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
                            print("✓ 模型验证完成！")
                            return
                        except:
                            pass
                    
                    if retry_count < max_retries:
                        import time
                        wait_time = retry_count * 5  # 递增等待时间
                        print(f"  等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        print("  继续下载...\n")
                        continue
                
                elif is_path_error:
                    print(f"\n⚠ 检测到路径错误，检查本地文件... (第 {retry_count}/{max_retries} 次)")
                    
                    # 检查本地目录是否存在
                    if os.path.exists(local_dir):
                        print(f"  尝试直接使用本地目录验证...")
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(local_dir)
                            print(f"✓ 使用本地已存在的模型文件")
                            print(f"✓ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
                            print("✓ 模型验证完成！")
                            return
                        except Exception as ve:
                            print(f"  本地文件验证失败: {ve}")
                            if retry_count < max_retries:
                                print("  尝试重新下载...\n")
                                continue
                    else:
                        if retry_count < max_retries:
                            print("  本地目录不存在，尝试重新下载...\n")
                            continue
                
                # 如果重试次数用完或不是可重试的错误
                if retry_count >= max_retries:
                    print(f"\n❌ 下载模型时出错: {error_str}")
                    print("\n请确保:")
                    print("1. 已安装 huggingface_hub: pip install huggingface_hub")
                    print("2. 已登录 HuggingFace: huggingface-cli login")
                    print("3. 有足够的磁盘空间")
                    print("4. 网络连接正常")
                    print("\n如需强制重新下载，请使用:")
                    print("  python download_model.py --force")
                    raise
                else:
                    # 继续重试
                    continue

if __name__ == "__main__":
    import sys
    # 检查是否禁用自动镜像选择
    auto_mirror = '--no-mirror' not in sys.argv
    download_model(auto_mirror=auto_mirror)

