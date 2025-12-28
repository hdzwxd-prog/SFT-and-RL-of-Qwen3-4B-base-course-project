#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查环境设置和依赖
"""
import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠️  警告: 推荐使用Python 3.9或更高版本")
        return False
    print("✓ Python版本符合要求")
    return True

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        if import_name == "llamafactory":
            # LLaMA Factory可能不在标准位置
            try:
                import llamafactory
                print(f"✓ {package_name} 已安装 (版本: {llamafactory.__version__})")
                return True
            except ImportError:
                print(f"✗ {package_name} 未安装")
                return False
        else:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name} 已安装 (版本: {version})")
            return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA不可用")
            return False
    except ImportError:
        print("✗ PyTorch未安装，无法检查CUDA")
        return False

def check_files():
    """检查必要的文件"""
    import os
    files_to_check = [
        ("train_config.yaml", "训练配置文件"),
        ("data/dataset_info.json", "数据集信息配置"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description}不存在: {file_path}")
            all_exist = False
    
    return all_exist

def check_model_dataset():
    """检查模型和数据集是否已下载"""
    import os
    model_dir = "./Qwen3-4B-Base"
    dataset_dir = "./tulu-3-sft-personas-instruction-following"
    
    model_exists = os.path.exists(model_dir)
    dataset_exists = os.path.exists(dataset_dir)
    
    if model_exists:
        print(f"✓ 模型目录存在: {model_dir}")
    else:
        print(f"✗ 模型目录不存在: {model_dir}")
        print("  运行: python download_model.py")
    
    if dataset_exists:
        print(f"✓ 数据集目录存在: {dataset_dir}")
    else:
        print(f"✗ 数据集目录不存在: {dataset_dir}")
        print("  运行: python download_dataset.py")
    
    return model_exists and dataset_exists

def main():
    """主函数"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    all_ok = True
    
    print("\n[1] Python版本检查")
    all_ok = check_python_version() and all_ok
    
    print("\n[2] 必要包检查")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("llamafactory", "llamafactory"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    print("\n[3] CUDA检查")
    cuda_ok = check_cuda()
    if not cuda_ok:
        all_ok = False
    
    print("\n[4] 配置文件检查")
    if not check_files():
        all_ok = False
    
    print("\n[5] 模型和数据集检查")
    if not check_model_dataset():
        all_ok = False
        print("\n提示: 运行 python download_all.py 下载模型和数据集")
    
    print("\n" + "=" * 60)
    if all_ok and cuda_ok:
        print("✓ 所有检查通过！可以开始训练")
        print("  运行: python train.py 或 llamafactory-cli train train_config.yaml")
    else:
        print("✗ 部分检查未通过，请根据上述提示修复问题")
    print("=" * 60)

if __name__ == "__main__":
    main()

