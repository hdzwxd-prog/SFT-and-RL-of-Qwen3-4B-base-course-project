#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLaMA Factory进行模型微调的主脚本
配置为2张A6000 GPU
"""
import os
import sys
import subprocess

def check_llama_factory():
    """检查LLaMA Factory是否已安装"""
    try:
        import llamafactory
        print(f"LLaMA Factory已安装: {llamafactory.__version__}")
        return True
    except ImportError:
        print("错误: 未安装LLaMA Factory")
        print("请运行: pip install llamafactory")
        return False

def check_files():
    """检查必要的文件是否存在"""
    required_files = [
        "./Qwen3-4B-Base",
        "./tulu-3-sft-personas-instruction-following",
        "./train_config.yaml",
        "./data/dataset_info.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误: 以下文件/目录不存在:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请先运行下载脚本: python download_all.py")
        return False
    
    print("所有必要文件检查通过！")
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("Qwen3-4B-Base 微调训练脚本")
    print("=" * 60)
    
    # 检查LLaMA Factory
    if not check_llama_factory():
        sys.exit(1)
    
    # 检查文件
    if not check_files():
        sys.exit(1)
    
    # 获取配置文件路径
    config_file = "./train_config.yaml"
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)
    
    print(f"\n使用配置文件: {config_file}")
    print("开始训练...")
    print("=" * 60)
    
    # 使用llamafactory-cli进行训练
    # 对于2张GPU，LLaMA Factory会自动使用DDP
    cmd = [
        "llamafactory-cli",
        "train",
        config_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出错: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)

if __name__ == "__main__":
    main()

