#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键下载模型和数据集
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_model import download_model
from download_dataset import download_dataset

def main():
    """下载模型和数据集"""
    print("=" * 60)
    print("开始下载项目所需文件")
    print("=" * 60)
    
    # 下载模型
    print("\n[1/2] 下载模型...")
    try:
        download_model()
    except Exception as e:
        print(f"模型下载失败: {e}")
        return
    
    # 下载数据集
    print("\n[2/2] 下载数据集...")
    try:
        download_dataset()
    except Exception as e:
        print(f"数据集下载失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("所有文件下载完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查下载的文件是否完整")
    print("2. 运行训练脚本: python train.py 或 llamafactory-cli train train_config.yaml")

if __name__ == "__main__":
    main()

