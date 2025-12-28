#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本包装器
根据adapter_path自动生成output_dir，避免覆盖现有结果
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path

def extract_checkpoint_info(adapter_path):
    """从adapter路径提取checkpoint类型和名称"""
    path = Path(adapter_path).resolve()
    parts = path.parts
    
    checkpoint_type = 'unknown'
    checkpoint_name = 'unknown'
    
    for i, part in enumerate(parts):
        if part in ['sft', 'dpo', 'kto']:
            checkpoint_type = part
            if i + 1 < len(parts):
                checkpoint_name = parts[i + 1]
            break
    
    return checkpoint_type, checkpoint_name

def generate_output_dir(adapter_path, base_dir='./evaluation_results'):
    """根据adapter路径生成唯一的output_dir"""
    checkpoint_type, checkpoint_name = extract_checkpoint_info(adapter_path)
    
    # 生成output_dir: evaluation_results_{type}_{checkpoint_name}
    output_dir = f'{base_dir}_{checkpoint_type}_{checkpoint_name}'
    
    return output_dir, checkpoint_type, checkpoint_name

def update_config_file(config_path, output_dir):
    """更新配置文件中的output_dir"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 备份原始配置
    backup_path = config_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
    
    # 更新output_dir
    config['output_dir'] = output_dir
    
    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    return backup_path

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'eval_config.yaml'
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    adapter_path = config.get('adapter_name_or_path', '')
    if not adapter_path:
        print("错误: 配置文件中未找到 adapter_name_or_path")
        sys.exit(1)
    
    # 生成output_dir
    output_dir, checkpoint_type, checkpoint_name = generate_output_dir(adapter_path)
    
    print("=" * 60)
    print("评估配置")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"Adapter路径: {adapter_path}")
    print(f"Checkpoint类型: {checkpoint_type}")
    print(f"Checkpoint名称: {checkpoint_name}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    print()
    
    # 检查目录是否已存在
    if os.path.exists(output_dir):
        print(f"⚠ 警告: 输出目录已存在: {output_dir}")
        response = input("是否继续？（会覆盖现有结果）[y/N]: ")
        if response.lower() != 'y':
            print("已取消")
            sys.exit(0)
    
    # 更新配置文件
    backup_path = update_config_file(config_path, output_dir)
    print(f"✓ 配置文件已更新，备份保存在: {backup_path}")
    print()
    
    # 运行评估
    print("正在运行评估...")
    print(f"命令: llamafactory-cli train {config_path}")
    print()
    
    result = subprocess.run(['llamafactory-cli', 'train', config_path])
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("✓ 评估完成!")
        print("=" * 60)
        print(f"结果已保存到: {output_dir}")
    else:
        print()
        print("=" * 60)
        print("✗ 评估失败")
        print("=" * 60)
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()

