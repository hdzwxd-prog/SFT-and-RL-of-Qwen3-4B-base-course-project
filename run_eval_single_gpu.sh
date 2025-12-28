#!/bin/bash
# 单GPU评估脚本（如果GPU 1有空闲，可以指定使用GPU 1）
# 使用方法: bash run_eval_single_gpu.sh

# 如果GPU 1有空闲，可以使用GPU 1（修改为0使用GPU 0）
export CUDA_VISIBLE_DEVICES=0

# 使用低显存配置进行评估
echo "使用单GPU进行评估（GPU $CUDA_VISIBLE_DEVICES）..."
echo "注意：如果训练正在运行，请等待训练完成后再评估"
echo ""

# 检查GPU显存
echo "当前GPU显存使用情况："
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "开始评估..."
llamafactory-cli train eval_config_lowmem.yaml

