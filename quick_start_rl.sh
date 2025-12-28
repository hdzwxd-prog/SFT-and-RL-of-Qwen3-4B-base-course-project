#!/bin/bash
# 快速开始RL训练的脚本

echo "=========================================="
echo "强化学习训练快速开始脚本"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ 检测到未激活虚拟环境，正在激活..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✓ 虚拟环境已激活"
    else
        echo "❌ 未找到虚拟环境，请先创建并激活虚拟环境"
        exit 1
    fi
else
    echo "✓ 虚拟环境已激活: $VIRTUAL_ENV"
fi

echo ""

# 步骤1: 下载偏好数据集
echo "步骤 1: 下载偏好数据集"
echo "----------------------------------------"
echo "请选择要下载的数据集："
echo "  1) ultrafeedback (英文，推荐，约6.4万条)"
echo "  2) orca_dpo (英文，约10万条)"
echo "  3) dpo_en_zh (中英文混合，约2万条)"
echo "  4) 跳过下载（如果已下载）"
read -p "请选择 (1-4): " choice

case $choice in
    1)
        python download_preference_dataset.py ultrafeedback
        ;;
    2)
        python download_preference_dataset.py orca_dpo
        ;;
    3)
        python download_preference_dataset.py dpo_en_zh
        ;;
    4)
        echo "跳过数据集下载"
        ;;
    *)
        echo "无效选择，跳过下载"
        ;;
esac

echo ""

# 步骤2: 检查SFT checkpoint
echo "步骤 2: 检查SFT checkpoint"
echo "----------------------------------------"
echo "请确认SFT checkpoint路径:"
ls -lt saves/qwen3-4b-base/lora/sft/checkpoint-* 2>/dev/null | head -5

if [ $? -ne 0 ]; then
    echo "⚠ 未找到SFT checkpoint，请先完成SFT训练"
    exit 1
fi

read -p "请输入要使用的checkpoint名称（例如: checkpoint-2000）: " checkpoint_name
CHECKPOINT_PATH="saves/qwen3-4b-base/lora/sft/$checkpoint_name"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "✓ 找到checkpoint: $CHECKPOINT_PATH"

# 更新配置文件
sed -i "s|adapter_name_or_path:.*|adapter_name_or_path: ./$CHECKPOINT_PATH|" dpo_config.yaml
echo "✓ 已更新 dpo_config.yaml 中的 adapter_name_or_path"

echo ""

# 步骤3: 确认配置
echo "步骤 3: 确认训练配置"
echo "----------------------------------------"
echo "当前配置摘要:"
echo "  基础模型: $(grep '^model_name_or_path:' dpo_config.yaml | awk '{print $2}')"
echo "  SFT Checkpoint: $CHECKPOINT_PATH"
echo "  数据集: $(grep '^dataset:' dpo_config.yaml | awk '{print $2}')"
echo "  输出目录: $(grep '^output_dir:' dpo_config.yaml | awk '{print $2}')"
echo ""

read -p "是否开始训练? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消训练"
    exit 0
fi

echo ""

# 步骤4: 开始训练
echo "步骤 4: 开始DPO训练"
echo "----------------------------------------"
echo "开始训练，请稍候..."
echo ""

llamafactory-cli train dpo_config.yaml

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 检查训练日志和损失曲线"
echo "2. 使用评估脚本测试模型: llamafactory-cli train eval_config.yaml"
echo "3. 或使用交互式对话测试: llamafactory-cli chat ..."
echo ""

