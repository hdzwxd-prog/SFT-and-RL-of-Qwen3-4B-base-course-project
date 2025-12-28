# 故障排除指南

本文档提供项目使用过程中常见问题的解决方案和调试技巧。

## 目录

1. [环境配置问题](#环境配置问题)
2. [训练相关问题](#训练相关问题)
3. [评估相关问题](#评估相关问题)
4. [模型加载问题](#模型加载问题)
5. [调试技巧](#调试技巧)

---

## 环境配置问题

### 问题1: 虚拟环境相关问题

**症状**: 找不到已安装的包、Command not found

**解决方案**:
1. 确保已激活虚拟环境（命令行前应有环境名称提示）
2. 如果虚拟环境损坏，可以删除后重新创建：
   ```bash
   # venv方式
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # conda方式
   conda deactivate
   conda remove -n qwen3-ft --all -y
   conda create -n qwen3-ft python=3.10 -y
   conda activate qwen3-ft
   pip install -r requirements.txt
   ```

### 问题2: 依赖安装失败

**症状**: pip install 失败、版本冲突

**解决方案**:
1. 升级pip: `pip install --upgrade pip`
2. 使用国内镜像: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. 检查Python版本: 确保使用 Python 3.9+
4. 检查CUDA版本: 确保CUDA版本与PyTorch兼容

---

## 训练相关问题

### 问题1: 显存不足 (OOM)

**症状**: CUDA out of memory

**解决方案**:
1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 减小 `cutoff_len` 或 `max_length`
4. 启用 `gradient_checkpointing`（已在配置中启用）
5. 使用 QLoRA (在配置中设置 `quantization_bit: 4` 或 `8`)

### 问题2: 多GPU训练问题

**症状**: NCCL错误、GPU通信失败

**解决方案**:
1. 确认 `CUDA_VISIBLE_DEVICES` 设置正确
2. 检查NCCL环境变量
3. 确保所有GPU可访问: `nvidia-smi`
4. 尝试单GPU训练进行测试

### 问题3: 训练中断恢复

**症状**: 训练中断后如何继续

**解决方案**:
- LLaMA Factory会自动从检查点恢复，重新运行相同命令即可
- 检查 `saves/` 目录下的checkpoint文件

---

## 评估相关问题

### 问题1: lm_eval PEFT兼容性问题

**症状**: `TypeError: Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'`

**原因分析**:
- lm_eval在处理PEFT模型时，可能将`dtype`参数错误传递给模型初始化
- PEFT模型不支持在model_args中指定`dtype`参数

**解决方案**:
1. **推荐方案**: 使用 `evaluate_ifeval_with_llamafactory.py` 而不是 `evaluate_instruction_following.py`
   - 该脚本直接使用transformers和peft加载模型，避免lm_eval的PEFT兼容性问题
   - 使用 `--use_lm_eval` 参数可以计算IFEval评分

2. **备选方案**: 使用LLaMA Factory的评估功能
   ```bash
   llamafactory-cli train eval_config.yaml
   ```

3. **手动加载模型**:
   ```python
   from transformers import AutoModelForCausalLM
   from peft import PeftModel
   
   # 手动加载基础模型
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_path,
       torch_dtype=torch.bfloat16,
       device_map="auto"
   )
   
   # 加载 PEFT 适配器
   model = PeftModel.from_pretrained(base_model, adapter_path)
   ```

### 问题2: 评估结果不一致

**症状**: 不同评估方法得到的结果差异较大

**解决方案**:
1. 确保使用相同的评估配置
2. 检查数据集版本是否一致
3. 确认评估样本数量相同
4. 使用相同的随机种子

---

## 模型加载问题

### 问题1: 模型路径错误

**症状**: FileNotFoundError、模型加载失败

**解决方案**:
1. 检查模型路径是否正确
2. 确认模型文件是否完整
3. 检查文件权限

### 问题2: 模型格式不兼容

**症状**: 无法加载checkpoint、版本不兼容

**解决方案**:
1. 检查transformers和peft版本
2. 确保使用兼容的版本组合
3. 尝试更新到最新版本: `pip install --upgrade transformers peft`

---

## 调试技巧

### 1. 阅读错误消息

查找关键短语：
- `unexpected`: 参数名称错误
- `missing`: 缺少必需参数
- `invalid`: 参数值无效
- `not found`: 文件或模块未找到

注意错误类型：
- `TypeError`: 类型错误，参数类型不匹配
- `AttributeError`: 属性错误，对象没有该属性
- `ValueError`: 值错误，参数值无效
- `FileNotFoundError`: 文件未找到

### 2. 查看完整调用栈

从最底层（最后一个错误）开始看，向上追踪到你的代码：
```python
File ".../your_script.py", line 70, in your_function
    # 你的代码
File ".../library/module.py", line 245, in library_function
    # 库的代码
...
```

### 3. 检查参数和类型

- 确认参数名称是否正确
- 确认参数类型是否匹配
- 查看函数/类的文档或源代码

### 4. 搜索类似问题

- 在项目代码中搜索类似用法
- 查找相关文档或示例
- 搜索GitHub issues

### 5. 逐步测试

- 先简化代码，只测试问题部分
- 尝试不同的参数组合
- 检查库版本兼容性

### 6. 使用调试工具

```python
# 打印关键变量
print(f"Model path: {model_path}")
print(f"Checkpoint path: {checkpoint_path}")

# 检查对象类型
print(type(model))
print(model.__class__.__name__)

# 检查对象属性
print(dir(model))
```

### 7. 环境检查脚本

运行环境检查脚本：
```bash
python check_setup.py
```

这会检查：
- Python版本
- 依赖包安装情况
- CUDA可用性
- 配置文件存在性
- 模型和数据集路径

---

## 常见错误修复示例

### 错误1: dtype参数错误

**错误信息**:
```
TypeError: Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'
```

**修复方法**:
```python
# 错误代码
model_args = f"pretrained={BASE_MODEL_PATH},peft={adapter_path},device_map=auto,dtype=bfloat16"

# 正确代码（PEFT模型）
model_args = f"pretrained={BASE_MODEL_PATH},peft={adapter_path},device_map=auto"

# 或者（基础模型）
model_args = f"pretrained={model_path},device_map=auto,torch_dtype=bfloat16"
```

**原因**: PEFT模型会自动继承基础模型的dtype，不需要（也不支持）在model_args中指定dtype。如果一定要指定，应使用`torch_dtype`而不是`dtype`。

### 错误2: 数据集加载失败

**错误信息**: Dataset not found、Cache directory error

**修复方法**:
1. 检查数据集缓存目录是否存在
2. 确认环境变量设置正确:
   ```bash
   export HF_DATASETS_CACHE=./MLS_project_stage1/dataset_cache
   export HF_HOME=./MLS_project_stage1/dataset_cache
   ```
3. 重新下载数据集

### 错误3: 显存不足

**错误信息**: CUDA out of memory

**修复方法**:
1. 减小batch size
2. 启用梯度检查点
3. 使用混合精度训练
4. 减少序列长度
5. 使用量化（QLoRA）

---

## 获取帮助

如果以上方法都无法解决问题：

1. **检查日志**: 查看详细的错误日志和调用栈
2. **查看文档**: 参考相关库的官方文档
3. **搜索问题**: 在GitHub issues中搜索类似问题
4. **简化复现**: 创建一个最小复现示例
5. **版本信息**: 记录Python、PyTorch、transformers等关键库的版本

---

## 预防措施

1. **使用虚拟环境**: 避免依赖冲突
2. **定期备份**: 备份重要的checkpoint和配置文件
3. **版本控制**: 使用git管理代码和配置
4. **环境检查**: 运行前先检查环境配置
5. **逐步测试**: 先在小数据集上测试，确认无误后再全量运行
