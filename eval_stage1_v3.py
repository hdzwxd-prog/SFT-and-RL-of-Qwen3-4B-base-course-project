# 先运行export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1断网
# # 安装 socks 支持，防止 PIQA 下载时因为网络协议报错
# proxy_off关闭代理
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "httpx[socks]"


# # 1. 临时关闭离线模式
# unset HF_DATASETS_OFFLINE
# unset HF_HUB_OFFLINE
# export HF_ENDPOINT=https://hf-mirror.com
# # 2. 回到项目根目录
# cd /fxddata/250010031/MLS_project
# # 3. 运行之前的全量下载脚本 (full_download_verify.py 或 fix_mmlu_subsets.py)
# # 这里建议直接用下面这个命令快速把 MMLU 和其他核心补齐：
# python fix_mmlu_subsets.py 


# 1. 临时取消离线限制
# unset HF_DATASETS_OFFLINE
# unset HF_HUB_OFFLINE
# export HF_ENDPOINT=https://hf-mirror.com

# 2. 运行补全脚本
# python fix_missing_others.py

# 1. 设置镜像和缓存路径 (确保路径和你项目一致)
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_DATASETS_CACHE="/fxddata/250010031/MLS_project/dataset_cache"

# 2. 一行命令下载 HellaSwag (自动重试、自动解压、自动存入指定缓存)
# python -c "from datasets import load_dataset; print('正在下载 HellaSwag...'); load_dataset('Rowan/hellaswag', cache_dir='/fxddata/250010031/MLS_project/dataset_cache', trust_remote_code=True); print('✅ 下载完成')"


# 1. 务必取消离线模式，否则无法下载
# unset HF_DATASETS_OFFLINE
# unset HF_HUB_OFFLINE
# export HF_ENDPOINT=https://hf-mirror.com

# 2. 运行补全脚本
# python fix_final_missing.py


# 1. 卸载现有版本
# pip uninstall -y huggingface-hub

# 2. 安装符合要求的版本（0.34.0 是 lm_eval 兼容的稳定版）
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface-hub==0.34.0"

# 3. 验证版本（确保输出是 0.34.0）
# python -c "import huggingface_hub; print(huggingface_hub.__version__)"

# 4. 确保 lm_eval 是最新兼容版
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade lm_eval


# 临时关闭离线模式
# unset HF_DATASETS_OFFLINE
# unset HF_HUB_OFFLINE
# export HF_ENDPOINT=https://hf-mirror.com

# 单独下载 mmlu 的 professional_law 配置
# python -c "from datasets import load_dataset; load_dataset('cais/mmlu', 'professional_law', cache_dir='/fxddata/250010031/MLS_project/dataset_cache', trust_remote_code=True); print('✅ professional_law 下载完成')"

# 重新开启离线模式
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1



# 1. 开启离线保护（强制只读本地数据，秒级加载，杜绝联网报错）
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# 2. 运行科研级评估脚本
# (注意：确保你保存了我之前给你的那个带 log_samples=True 的代码为 eval_stage1_pro.py)
# python eval_stage1_v3.py

# 进入缓存目录
# cd /fxddata/250010031/MLS_project/dataset_cache

# 创建软链接：把 openai___gsm8k 映射为 gsm8k（lm_eval 要找的名称）
# ln -s openai___gsm8k gsm8k

# 验证链接是否创建成功（输出 gsm8k -> openai___gsm8k 即为成功）
# ls -l | grep gsm8k






import os
import json
import lm_eval
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from datetime import datetime

# ==========================================
# 1. 自动环境配置 (相对路径与镜像加速)
# ==========================================
# 使用当前目录作为根目录，确保移植性
CURRENT_DIR = os.getcwd()
MODEL_PATH = "/fxddata/share/Qwen3-4B-Base" # 基座模型路径
CACHE_DIR = os.path.join(CURRENT_DIR, "dataset_cache")
RESULT_DIR = os.path.join(CURRENT_DIR, "detailed_eval_results")

# 创建必要的文件夹
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 开启镜像加速并锁定本地缓存，确保 lm-eval 能找到我们下载的数据
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# ==========================================
# 2. 评估维度配置 (严格遵循说明书 5.1-5.5 维度 )
# ==========================================
TASK_CONFIGS = {
    "Academic Knowledge": {
        "tasks": ["mmlu", "arc_challenge"],
        "num_fewshot": 5, # [性能优化] 行业标准 5-shot 以激发百科知识 [cite: 31, 32]
        "desc": "学术与世界知识"
    },
    "Commonsense Reasoning": {
        # "tasks": ["hellaswag", "winogrande", "csqa"],
        
        "tasks": [ "winogrande", "commonsense_qa"],
        "num_fewshot": 0, # [性能优化] 考察直觉，通常 0-shot 效果更能反映基座能力 [cite: 36]
        "desc": "常识推理与语言消歧"
    },
    "Physical Reasoning": {
        "tasks": ["piqa", "arc_challenge"],
        "num_fewshot": 0, # [性能优化] 物理常识通常无需示例引导 [cite: 42]
        "desc": "物理推理与动力学"
    },
    "Math & Logic": {
        "tasks": ["gsm8k"],
        "num_fewshot": 8, # [性能优化] 关键：8-shot 激活 CoT，否则基座模型得分为 0 [cite: 48, 50]
        "desc": "数学逻辑与链式思考"
    },
    "Instruction Following": {
        "tasks": ["ifeval"],
        "num_fewshot": 0, # [性能优化] 严格测试 0-shot 下的复杂约束遵守情况 [cite: 53]
        "desc": "指令遵循与鲁棒性"
    }
}

# ==========================================
# 3. 智能数据加载与验证 (解决 PIQA 报错)
# ==========================================
def ensure_dataset_cached(path, config, name):
    """
    智能预加载函数：
    如果标准加载失败（如 PIQA 脚本错误），则自动切换到快照下载模式。
    确保 lm-eval 运行时数据已存在于本地 cache 中。
    """
    print(f"   正在检查数据集: {name}...", end="", flush=True)
    try:
        # 尝试标准加载
        load_dataset(path, config, cache_dir=CACHE_DIR, trust_remote_code=True)
        print(" [已就绪 ✅]")
    except Exception as e:
        # 针对 PIQA 等脚本不再支持的错误，使用快照下载
        if "scripts are no longer supported" in str(e) or "piqa" in name.lower():
            print(f"\n   [自动修复] 检测到脚本限制，正在为 {name} 启动快照下载...", end="", flush=True)
            try:
                # 下载到 cache 目录下的特定文件夹
                local_dir = os.path.join(CACHE_DIR, path.split('/')[-1])
                snapshot_download(
                    repo_id=path, 
                    repo_type="dataset", 
                    local_dir=local_dir, 
                    local_dir_use_symlinks=False,
                    endpoint="https://hf-mirror.com"
                )
                print(" [修复并下载成功 ✅]")
            except Exception as snap_e:
                print(f" [严重失败 ❌] {snap_e}")
        else:
            print(f" [错误] {str(e)[:100]}...")

def pre_flight_check():
    """运行前的全量数据检查"""
    print(f"[{datetime.now()}] >>> 启动数据预检 (Pre-flight Check)...")
    # 简单的映射关系用于预下载
    check_list = [
        ("cais/mmlu", "all", "MMLU"),
        ("allenai/ai2_arc", "ARC-Challenge", "ARC"),
        ("Rowan/hellaswag", None, "HellaSwag"),
        ("winogrande", "winogrande_xl", "WinoGrande"),
        ("piqa", None, "PIQA"), # 重点关注
        ("openai/gsm8k", "main", "GSM8K"),
        ("google/IFEval", None, "IFEval")
    ]
    for path, config, name in check_list:
        ensure_dataset_cached(path, config, name)
    print(f"[{datetime.now()}] >>> 数据预检完成。\n")

# ==========================================
# 4. 核心评估与详细记录 (Logging)
# ==========================================
def run_detailed_evaluation():
    summary_path = os.path.join(RESULT_DIR, "summary_report.json")
    failed_tasks_path = os.path.join(RESULT_DIR, "failed_tasks.json")
    
    # 读取已有进度
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            final_summary = json.load(f)
    else:
        final_summary = {}
    
    # 读取已有失败记录
    if os.path.exists(failed_tasks_path):
        with open(failed_tasks_path, "r") as f:
            failed_tasks = json.load(f)
    else:
        failed_tasks = {}

    # 屏蔽 git 无关报错
    import sys
    import contextlib
    @contextlib.contextmanager
    def suppress_git_errors():
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr = original_stderr

    # 导入 lm_eval 底层模块
    from lm_eval import evaluator, tasks, utils
    # from lm_eval.logger import eval_logger

    SAFE_BATCH_SIZE = 16 

    for group_name, cfg in TASK_CONFIGS.items():
        if group_name in final_summary:
            print(f">>> 维度 [{group_name}] 已有结果，跳过评估。")
            # continue
        
        print(f"\n>>> [开始评估] {group_name}")
        print(f"    说明: {cfg['desc']} (Shot={cfg['num_fewshot']})")
        
        successful_task_scores = []
        current_failed_tasks = []
        
        try:
            for single_task in cfg['tasks']:
                print(f"\n    ├─ 开始评估子任务: {single_task}")
                try:
                    # 1. 手动加载任务配置
                    task_dict = tasks.get_task_dict([single_task])
                    # 2. 配置模型参数
                    model_args = f"pretrained={MODEL_PATH},device_map=auto,dtype=bfloat16"
                    model = evaluator.load_model("hf", model_args=model_args)
                    
                    # 3. 运行评估
                    with suppress_git_errors():
                        results = evaluator.evaluate(
                            model=model,
                            task_dict=task_dict,
                            num_fewshot=cfg['num_fewshot'],
                            batch_size=SAFE_BATCH_SIZE,
                            limit=None,
                            log_samples=True,
                            parallel=False,
                            # logger=eval_logger,
                        )
                    
                    # 释放显存
                    del model
                    import torch
                    torch.cuda.empty_cache()
                    
                    # 提取任务分数
                    task_res = results["results"].get(single_task, {})
                    score = task_res.get("acc_norm,none") or task_res.get("acc,none") or task_res.get("prompt_level_strict_acc,none", 0)
                    successful_task_scores.append(score)
                    
                    # 保存核心分数
                    core_result = {
                        "task_name": single_task,
                        "num_fewshot": cfg['num_fewshot'],
                        "score": score,
                        "score_details": task_res,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    core_file = os.path.join(RESULT_DIR, f"core_{single_task.replace('/', '_')}.json")
                    with open(core_file, "w", encoding="utf-8") as f:
                        json.dump(core_result, f, indent=4, ensure_ascii=False)
                    
                    # ==========================================
                    # 核心修改：拆分样本存储（按学科/子维度）
                    # ==========================================
                    if "samples" in results and single_task in results["samples"]:
                        samples = results["samples"][single_task]
                        total_samples = len(samples)
                        
                        # 1. 创建任务专属文件夹（避免文件混乱）
                        task_dir = os.path.join(RESULT_DIR, f"samples_{single_task.replace('/', '_')}")
                        os.makedirs(task_dir, exist_ok=True)
                        
                        # 2. 针对 MMLU 按学科拆分（核心需求）
                        if single_task == "mmlu":
                            # 按学科分组样本
                            subject_samples = {}
                            for idx, sample in enumerate(samples):
                                # 提取学科信息（从 doc_id/metadata 或 prompt 中解析）
                                metadata = sample.get("metadata", {})
                                subject = metadata.get("subject", "unknown") or sample.get("doc_id", "unknown").split("_")[0]
                                
                                # 构建单样本数据
                                sample_data = {
                                    "sample_id": idx,
                                    "doc_id": sample.get("doc_id", "unknown"),
                                    "subject": subject,
                                    "full_prompt": sample.get("prompt", "") if isinstance(sample.get("prompt"), str) else str(sample.get("prompt")),
                                    "model_raw_output": sample.get("output", ""),
                                    "model_prediction": sample.get("pred", ""),
                                    "gold_answer": sample.get("gold", ""),
                                    "sample_score": sample.get("score", 0.0),
                                    "metadata": metadata
                                }
                                
                                # 按学科分组
                                if subject not in subject_samples:
                                    subject_samples[subject] = []
                                subject_samples[subject].append(sample_data)
                            
                            # 按学科保存文件（一个学科一个文件）
                            for subject, sub_samples in subject_samples.items():
                                subject_file = os.path.join(task_dir, f"mmlu_{subject}.json")
                                with open(subject_file, "w", encoding="utf-8") as f:
                                    json.dump(sub_samples, f, indent=4, ensure_ascii=False)
                            print(f"    │     ├─ MMLU 按学科拆分保存: {task_dir}")
                            print(f"    │     └─ 共 {len(subject_samples)} 个学科，总计 {total_samples} 个样本")
                        
                        # 3. 其他任务按批次拆分（避免单文件过大，每 1000 个样本一个文件）
                        else:
                            batch_size = 1000  # 可根据需要调整
                            num_batches = (total_samples // batch_size) + 1
                            
                            for batch_idx in range(num_batches):
                                start = batch_idx * batch_size
                                end = min((batch_idx + 1) * batch_size, total_samples)
                                batch_samples = samples[start:end]
                                
                                # 构建批次样本数据
                                batch_data = []
                                for idx_in_batch, sample in enumerate(batch_samples):
                                    global_idx = start + idx_in_batch
                                    sample_data = {
                                        "sample_id": global_idx,
                                        "doc_id": sample.get("doc_id", "unknown"),
                                        "full_prompt": sample.get("prompt", "") if isinstance(sample.get("prompt"), str) else str(sample.get("prompt")),
                                        "model_raw_output": sample.get("output", ""),
                                        "model_prediction": sample.get("pred", ""),
                                        "gold_answer": sample.get("gold", ""),
                                        "sample_score": sample.get("score", 0.0),
                                        "metadata": sample.get("metadata", {})
                                    }
                                    batch_data.append(sample_data)
                                
                                # 保存批次文件
                                batch_file = os.path.join(task_dir, f"{single_task}_batch_{batch_idx+1}.json")
                                with open(batch_file, "w", encoding="utf-8") as f:
                                    json.dump(batch_data, f, indent=4, ensure_ascii=False)
                            
                            print(f"    │     ├─ {single_task} 按批次拆分保存: {task_dir}")
                            print(f"    │     └─ 共 {num_batches} 个批次，总计 {total_samples} 个样本")
                    else:
                        print(f"    │     └─ 警告：未找到 {single_task} 样本数据")
                    
                    print(f"    │  └─ {single_task}: {score:.4f} | 核心分数: {core_file}")
                        
                except Exception as task_e:
                    error_msg = str(task_e)[:200]
                    current_failed_tasks.append({
                        "task": single_task,
                        "error": error_msg,
                        "group": group_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    print(f"    │  └─ ❌ {single_task} 评估失败: {error_msg}")
                    continue
            
            if current_failed_tasks:
                failed_tasks[group_name] = current_failed_tasks
                with open(failed_tasks_path, "w", encoding="utf-8") as f:
                    json.dump(failed_tasks, f, indent=4, ensure_ascii=False)
            
            if successful_task_scores:
                final_summary[group_name] = np.mean(successful_task_scores)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(final_summary, f, indent=4)
                print(f"\n>>> ✅ 维度 [{group_name}] 部分完成，成功任务平均分: {final_summary[group_name]:.4f}")
            else:
                print(f"\n>>> ❌ 维度 [{group_name}] 所有任务均失败，无分数可保存")
                
        except Exception as group_e:
            print(f"   [严重错误] 维度 [{group_name}] 初始化失败: {group_e}")
            import traceback
            traceback.print_exc()
            continue

    if failed_tasks:
        print(f"\n========================================")
        print(f"❌ 失败任务汇总 (详情见 {failed_tasks_path}):")
        for group, tasks in failed_tasks.items():
            print(f"  维度 {group}:")
            for task in tasks:
                print(f"    - {task['task']}: {task['error'][:100]}...")
    else:
        print(f"\n========================================")
        print(f"✅ 所有任务均成功完成！")

    return final_summary
# ==========================================
# 5. 可视化绘图
# ==========================================
def plot_radar(data):
    if not data: return
    labels = list(data.keys())
    values = list(data.values())
    
    # 闭合雷达图
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.3)
    ax.plot(angles, values, color='teal', linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_title("Qwen3-4B-Base Capability Assessment (Stage 1)", size=16, y=1.05)
    
    output_img = os.path.join(RESULT_DIR, "capability_radar.png")
    plt.savefig(output_img)
    print(f"\n[可视化] 最终雷达图已保存至: {output_img}")

if __name__ == "__main__":
    # 1. 数据预检与修复 (PIQA)
    # pre_flight_check()
    
    # 2. 运行详细评估
    scores = run_detailed_evaluation()
    
    # 3. 绘图
    plot_radar(scores)
    
    print(f"\n========================================")
    print(f"所有任务完成。请查看 {RESULT_DIR} 目录下的 JSON 日志文件。")
    print(f"这些 log_*.json 文件包含模型对每一道题的原始回答，可用于撰写详细报告。")