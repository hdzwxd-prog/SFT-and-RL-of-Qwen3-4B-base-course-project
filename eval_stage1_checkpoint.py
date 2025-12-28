import os
import json
import lm_eval
import matplotlib.pyplot as plt
import numpy as np

# --- 路径与配置 ---
MODEL_PATH = "/fxddata/share/Qwen3-4B-Base"
OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 根据 MLSYS Project 1 说明书划分的五个核心维度 [cite: 31-60]
TASK_GROUPS = {
    "Academic Knowledge": ["mmlu", "arc_challenge"],  # [cite: 32]
    "Commonsense Reasoning": ["hellaswag", "winogrande", "csqa"],  # [cite: 37]
    "Physical Reasoning": ["piqa", "arc_challenge"],  # [cite: 43]
    "Math & Logic": ["gsm8k"],  # [cite: 49]
    "Instruction Following": ["ifeval"]  # [cite: 54]
}

def run_evaluation():
    # 尝试加载已有的汇总结果，如果不存在则初始化为空
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            final_summary = json.load(f)
        print(f"[提示] 检测到已有的汇总结果，包含维度: {list(final_summary.keys())}")
    else:
        final_summary = {}

    # 针对 40GB MIG 实例，固定较小的 batch_size 确保稳定
    SAFE_BATCH_SIZE = 8 

    for group_name, tasks in TASK_GROUPS.items():
        # 如果这个维度已经跑过了，直接跳过
        if group_name in final_summary:
            print(f">>> 维度 [{group_name}] 已有结果，跳过。")
            continue
        
        print(f"\n>>> 正在开始评估维度: {group_name} (任务列表: {tasks})")
        
        try:
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={MODEL_PATH},device_map=auto,dtype=bfloat16",
                tasks=tasks,
                num_fewshot=0, 
                batch_size=SAFE_BATCH_SIZE
            )
            
            group_scores = []
            for t in tasks:
                res = results["results"].get(t, {})
                # 按照 acc_norm > acc > specific_metrics 的顺序提取得分 [cite: 30]
                score = res.get("acc_norm,none") or res.get("acc,none") or res.get("prompt_level_strict_acc,none", 0)
                group_scores.append(score)
                print(f"   - 任务 {t} 完成，得分: {score:.4f}")
            
            # 计算该维度的平均分并存入字典
            final_summary[group_name] = np.mean(group_scores)
            
            # 【关键一步】每跑完一个维度，立即写入磁盘保存进度
            with open(summary_path, "w") as f:
                json.dump(final_summary, f, indent=4)
            print(f">>> [已保存] 维度 {group_name} 的结果已写入磁盘。")
            
        except Exception as e:
            print(f"   [严重错误] 维度 {group_name} 运行失败: {e}")
            # 这里不保存 0 分，下次重启可以重新尝试
            continue

    return final_summary

def plot_radar(data):
    if not data:
        print("[警告] 没有数据可供绘图。")
        return
        
    labels = list(data.keys())
    values = list(data.values())
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='forestgreen', alpha=0.3)
    ax.plot(angles, values, color='forestgreen', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Qwen3-4B-Base Capability Cluster (Stage 1)", size=15)
    
    plt.savefig(f"{OUTPUT_DIR}/capability_radar.png")
    print(f"\n[成功] 最终雷达图已更新至: {OUTPUT_DIR}/capability_radar.png")

if __name__ == "__main__":
    scores = run_evaluation()
    plot_radar(scores)