import os
import time
from datasets import load_dataset

# ================= 配置区 =================
PROJECT_ROOT = "/fxddata/250010031/MLS_project"
CACHE_DIR = os.path.join(PROJECT_ROOT, "dataset_cache")

# 强制使用国内镜像 (必须联网)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
# ==========================================

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

def robust_download():
    print(f"🔄 启动 MMLU 智能补全模式...")
    success_count = 0
    
    for i, subject in enumerate(MMLU_SUBJECTS):
        print(f"[{i+1}/{57}] 检查: {subject:<35} ", end="", flush=True)
        
        # 最大重试次数 3 次
        for attempt in range(3):
            try:
                # 尝试加载 (如果本地已有且完好，load_dataset 会直接校验通过，速度很快)
                load_dataset(
                    "cais/mmlu", 
                    subject, 
                    cache_dir=CACHE_DIR,
                    trust_remote_code=True
                )
                print("✅ 已就绪")
                success_count += 1
                break # 成功则跳出重试循环
            except Exception as e:
                if attempt < 2:
                    print(f"\n   ⚠️  超时重试 ({attempt+1}/3)... ", end="", flush=True)
                    time.sleep(2) # 歇 2 秒再试
                else:
                    print(f"❌ 最终失败: {str(e)[:50]}")

    print(f"\n📊 最终统计: {success_count}/57 个科目已就绪。")
    if success_count == 57:
        print("🎉 完美！现在可以去跑 eval_stage1_pro.py 了！")
    else:
        print("⚠️  仍有失败项，请再次运行此脚本。")

if __name__ == "__main__":
    robust_download()