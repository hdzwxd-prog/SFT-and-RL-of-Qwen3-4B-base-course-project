import os
import time
from datasets import load_dataset

# ... (前面的配置部分保持不变) ...
PROJECT_ROOT = os.getcwd()
CACHE_DIR = os.path.join(PROJECT_ROOT, "dataset_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
# ...

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

def fix_mmlu_with_report():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 启动 MMLU 修复任务...")
    
    success = 0
    failed_subjects = []  # <--- 新增：用来存黑名单

    for i, subject in enumerate(MMLU_SUBJECTS):
        print(f"[{i+1}/{len(MMLU_SUBJECTS)}] 检查: {subject:<35} ... ", end="", flush=True)
        try:
            # 已经下载好的会秒过
            load_dataset("cais/mmlu", subject, cache_dir=CACHE_DIR, trust_remote_code=True)
            print("✅ 成功", flush=True)
            success += 1
        except Exception as e:
            print(f"❌ 失败", flush=True)
            failed_subjects.append(subject) # <--- 记录失败者

    print(f"\n{'='*40}")
    print(f"📊 最终统计: 成功 {success} / 失败 {len(failed_subjects)}")
    
    if failed_subjects:
        print(f"⚠️  以下科目下载失败，请手动记录或针对性重试：")
        for sub in failed_subjects:
            print(f"   🔴 {sub}")
    else:
        print("🎉 全部完美通过！")
    print(f"{'='*40}")

if __name__ == "__main__":
    fix_mmlu_with_report()