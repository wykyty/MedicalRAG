import os
import sys
import argparse
from datetime import datetime

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # 设置多进程启动方式，防止 vLLM 冲突

flashrag_path = '/data/wyh/MedicalRAG/FlashRAG'
sys.path.insert(0, flashrag_path)


from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

# ================= 1. Prompt 定义区域 =================

TEST_PROMPT = """
Answer the following question.
{reference}
"""

# Baseline (无 RAG)
BASELINE_SYSTEM_PROMPT = """
Answer the following question. You should only use your own knowledge.
"""

# Strict RAG (严格检索)
STRICT_SYSTEM_PROMPT = """
Answer the question based ONLY on the following Reference Q&A.
If the reference does not contain the answer, say "I don't know".

Reference Q&A:
{reference}
"""

# Hybrid RAG (混合策略)
HYBRID_SYSTEM_PROMPT = """
Answer the question using your own knowledge and the following Reference Q&A.

Reference Q&A:
{reference}
"""

# 策略字典映射
PROMPT_MAP = {
    "test": TEST_PROMPT,
    "baseline": BASELINE_SYSTEM_PROMPT,
    "strict": STRICT_SYSTEM_PROMPT,
    "hybrid": HYBRID_SYSTEM_PROMPT
}

# ================= 2. 主程序逻辑 =================

def main(args):
    # 1. 确定 Prompt 和 保存路径
    system_prompt = PROMPT_MAP[args.strategy]
    
    # 结果保存路径：output/{strategy}/
    base_save_dir = "/data/wyh/MedicalRAG/output"
    current_save_dir = os.path.join(base_save_dir, args.strategy)
    os.makedirs(current_save_dir, exist_ok=True)

    print(f"\n{'='*40}")
    print(f"🚀 正在启动实验: {args.strategy.upper()}")
    print(f"📂 结果保存路径: {current_save_dir}")
    print(f"{'='*40}\n")

    # 2. 构建配置字典
    config_dict = {
        "data_dir": "/data/wyh/MedicalRAG/data",
        "dataset_name": "Huatuo26M-Lite",
        "split": args.split,

        # 框架与评测
        "framework": "host",  # 'host', 'api', 'vllm'
        # "generator_model": "qwen2.5-7B-instruct",
        "generator_batch_size": args.batch_size,
        "generation_params": {
            "max_tokens": 512,
            "temperature": 0.1, # 医学问题保持低随机性
            "top_p": 0.9
        },
        # "metrics": ['acc', 'em', 'f1', 'bleu', 'rouge-l', 'recall', 'precision', 'rouge-1', 'rouge-2'],
        "metrics": ['gpt_harmful_rate', 'gpt_hallucination_rate'],
        
        "api_setting": {
            "model_name": "gpt-4o-mini",
            "generator_model": "deepseek-r1",
            "concurrency": args.batch_size, # API 并发数
            "timeout_sec": 60,
            # "api_key": os.getenv("OPENAI_API_KEY")
        },

        # 检索配置
        # "index_path": "/data/wyh/MedicalRAG/data/indexes/huatuo_bm25_index/bm25",
        "index_path": "/data/wyh/MedicalRAG/data/indexes/huatuo_bge_index/bge_Flat.index",
        "corpus_path": "/data/wyh/MedicalRAG/data/indexes/corpus.jsonl",
        "retrieval_method": "bge",
        "retrieval_topk": 5,
        # "bm25_backend": "bm25s", # if and only if retrieval_method == bm25

        # 硬件配置
        "gpu_id": args.gpu_id,
        "gpu_num": len(args.gpu_id.split(',')),
        "gpu_memory_utilization": 0.8,
        
        # 保存路径
        "save_dir": current_save_dir
    }

    # 3. 初始化 Config 和 Dataset
    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    # Prompt
    template = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt="""Question: {question}"""
    )

    # Pipeline
    pipeline = SequentialPipeline(config, template)

    if args.strategy == "baseline" or args.strategy == "test":
        print(">>> 正在执行 Baseline 模式 (Naive Run - 无检索)...")
        result = pipeline.naive_run(test_data, do_eval=True)
    else:
        print(f">>> 正在执行 {args.strategy} RAG 模式 (Run - 含检索)...")
        result = pipeline.run(test_data, do_eval=True)

    print(f"\n✅ 实验结束！结果已保存至: {current_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical RAG Experiment Runner")
    parser.add_argument("--strategy", type=str, required=True, 
                        choices=["baseline", "strict", "hybrid", "test"],
                        help="选择实验策略: baseline(无RAG), strict(严格RAG), hybrid(混合RAG), test(测试)")
    parser.add_argument("--gpu_id", type=str, default="0, 1", help="使用的 GPU ID，例如 '0,1'")
    parser.add_argument("--split", type=str, default="test", help="测试集切分名称")
    parser.add_argument("--batch_size", type=int, default=64, help="推理 Batch Size")
    
    args = parser.parse_args()

    main(args)
