import os
import sys
import argparse
from datetime import datetime

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # 设置多进程启动方式，防止 vLLM 冲突

# 添加项目路径
flashrag_path = '/data/wyh/MedicalRAG/FlashRAG'
sys.path.insert(0, flashrag_path)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

# ================= 1. Prompt 定义区域 =================
# 策略 A: Baseline (无 RAG)
# 重点：强调利用内部知识，不提及检索，不提及参考案例
BASELINE_SYSTEM_PROMPT = """你是一名经验丰富的医学专家助手。请利用你所掌握的专业医学知识来回答用户的问题。
"""

# 策略 B: Strict RAG (严格检索)
# 重点：强制依赖检索结果，如果检索结果无关则拒答
STRICT_SYSTEM_PROMPT = """你是一名经验丰富的医学专家助手。为了回答用户的问题，系统检索了数据库中相似的历史问答（Reference Cases）供你参考。

请按照以下步骤进行回答：
1. **分析参考案例**：仔细阅读提供的历史问答，关注 Question（相似问题）和 Answer（专家解答）。
2. **核对疾病领域**：检查 Related_diseases 和 Label，确保参考案例与用户问题的疾病领域一致。
3. **综合生成**：基于参考案例中的医学知识，针对用户的具体问题生成准确、通顺的回答。

【注意】：
- 如果参考案例与用户问题**完全无关**，请忽略该案例。
- 如果所有案例都无关，请诚实回答“数据库中没有相关参考信息，无法回答”。
- 严禁根据记忆编造药物剂量或治疗方案，必须基于参考信息。

---
【历史参考案例 (Reference Cases)】:
{reference}
"""

# 策略 C: Hybrid RAG (混合策略)
# 重点：检索优先，但允许安全降级到通用建议，兼顾准确率和回复率
HYBRID_SYSTEM_PROMPT = """你是一名经验丰富的医学专家助手。系统检索了部分历史问答（Reference Cases）供你参考。

请严格按照以下思维链（Chain of Thought）进行回答：

Step 1: **相关性评估**
- 仔细阅读【Reference Cases】，判断它们是否包含能回答用户【User Question】的关键信息。
- 如果参考案例中的疾病、症状或治疗方案与用户问题高度相关，标记为 [引用模式]。
- 如果参考案例与用户问题无关（如疾病领域不同、问题类型不同），标记为 [通用模式]。

Step 2: **回答生成**
- **如果是 [引用模式]**：
  - 必须完全基于参考案例中的信息回答。
  - 在回答末尾注明：“（依据检索到的类似病例：Case X）”。
  
- **如果是 [通用模式]**：
  - 忽略无关的参考案例。
  - 基于你自身的医学专业知识回答。
  - **关键约束**：必须在回答开头声明：“检索库中无完全匹配案例，以下建议基于通用医学知识，仅供参考。”
  - **安全红线**：在通用模式下，严禁提供精确的处方药剂量，只能提供治疗原则或非处方建议。

---
【历史参考案例 (Reference Cases)】:
{reference}
"""

# 通用的用户输入模板
USER_PROMPT_TEMPLATE = """
【用户当前问题 (User Question)】: 
{question}

【你的专家建议】:
"""

# 策略字典映射
PROMPT_MAP = {
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
        # "generator_model": ""
        # 框架与评测
        "framework": "host", 
        "metrics": ['gpt_score', 'gpt_hallucination_rate', 'gpt_harmful_rate'],
        
        "api_setting": {
            "model_name": "gpt-4o-mini",
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
        # "bm25_backend": "bm25s", # if and only if retrieval_method == bge

        # 硬件配置
        "gpu_id": args.gpu_id,
        "gpu_num": len(args.gpu_id.split(',')),
        "gpu_memory_utilization": 0.8,
        
        # 保存路径
        "save_dir": current_save_dir,
        
        # 重要的生成参数
        "generator_batch_size": args.batch_size,
        "generation_params": {
            "max_new_tokens": 512,
            "temperature": 0.1, # 医学问题保持低随机性
            "top_p": 0.9
        }
    }

    # 3. 初始化 Config 和 Dataset
    config = Config("config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    # Prompt
    template = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt=USER_PROMPT_TEMPLATE
    )

    # Pipeline
    pipeline = SequentialPipeline(config, template)

    if args.strategy == "baseline":
        print(">>> 正在执行 Baseline 模式 (Naive Run - 无检索)...")
        result = pipeline.naive_run(test_data, do_eval=True)
    else:
        print(f">>> 正在执行 {args.strategy} RAG 模式 (Run - 含检索)...")
        result = pipeline.run(test_data, do_eval=True)

    print(f"\n✅ 实验结束！结果已保存至: {current_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical RAG Experiment Runner")
    parser.add_argument("--strategy", type=str, required=True, 
                        choices=["baseline", "strict", "hybrid"],
                        help="选择实验策略: baseline(无RAG), strict(严格RAG), hybrid(混合RAG)")
    parser.add_argument("--gpu_id", type=str, default="0, 1", help="使用的 GPU ID，例如 '0,1'")
    parser.add_argument("--split", type=str, default="test", help="测试集切分名称")
    parser.add_argument("--batch_size", type=int, default=64, help="推理 Batch Size")
    
    args = parser.parse_args()

    main(args)
