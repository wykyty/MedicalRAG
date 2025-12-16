import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
flashrag_path = '/data/wyh/MedicalRAG/FlashRAG'
sys.path.insert(0, flashrag_path)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

BaselinePrompt = """
你是一名经验丰富的医学专家助手。请利用你所掌握的专业医学知识来回答用户的问题。

【用户当前问题 (User Question)】: 
{question}

【你的专家建议】:
"""

Prompt = """
你是一名经验丰富的医学专家助手。为了回答用户的问题，系统检索了数据库中相似的历史问答（Reference Cases）供你参考。

请按照以下步骤进行回答：
1. **分析参考案例**：仔细阅读提供的历史问答，关注 Question（相似问题）和 Answer（专家解答）。
2. **核对疾病领域**：检查 Related_diseases 和 Label，确保参考案例与用户问题的疾病领域一致。
3. **综合生成**：基于参考案例中的医学知识，针对用户的具体问题生成准确、通顺的回答。

【注意】：
- 如果参考案例与用户问题**完全无关**，请忽略该案例。
- 如果所有案例都无关，请诚实回答“  ”。
- 严禁根据记忆编造药物剂量或治疗方案，必须基于参考信息。

---
【历史参考案例 (Reference Cases)】:
{reference}

"""

def main(config_dict):
    # preparation
    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["30"]

    templete = PromptTemplate(
        config,
        system_prompt=Prompt,
        user_prompt=""""
【用户当前问题 (User Question)】: 
{question}

【你的专家建议】:
        """
    )
    pipeline = SequentialPipeline(config, templete)
    result = pipeline.run(test_data, do_eval=True) 
    # result = pipeline.naive_run(test_data, do_eval=True)

if __name__ == "__main__":

    save_dir = "/data/wyh/MedicalRAG/output"
    config_dict = {
        "data_dir": "/data/wyh/MedicalRAG/data",
        "dataset_name": "Huatuo26M-Lite",
        "split": "30",

        # framework and metrics
        "framework" : "host", 
        "metrics": ['gpt_score', 'gpt_hallucination_rate', 'gpt_harmful_rate'],
      
        "api_setting": {
            "model_name": "gpt-4o-mini",
            "concurrency": 64,
            "timeout_sec": 60,
        },

        "index_path": "/data/wyh/MedicalRAG/data/indexes/huatuo_bge_index/bge_Flat.index",
        "corpus_path": "/data/wyh/MedicalRAG/data/indexes/corpus.jsonl",
        "retrieval_method": "bge",
        "retrieval_topk": 5,
        "gpu_id": "3,5", 
        "gpu_num" : 4,
        "gpu_memory_utilization" : 0.8,
        "save_dir" : save_dir
    }
    main(config_dict)
