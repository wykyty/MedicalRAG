import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
flashrag_path = '/data/wyh/RAG-Safer-Code/FlashRAG'
sys.path.insert(0, flashrag_path)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

BaselinePrompt = """
你是一名经验丰富的医学专家助手。请利用你所掌握的专业医学知识来回答用户的问题。

请按照以下步骤进行回答：
1. **分析用户描述**：仔细阅读用户的问题，提取关键症状、疾病名称或咨询意图。
2. **调用专业知识**：基于你的医学训练数据，确认该问题所属的医学领域及相关治疗原则。
3. **综合生成**：针对用户的具体问题生成准确、通顺的回答。

【注意】：
- 你的回答必须严谨、客观。
- 如果你对该问题**不确定**或**知识储备不足**，请诚实回答“我的知识库中缺乏相关确切信息，无法回答”，严禁强行作答。
- **严禁根据模糊记忆编造**具体的药物剂量、手术方案或临床数据，这可能危害用户健康。

---
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
        "metrics": ['em','f1','acc'],

        #  api setting        
        "api_setting": {
            "model_name": "gpt-4o-mini",
            "concurrency": 64,
            "timeout_sec": 60,
        },

        # retrieval
        "index_path": "/data/wyh/MedicalRAG/data/indexes/huatuo_bge_index/bge_Flat.index",
        "corpus_path": "/data/wyh/MedicalRAG/data/Huatuo26M-Lite/corpus.jsonl",
        "retrieval_method": "bge",
        "retrieval_topk": 5,
        # "bm25_backend" : "bm25s",
        
        # others
        "gpu_id": "3,5", 
        "gpu_num" : 4,
        "gpu_memory_utilization" : 0.8,
        "save_dir" : save_dir
    }
    main(config_dict)
