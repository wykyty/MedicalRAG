export HF_ENDPOINT="https://hf-mirror.com"

# python utils/process_data.py

# ./utils/build_bge_index.sh

#### 1. 运行 Baseline (纯模型，无检索)
# python main.py --strategy baseline --split test

# #### 2. 运行 Strict RAG (严格检索)
python main.py --strategy strict --split test --gpu_id 4,5

# #### 3. 运行 Hybrid RAG (混合策略)
# python main.py --strategy hybrid --split test