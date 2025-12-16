export PYTHONPATH=/data/wyh/MedicalRAG/FlashRAG:$PYTHONPATH
export HF_ENDPOINT="https://hf-mirror.com"

CUDA_VISIBLE_DEVICES=4,5 python -m flashrag.retriever.index_builder \
    --retrieval_method bge \
    --model_path BAAI/bge-large-zh-v1.5 \
    --corpus_path /data/wyh/MedicalRAG/data/indexes/corpus.jsonl \
    --save_dir /data/wyh/MedicalRAG/data/indexes/huatuo_bge_index \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method cls \
    --faiss_type Flat 