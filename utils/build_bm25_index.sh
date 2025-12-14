python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /data/wyh/MedicalRAG/data/Huatuo26M-Lite/corpus.jsonl \
    --bm25_backend bm25s \
    --save_dir /data/wyh/MedicalRAG/data/indexes/huatuo_bm25_index