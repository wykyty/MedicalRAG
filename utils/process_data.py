from datasets import load_dataset
import os

def get_data():
    data_dir = "data/Huatuo26M-Lite"
    os.makedirs(data_dir, exist_ok=True)

    # 下载Huatuo26-Lite train split, 作为检索库
    output = os.path.join(data_dir, "train.jsonl")
    ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split='train')
    ds.to_json(output, lines=True, force_ascii=False)

    # 下载test datasets
    output = os.path.join(data_dir, "test.jsonl")
    ds = load_dataset("FreedomIntelligence/huatuo26M-testdatasets", split='test')
    ds.to_json(output, lines=True, force_ascii=False)


import json
import os
from tqdm import tqdm 
input_path = "data/Huatuo26M-Lite/train.jsonl"
output_path = "data/indexes/corpus.jsonl"

def format_jsonl(input_file, output_file):
    print(f"开始处理: {input_file}")
    
    # 1. 计算总行数用于 tqdm 进度条
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1

    # 2. 开始转换
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        new_id = 1
        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                # 获取字段，使用 .get() 防止某个字段缺失导致报错
                question = data.get("question", "")
                answer = data.get("answer", "")
                label = data.get("label", "")
                related_diseases = data.get("related_diseases", "")
                
                # 构建新的 contents 字符串
                contents_str = (
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Label: {label}\n"
                    f"Related_diseases: {related_diseases}"
                )
                
                # 构建新的对象
                new_record = {
                    "id": new_id,
                    "contents": contents_str
                }
                
                # 写入新文件
                f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                
                new_id += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
                continue

    print(f"处理完成！\n输出文件位于: {output_file}")
    print(f"共处理条目: {new_id - 1}")

if __name__=="__main__":
    # get_data()
    # print("get data done.")

    format_jsonl(input_path, output_path)
    print("corpus builded.")