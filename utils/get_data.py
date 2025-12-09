from datasets import load_dataset
import os

output = "/data/wyh/MedicalRAG/data/Huatuo26M-Lite"
os.makedirs(output, exist_ok=True)
output = os.path.join(output, "train.jsonl")

ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split='train')
ds.to_json(output, lines=True, force_ascii=False)

print("done.")