from datasets import load_dataset
import os

output = "/data/wyh/MedicalRAG/data/Huatuo26M-Lite"
os.makedirs(output, exist_ok=True)
output = os.path.join(output, "test.jsonl")

# ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split='train')
# ds.to_json(output, lines=True, force_ascii=False)

ds = load_dataset("FreedomIntelligence/huatuo26M-testdatasets", split='test')
ds.to_json(output, lines=True, force_ascii=False)

print("done.")