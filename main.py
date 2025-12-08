import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from rag_pipeline import MedicalSafePipeline

# ================= 配置 =================
# 建议申请 DeepSeek API (兼容 OpenAI)，便宜且适合中文医学
API_KEY = "sk-xxxxxxxxxxxxxxx" 
BASE_URL = "https://api.deepseek.com"
PDF_PATH = "medical_guideline.pdf" # 请放入你的PDF文件
# =======================================

def init_system():
    print(">>> 1. 初始化向量数据库...")
    # 这里用简单的 FAISS 做演示，实际可换用 FlashRAG 支持的更高级索引
    if not os.path.exists(PDF_PATH):
        print("请先上传一个 PDF 文件用于测试！")
        return None, None

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(">>> 2. 初始化 LLM...")
    llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model_name="deepseek-chat")
    
    return vector_store, llm

def main():
    vector_store, llm = init_system()
    if not vector_store: return

    # 实例化 Pipeline
    pipeline = MedicalSafePipeline(vector_store, llm)

    # ================= 模拟大作业的“对比实验” =================
    test_query = "高血压患者能不能喝红酒？"
    
    print("\n" + "="*40)
    print("实验组 A: Baseline (无 Refiner)")
    print("="*40)
    res_a = pipeline.run(test_query, use_refiner=False)
    print(f"回答: {res_a['answer']}")
    print(f"评估: {res_a['evaluation']}")

    print("\n" + "="*40)
    print("实验组 B: Proposed Method (含 Safety Refiner)")
    print("="*40)
    res_b = pipeline.run(test_query, use_refiner=True)
    print(f"回答: {res_b['answer']}")
    print(f"评估: {res_b['evaluation']}")

    # 报告中可以将 res_a 和 res_b 的结果填入表格进行对比
    # 简单的结果对比打印
    print("\n" + "-"*30 + " 实验总结 " + "-"*30)
    print(f"Baseline 检索文档数: {res_a.get('raw_docs_count', 0)}")
    print(f"Refined  检索文档数: {len(res_b.get('context', '').split('[证据')) - 1} (经过筛选)")


if __name__ == "__main__":
    main()
