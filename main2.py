import os
import sys
# 确保能找到同级目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from rag_pipeline import MedicalSafePipeline

# ================= 配置区域 (针对 vLLM 本地部署修改) =================
# 1. 仍然选择 'api' 模式！
# 因为 vLLM 实际上是把自己伪装成了一个 API 服务器，所以对 Python 代码来说它就是个 API。
MODE = 'api' 

# 2. vLLM 的标准配置
# API Key: vLLM 本地默认不需要 Key，但 LangChain 强制要求填一个，填 "EMPTY" 即可
API_KEY = "EMPTY" 

# Base URL: 指向你本地的 vLLM 服务地址，注意后面要加 /v1
BASE_URL = "http://localhost:8000/v1" 

# Model Name: 这里必须填你 vLLM 启动时加载的模型名称！
# 如果你启动的是 Qwen1.5-7B，这里就填对应的名字，如果不确定，可以通过 curl http://localhost:8000/v1/models 查看
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat" 

# 3. 数据路径
PDF_PATH = "medical_guideline.pdf" 
# ===============================================================

def init_system():
    print(">>> 1. 初始化向量数据库...")
    if not os.path.exists(PDF_PATH):
        print(f"错误：请在当前目录下放入一个名为 {PDF_PATH} 的医学指南 PDF 文件！")
        # 为了演示，如果文件不存在，我们创建一个假的向量库（避免报错退出，方便你调试流程）
        from langchain.docstore.document import Document
        docs = [Document(page_content="这是一个测试文档。高血压患者应控制盐摄入。")]
    else:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

    # 文本切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # 初始化 Embedding
    # 注意：Embedding 模型还是在本地运行（CPU/GPU），它很小，不会占用太多资源
    print("    正在加载 Embedding 模型 (sentence-transformers)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f">>> 2. 连接本地 vLLM 服务 ({BASE_URL})...")
    # LangChain 会自动通过这个类去连接你的 vLLM 端口
    llm = ChatOpenAI(
        openai_api_key=API_KEY, 
        openai_api_base=BASE_URL, 
        model_name=MODEL_NAME,
        temperature=0.1, # 医学问题保持低温度
        max_tokens=2048
    )
    
    return vector_store, llm

def main():
    try:
        vector_store, llm = init_system()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("提示: 请确保你已经在另一个终端启动了 vLLM 服务，命令参考：")
        print("python -m vllm.entrypoints.openai.api_server --model <your-model> --port 8000")
        return

    # 实例化 Pipeline
    pipeline = MedicalSafePipeline(vector_store, llm)

    # ================= 模拟大作业的“对比实验” =================
    test_query = "根据指南，高血压患者能不能喝红酒？"
    
    print("\n" + "="*40)
    print("实验组 A: Baseline (无 Safety Refiner)")
    print("="*40)
    # 第一次运行：不经过优化器，模拟普通 RAG
    res_a = pipeline.run(test_query, use_refiner=False)
    print(f"\n[AI 回答]:\n{res_a['answer']}\n")
    print(f"[评估结果]: {res_a['evaluation']}")

    print("\n" + "="*40)
    print("实验组 B: 本文方法 (引入 Safety Refiner)")
    print("="*40)
    # 第二次运行：经过优化器，模拟你的改进方法
    res_b = pipeline.run(test_query, use_refiner=True)
    print(f"\n[AI 回答]:\n{res_b['answer']}\n")
    print(f"[评估结果]: {res_b['evaluation']}")

    # 简单的结果对比打印
    print("\n" + "-"*30 + " 实验总结 " + "-"*30)
    print(f"Baseline 检索文档数: {res_a.get('raw_docs_count', 0)}")
    print(f"Refined  检索文档数: {len(res_b.get('context', '').split('[证据')) - 1} (经过筛选)")

if __name__ == "__main__":
    main()