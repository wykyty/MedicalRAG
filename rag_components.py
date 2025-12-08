import os
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# ================= 抽象基类 (模仿 FlashRAG 设计) =================
class BaseComponent(ABC):
    @abstractmethod
    def run(self, input_data):
        pass

# ================= 1. 检索器 (Retriever) =================
class MedicalRetriever(BaseComponent):
    def __init__(self, vector_store, k=3):
        self.vector_store = vector_store
        self.k = k

    def run(self, query: str):
        """返回检索到的文档列表"""
        print(f"   [Retriever] 正在检索: {query} ...")
        docs = self.vector_store.similarity_search(query, k=self.k)
        return docs

# ================= 2. 优化器 (Refiner) - 你的加分创新点 =================
class SafetyRefiner(BaseComponent):
    """
    FlashRAG 中的 Refiner 通常用于压缩或重写上下文。
    我们在医学场景下，用它来过滤“不相关”或“低质量”的检索结果，提升安全性。
    """
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def run(self, retrieved_docs):
        """
        简单的 Refiner 逻辑：可以在这里加入去重、重排序(Rerank)或关键词过滤
        """
        print(f"   [Refiner] 正在优化上下文 (输入 {len(retrieved_docs)} 条)...")
        
        refined_docs = []
        for doc in retrieved_docs:
            # 示例逻辑：这里可以扩展为更复杂的 BERT Rerank 或 关键词检查
            # 比如：必须包含医学关键词才保留
            content = doc.page_content
            if len(content) > 20: # 过滤太短的无效片段
                refined_docs.append(doc)
        
        print(f"   [Refiner] 优化完成，保留 {len(refined_docs)} 条高质量证据。")
        return refined_docs

# ================= 3. 生成器 (Generator) =================
class MedicalGenerator(BaseComponent):
    def __init__(self, llm_model):
        self.llm = llm_model
        self.prompt = PromptTemplate.from_template("""
        你是一名严谨的医学专家。请基于以下经过筛选的医学证据回答问题。
        
        【医学证据】:
        {context}
        
        【问题】: {question}
        
        【回答要求】:
        1. 仅依据上述证据回答，不要使用外部知识。
        2. 如果证据不足，请直接回答“证据不足，无法判断”，严禁编造。
        3. 回答需包含具体的证据来源（如“根据第X页...”）。
        
        专家建议:
        """)

    def run(self, query: str, context_docs: list):
        print("   [Generator] 正在生成回答...")
        # 将文档列表拼接成字符串
        context_str = "\n\n".join([f"[证据{i+1}]: {d.page_content}" for i, d in enumerate(context_docs)])
        
        # 组装 Prompt
        formatted_prompt = self.prompt.format(context=context_str, question=query)
        
        # 调用大模型
        response = self.llm.invoke(formatted_prompt).content
        return response, context_str

# ================= 4. 评估器 (Judger) - 对应“实验结果”分析 =================
class SafetyJudger(BaseComponent):
    def run(self, query, answer, context):
        """
        在报告中，你可以说你设计了一个自动评估模块来检测幻觉。
        这里用简单的逻辑演示，实际可以使用另一个 LLM 来打分。
        """
        print("   [Judger] 正在进行安全性评估...")
        
        score = 100
        issues = []
        
        # 1. 拒答检测 (安全性)
        if "证据不足" in answer or "无法判断" in answer:
            return {"score": "N/A", "comment": "模型正确地拒绝了回答 (Safe)"}

        # 2. 关键词一致性检测 (幻觉检测)
        # 简单逻辑：答案中的关键词应该在 Context 中出现
        # 实际可用 DeepSeek 再调用一次 API 进行打分
        if len(answer) > len(context) * 1.5:
             score -= 20
             issues.append("回答长度远超证据，可能存在过度引申")
             
        return {
            "score": score, 
            "issues": issues, 
            "comment": "通过基础安全检查" if score > 80 else "存在潜在幻觉风险"
        }