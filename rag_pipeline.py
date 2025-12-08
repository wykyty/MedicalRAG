from rag_components import MedicalRetriever, SafetyRefiner, MedicalGenerator, SafetyJudger

class MedicalSafePipeline:
    """
    模仿 FlashRAG 的 Pipeline 结构:
    Query -> Retrieve -> Refine -> Generate -> Judge
    """
    def __init__(self, vector_store, llm):
        # 初始化各个模块
        self.retriever = MedicalRetriever(vector_store)
        self.refiner = SafetyRefiner() # 这就是你对比实验的变量：有无 Refiner 的区别
        self.generator = MedicalGenerator(llm)
        self.judger = SafetyJudger()

    def run(self, query, use_refiner=True):
        """
        执行完整的 RAG 流程
        :param use_refiner: 控制变量，用于做对比实验 (A/B Test)
        """
        result_log = {"query": query}
        
        # 1. Retrieve
        raw_docs = self.retriever.run(query)
        result_log["raw_docs_count"] = len(raw_docs)
        
        # 2. Refine (核心实验点)
        if use_refiner:
            context_docs = self.refiner.run(raw_docs)
            result_log["strategy"] = "With Safety Refiner"
        else:
            print("   [Pipeline] 跳过 Refiner (Baseline模式)...")
            context_docs = raw_docs
            result_log["strategy"] = "Baseline (No Refiner)"
            
        # 3. Generate
        answer, context_str = self.generator.run(query, context_docs)
        result_log["answer"] = answer
        result_log["context"] = context_str
        
        # 4. Judge (自动评估)
        eval_result = self.judger.run(query, answer, context_str)
        result_log["evaluation"] = eval_result
        
        return result_log