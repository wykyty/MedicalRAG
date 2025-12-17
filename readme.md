# MedicalRAG: 基于混合策略的医学领域大模型幻觉抑制研究

## 项目简介

本项目是《人工智能》课程期末大作业的实现代码。项目基于 **FlashRAG** 框架，针对医学领域的“模型幻觉”问题，构建了一个包含检索、优化、生成全流程的 RAG 系统。

系统核心特性：

- **检索器 (Retriever)**: 集成 **BAAI/bge-large-zh-v1.5** 模型，构建稠密向量索引，解决中文医学语义匹配问题。
    
- **数据源 (Corpus)**: 使用 **Huatuo-26M-Lite** 数据集，提供真实的医生问答知识库。
    
- **生成策略**: 实现了“检索优先 + 知识降级”的 **Hybrid Prompt** 策略，在保证准确性的同时兼顾回答的可用性。
    
- **架构**: 基于 FlashRAG 进行模块化改造，增加了 Safety Refiner 和自动化评估模块。
    

## 目录结构

```
MedicalRAG/
├── FlashRAG/               # 核心框架源码 (基于官方版修改，请使用此本地版本)
├── utils/
│   ├── build_bge_index.sh  # BGE 索引构建脚本
│   ├── process_data.py     # 数据预处理脚本
│   └── ...
├── indexes/                # 存放构建好的 FAISS 索引 (运行脚本后生成)
├── main.py                 # 主程序入口 (包含对比实验逻辑)
├── run.sh                  # 一键运行脚本
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明

```

## 环境安装

1. 建议使用 Python 3.10+ 环境：
    
    ```
    conda create -n medical_rag python=3.10
    conda activate medical_rag
    
    ```
    
2. 安装依赖：
    
    ```
    pip install -r requirements.txt
    ```
    
    _注意：本项目包含修改版的 FlashRAG 源码，无需 `pip install flashrag`，直接使用本地包即可。_
    

## 快速开始

### 第一步：准备数据与模型

使用脚本下载数据集：

```
python utils/process_data.py
```

确保你已经下载了 Huatuo 数据集，并将其放置在指定目录（如 `data/`）。

### 第二步：构建向量索引

本项目使用 BGE 模型构建索引。运行以下脚本：

```
bash utils/build_bge_index.sh
```

该脚本会自动加载 BGE 模型，对语料进行切分、编码，并保存 FAISS 索引到 `indexes/` 目录。

### 第三步：运行实验

配置 `main.py` 中的 API Key (支持 OpenAI 或 vLLM 本地部署接口)，然后运行：

```
python main.py
```

或者使用 Shell 脚本：

```
bash run.sh
```

程序将输出：

1. **Baseline (无 RAG)** 的回答。
    
2. **Strict RAG** 的回答。
    
3. **Hybrid RAG (Ours)** 的回答。
    
4. **Evaluator** 的自动打分结果。
    

# 实验配置说明

在 `main.py` 中可以修改以下关键参数：
    
- `RETRIEVAL_METHOD`: 设置为 'bge'。
    
- `TOP_K`: 检索召回数量，默认为 5。
    

## 致谢

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG "null")
    
- [Huatuo-26M](https://github.com/FreedomIntelligence/Huatuo-26M "null")
    
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding "null")