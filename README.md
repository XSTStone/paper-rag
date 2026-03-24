# 论文检索 RAG 应用

基于 LlamaIndex + ChromaDB + DeepSeek 的论文检索与问答系统。

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek API |
| RAG 框架 | LlamaIndex |
| 向量数据库 | ChromaDB |
| 嵌入模型 | BGE (OpenAI 兼容接口) |
| 前端界面 | Streamlit |
| 文档解析 | PyMuPDF |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `EMBEDDING_API_KEY`: 嵌入模型 API 密钥（如使用硅基流动等服务）

### 3. 准备论文数据

将 PDF 论文文件放入 `data/papers/` 目录。

### 4. 启动应用

```bash
streamlit run app.py
```

## 项目结构

```
paper_rag/
├── data/
│   └── papers/              # PDF 论文存储目录
├── storage/                 # 向量索引持久化目录
├── src/
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── parser.py            # PDF 解析
│   ├── indexer.py           # 索引构建
│   ├── retriever.py         # 检索模块
│   └── query_engine.py      # 查询引擎
├── app.py                   # Streamlit 主程序
├── requirements.txt         # 依赖列表
├── .env.example             # 环境变量模板
└── README.md                # 项目说明
```

## 功能特性

- PDF 论文自动解析与索引
- 向量相似度检索
- 多轮对话支持
- 引用溯源（显示来源论文）
- 对话历史记录
