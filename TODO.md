# 论文检索 RAG 应用 - 技术栈选型与任务清单

## 一、技术栈选型总览

| 组件 | 选型 | 状态 |
|------|------|------|
| LLM | DeepSeek API | ✅ 已确定 |
| RAG 框架 | LlamaIndex | ✅ |
| 向量数据库 | ChromaDB | ✅ |
| 嵌入模型 | OpenAI Compatible (BGE) | ✅ |
| 前端界面 | Streamlit | ✅ |
| 文档解析 | LlamaIndex PDF Parser + PyMuPDF | ✅ |

---

## 二、技术栈详细对比

### 1. LLM - DeepSeek API

| 方案 | 优势 | 劣势 |
|------|------|------|
| **DeepSeek** | - 成本极低 ($0.27/1M tokens 输入)<br>- 中文理解能力强<br>- 支持 64K 上下文<br>- OpenAI 兼容接口 | - 国际知名度较低<br>- 生态工具相对较少 |
| Claude API | - 推理能力强<br>- 长上下文 (200K)<br>- 安全性高 | - 成本高 ($3-15/1M tokens)<br>- 需要外币支付 |
| GPT-4 | - 生态最完善<br>- 多语言支持好 | - 成本高<br>- 中文场景不如 DeepSeek 性价比 |
| 本地部署 (Qwen/Llama) | - 数据完全私有<br>- 无 API 调用成本 | - 需要 GPU 资源<br>- 运维复杂 |

**选择原因**: 用户已指定 DeepSeek，且其性价比极高，适合论文检索这种需要大量 token 消耗的场景。

---

### 2. RAG 框架 - LlamaIndex

| 方案 | 优势 | 劣势 |
|------|------|------|
| **LlamaIndex** | - 专为 RAG 设计，功能全面<br>- 丰富的数据连接器<br>- 多种检索策略 (向量/关键词/混合)<br>- 完善的查询引擎<br>- 与 LangChain 可互操作 | - 学习曲线稍陡<br>- 文档量大，需要时间掌握 |
| **LangChain** | - 生态更大，社区活跃<br>- 通用型框架，支持更多场景<br>- 更多集成 | - RAG 专精程度不如 LlamaIndex<br>- 抽象层次高，调试困难 |
| **Haystack** | - 工程化好<br>- 支持问答管道 | - 社区较小<br>- 更新频率低 |
| **自研** | - 完全可控<br>- 无依赖负担 | - 开发周期长<br>- 需要处理大量边缘情况 |

**选择原因**: LlamaIndex 在 RAG 场景下更专精，特别是论文检索这种需要复杂检索策略的场景，其 `RecursiveRetriever`、`RouterQueryEngine` 等高级特性非常适用。

---

### 3. 向量数据库 - ChromaDB

| 方案 | 优势 | 劣势 |
|------|------|------|
| **ChromaDB** | - 零配置，开箱即用<br>- 纯 Python，无需额外服务<br>- 支持持久化<br>- 轻量级，适合开发和本地部署<br>- 内存占用低 | - 分布式能力弱<br>- 高并发性能一般<br>- 不适合大规模生产 |
| **Qdrant** | - 性能优秀<br>- 支持分布式<br>- 丰富的过滤查询<br>- Rust 实现，效率高 | - 需要独立服务<br>- 部署复杂度稍高 |
| **Pinecone** | - 全托管服务<br>- 扩展性好<br>- 维护成本低 | - 收费<br>- 数据需要出境 |
| **FAISS** | - Facebook 出品，成熟稳定<br>- 性能极佳 | - 仅支持内存<br>- 需要自己实现持久化<br>- API 较底层 |
| **Milvus** | - 功能全面<br>- 支持分布式 | - 架构复杂<br>- 资源消耗大 |

**选择原因**: 论文检索场景数据量通常在万级以下，ChromaDB 完全够用，且零配置特性让开发更专注业务逻辑。后续如需升级可轻松迁移到 Qdrant。

---

### 4. 嵌入模型 - BGE (通过 OpenAI 兼容接口)

| 方案 | 优势 | 劣势 |
|------|------|------|
| **BGE-Large-Zh** | - 中文效果 SOTA<br>- 开源免费<br>- 可本地部署 | - 需要自己部署服务 |
| **text-embedding-3-small** | - 英文效果好<br>- API 调用简单<br>- 成本低 | - 中文略逊于 BGE<br>- 需要 API Key |
| **M3E** | - 中文效果好<br>- 支持长文本 | - 部署稍复杂 |

**选择原因**: 考虑到论文可能包含中英文，采用硅基流动或其他服务商提供的 BGE 嵌入 API，通过 OpenAI 兼容接口调用，兼顾效果和便捷性。

---

### 5. 前端界面 - Streamlit

| 方案 | 优势 | 劣势 |
|------|------|------|
| **Streamlit** | - 纯 Python，无需前端知识<br>- 快速原型开发<br>- 丰富的组件库<br>- 内置聊天界面组件 | - 定制能力有限<br>- 不适合复杂交互 |
| **FastAPI + React** | - 完全定制<br>- 前后端分离<br>- 可扩展性强 | - 开发周期长<br>- 需要前端技术栈 |
| **Gradio** | - 更简单<br>- 适合 Demo | - 功能相对单一 |
| **Chainlit** | - 专为 LLM 应用设计<br>- 内置对话管理 | - 生态较新 |

**选择原因**: 论文检索应用核心是对话式查询，Streamlit 的 `st.chat_message` 和对话历史功能开箱即用，适合快速搭建。

---

### 6. 文档解析 - PyMuPDF + LlamaIndex Parser

| 方案 | 优势 | 劣势 |
|------|------|------|
| **PyMuPDF (fitz)** | - 解析速度快<br>- 支持文本+图片<br>- 保留格式好 | - 对复杂公式支持有限 |
| **pdfplumber** | - 表格解析好 | - 速度较慢 |
| **PyPDF2** | - 纯 Python | - 功能较基础 |
| **Marker** | - 支持公式识别<br>- 效果好 | - 需要额外依赖<br>- 速度慢 |

**选择原因**: PyMuPDF 在速度和效果之间取得平衡，对于大多数论文解析足够。后续可根据需要升级到 Marker 处理公式密集型论文。

---

## 三、任务清单

### Phase 1: 项目初始化
- [ ] 创建项目目录结构
- [ ] 编写 requirements.txt
- [ ] 配置环境变量模板 (.env.example)
- [ ] 初始化 Git 仓库 (可选)

### Phase 2: 核心功能开发
- [ ] 实现 PDF 文档加载和解析
- [ ] 实现文档分块 (Chunking) 策略
- [ ] 配置 DeepSeek LLM 连接
- [ ] 配置嵌入模型
- [ ] 构建 ChromaDB 向量索引
- [ ] 实现索引持久化
- [ ] 实现向量检索功能
- [ ] 实现检索结果重排序 (可选)

### Phase 3: 查询引擎
- [ ] 构建基础 QueryEngine
- [ ] 实现对话历史管理
- [ ] 实现引用溯源 (显示来源论文)
- [ ] 实现多轮对话支持

### Phase 4: 前端界面
- [ ] 搭建 Streamlit 基础框架
- [ ] 实现文件上传和索引功能
- [ ] 实现聊天界面
- [ ] 实现检索结果展示 (含来源标注)
- [ ] 实现历史记录管理

### Phase 5: 优化与测试
- [ ] 添加日志系统
- [ ] 性能优化 (批处理、缓存)
- [ ] 错误处理和用户提示
- [ ] 测试用例编写

### Phase 6: 部署 (可选)
- [ ] Docker 容器化
- [ ] 服务部署脚本

---

## 四、项目目录结构

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
├── .gitignore
└── README.md                # 项目说明
```

---

## 五、依赖列表 (requirements.txt)

```txt
# RAG 框架
llama-index>=0.10.0
llama-index-llms-openai-like
llama-index-embeddings-huggingface

# 向量数据库
chromadb>=0.4.0

# 文档解析
pymupdf>=1.23.0
llama-index-readers-file

# 嵌入模型
sentence-transformers

# 前端
streamlit>=1.28.0

# 工具
python-dotenv
pydantic
```

---

## 六、环境变量配置 (.env.example)

```bash
# DeepSeek API
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 嵌入模型 (可选：硅基流动/其他服务商)
EMBEDDING_API_KEY=xxx
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 项目配置
CHROMA_PERSIST_DIR=./storage
DATA_DIR=./data/papers
```

---

## 七、下一步

确认技术栈无误后，可以开始 **Phase 1: 项目初始化**。

如需调整任何技术选型，请告诉我具体需求。
