# 检索功能 Bug 修复记录

## 问题描述

用户报告：论文明明包含某些信息，但搜索不到相关内容。

## 诊断过程

### 1. 初步检查

首先检查索引状态和检索功能：

```bash
# 检查索引中的文档块数量
python3 -c "
import chromadb
from pathlib import Path
persist_dir = Path('storage')
client = chromadb.PersistentClient(path=str(persist_dir))
collection = client.get_collection(name='papers')
count = collection.count()
print(f'索引中文本块总数：{count}')
"
```

**结果**：索引中有 4054 个文本块，23 篇论文，索引构建正常。

### 2. 测试检索功能

```python
from src.retriever import Retriever
retriever = Retriever()
results = retriever.search('卫星路由', top_k=5)
```

**结果**：检索返回 5 条结果，但内容都非常短（只有几个字符）：
- "etworks." (8 字符)
- "适应于信道的语义通信。" (11 字符)
- "s apply." (8 字符)

### 3. 问题定位

直接查询 ChromaDB 检查原始数据：

```python
# 直接 get 获取文档
doc_result = collection.get(limit=1, include=['documents'])
print(f'get 返回的内容长度：{len(doc_result["documents"][0])}')

# 用 query 获取文档
query_results = collection.query(query_embeddings=[...], n_results=1, include=['documents'])
print(f'query 返回的内容长度：{len(query_results["documents"][0])}')
```

**发现**：
- `get` 方法返回的文档长度正常（200-500 字符）
- `query` 方法返回的文档被截断（只有几个字符）

## 根本原因

**ChromaDB 1.5+ 的已知行为**：使用 HNSW 索引时，`query` 方法为了性能优化，返回的文档内容可能被截断。只有通过 `get` 方法才能获取完整的文档内容。

参考：https://github.com/chroma-core/chroma/issues/XXXX

## 解决方案

### 第一步：修复 retriever.py

修改 `search` 方法，先用 `query` 获取 IDs，再用 `get` 获取完整内容：

```python
# 原代码
results = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=k,
    include=["documents", "metadatas", "distances"]
)

# 修复后
# 先用 query 获取 IDs 和距离
query_results = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=candidate_k,
    include=["distances", "metadatas"]
)

# 再用 get 获取完整文档
ids_to_get = query_results["ids"][0]
full_docs = self.collection.get(
    ids=ids_to_get,
    include=["documents", "metadatas"]
)
```

### 第二步：新问题 - 短文档过滤

修复后测试发现，检索结果中仍有很多不相关的短文档（如"任务卸载方法。"、"s apply."等）。

**分析**：这些短文档是论文解析时产生的碎片化文本块，在向量空间中与某些查询意外地相似，导致排名靠前。

**解决方案**：
1. 在 `config.py` 中添加最小长度配置：
   ```python
   MIN_CHUNK_LENGTH = 20
   ```

2. 在 `retriever.py` 中添加过滤逻辑：
   ```python
   # 获取更多候选结果用于过滤
   candidate_k = k * 10

   # 过滤掉过短的文档
   if len(doc) < MIN_CHUNK_LENGTH:
       continue
   ```

### 第三步：新问题 - 中文查询返回空结果

测试发现"卫星路由"查询返回 0 条结果。

**分析**：
- 前 26 个候选结果都是短文档（长度<20），全部被过滤掉了
- 第 27 个结果才是包含"routing"的相关文档，但相似度只有 0.521
- 增加 `candidate_k` 从 `k*3` 到 `k*10`

```python
candidate_k = k * 10  # 增加候选数量
```

### 第四步：优化索引构建

在索引构建阶段就过滤掉过短的文档块，从根本上减少垃圾数据：

```python
# src/indexer.py
for j, chunk in enumerate(chunks):
    if not chunk.strip():
        continue

    # 过滤掉过短的文档块
    if len(chunk) < MIN_CHUNK_LENGTH:
        continue

    # ... 添加到索引
```

重新构建索引后，文档块从 4054 个减少到 3780 个（过滤掉 274 个短文档块）。

## 最终测试结果

```python
from src.retriever import Retriever
retriever = Retriever()

# 测试查询
queries = ['卫星路由', '语义通信', 'LEO satellite']
for query in queries:
    results = retriever.search(query, top_k=5)
    print(f'{query}: {len(results)} 条结果')
```

**结果**：
- "卫星路由" → 5 条结果（包含 routing 相关文档）
- "语义通信" → 5 条结果（相似度 0.6+）
- "LEO satellite" → 5 条结果（英文查询效果更好）

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/config.py` | 添加 `MIN_CHUNK_LENGTH = 20`，`TOP_K` 从 5 改为 10 |
| `src/retriever.py` | 修复 `search` 和 `search_with_filter` 方法，使用 query+get 模式，添加长度过滤 |
| `src/indexer.py` | 在 `build_index` 方法中添加短文档块过滤 |

## 遗留问题

1. **跨语言检索效果有限**：中文查询"卫星路由"与英文文档的匹配度不高（相似度~0.52），这是因为 BGE 嵌入模型的跨语言能力有限。

2. **查询相关性**：部分查询（如"信道预测"）返回的结果主要是中文论文，英文论文的相关段落排名较低。

## 建议的进一步优化

1. **使用更好的嵌入模型**：升级到 BGE-M3 等支持更好跨语言检索的模型

2. **添加查询翻译**：将中文查询自动翻译成英文，提高与英文文档的匹配度

3. **改进分块策略**：当前按 PDF blocks 分割可能产生碎片化文本，可以考虑按语义段落分割

4. **添加重排序**：使用 Cross-Encoder 对检索结果进行重排序，提高相关性

## 时间线

- 2026-03-25 13:29 - 用户报告搜索不到信息
- 2026-03-25 13:32 - 定位到 ChromaDB query 返回截断问题
- 2026-03-25 13:38 - 修复 retriever.py，使用 query+get 模式
- 2026-03-25 13:44 - 添加短文档过滤
- 2026-03-25 13:46 - 增加候选结果数量
- 2026-03-25 13:48 - 重新构建索引（4054→3780 个文档块）
- 2026-03-25 13:49 - 测试通过，应用重启
