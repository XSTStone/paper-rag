# 语义切分模块实现记录

## 面试技术文档

---

## 一、项目背景

### 1.1 原始问题

在论文 RAG 系统中，需要将 PDF 论文切分成文本块（chunk）以便：
- 构建向量索引
- 支持语义检索
- 作为 LLM 的上下文输入

### 1.2 初始方案：固定大小切分

**实现逻辑**（`src/indexer.py:split_text`）：

```python
def split_text(text: str, chunk_size=512, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 在句子边界处截断（如果在后半部分有句号）
        if end < len(text):
            for sep in ['.', '.', '!', '!', '?', '?', '\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[:last_sep + 1]
                    break

        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap  # 步长 462
    return chunks
```

**配置参数**：
- `CHUNK_SIZE = 512`：每块目标 512 字符
- `CHUNK_OVERLAP = 50`：块间重叠 50 字符（约 10%）
- `MIN_CHUNK_LENGTH = 20`：过滤过短块

---

## 二、固定大小切分的问题

### 2.1 语义断裂

```
示例文本：
"深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示。
卷积神经网络（CNN）是深度学习中最常用的架构之一，在图像处理任务中表现出色。

自然语言处理是人工智能的另一个重要领域，关注计算机和人类语言之间的交互。
Transformer 模型彻底改变了 NLP 领域，引入了自注意力机制。"

固定切分结果（512 字符）：
块 1: "深度学习是机器学习的一个子领域...表现出色。自然语言处理是人工" ← 问题！
块 2: "智能的另一个重要领域...自注意力机制。"

问题：块 1 包含了两个不相关主题的"半句话"
```

### 2.2 忽略文档结构

| 问题类型 | 说明 |
|---------|------|
| 章节边界 | 可能把章节标题和正文分开 |
| 段落边界 | 同一段落的内容被切到不同块 |
| 跨页内容 | 连续内容因分页被切断 |
| 公式/表格 | 结构化内容被打散 |

### 2.3 重叠的局限性

- 重叠只是机械复制，不能解决语义不完整
- 增加存储和计算开销（10% 冗余）

---

## 三、语义切分方案设计

### 3.1 核心思想

**"在语义边界处切分，保持块内主题一致性"**

```
文本 → 句子分割 → 嵌入向量 → 相似度计算 → 边界检测 → 合并成块
```

### 3.2 技术选型

| 组件 | 方案 | 理由 |
|------|------|------|
| 句子分割 | 正则 + 保护模式 | 轻量、可控、无需 API |
| 嵌入模型 | BGE-large-zh-v1.5 | 已有 API、中文优化、1024 维 |
| 相似度计算 | 余弦相似度（本地） | 公式简单、无需 API |
| 边界决策 | 阈值判断 | 可解释、可调参数 |

### 3.3 为什么选 BGE

**BAAI/bge-large-zh-v1.5** 参数：
- 维度：1024
- 语言：中文优化
- 训练数据：大规模中文语料
- 任务：语义相似度/检索

**对比其他方案**：

| 模型 | 维度 | 中文效果 | API 成本 |
|------|------|---------|---------|
| bge-large-zh | 1024 | 优秀 | 已有 |
| text-embedding-3-small | 1536 | 好 | $0.02/1M tokens |
| m3e-base | 768 | 很好 | 本地免费 |

---

## 四、实现细节

### 4.1 句子分割器（SentenceSplitter）

**文件**：`src/semantic_chunker.py`

```python
class SentenceSplitter:
    # 句子结束标点
    SENTENCE_ENDINGS = re.compile(r'[.!?!.?]+')

    # 保护模式（避免错误切分）
    PROTECTED_PATTERNS = [
        r'\d+\.\d+',      # 小数：3.14
        r'Fig\.\s*\d+',   # 图引用
        r'et al\.',       # 等
        r'i\.e\.',        # 即
    ]

    def split(self, text: str) -> List[str]:
        # 1. 保护特殊模式
        # 2. 按句子结束符分割
        # 3. 恢复保护内容
        # 4. 过滤过短句子
        return sentences
```

**为什么需要保护模式**：
```
错误示例：
"如图 3.14 所示，实验结果显示..."
→ 按 "." 分割 → "如图 3" 和 "14 所示..."

保护后：
"如图 __PROTECTED_0__ 所示，实验结果显示..."
→ 正确分割 → "如图 3.14 所示，实验结果显示..."
```

### 4.2 语义边界检测器（SemanticBoundaryDetector）

**嵌入向量获取**：
```python
def get_embedding(self, text: str) -> List[float]:
    response = httpx.post(
        f"{self.api_base}/embeddings",
        json={"model": self.model, "input": [text]}
    )
    return response.json()["data"][0]["embedding"]  # 1024 维
```

**余弦相似度计算**：
```python
@staticmethod
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    公式：cos(θ) = (A·B) / (||A|| * ||B||)

    结果范围：[-1, 1]
    - 1: 完全相同
    - 0: 无关
    - -1: 完全相反
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

**边界检测逻辑**：
```python
def detect_boundaries(self, sentences: List[str]) -> List[Tuple[int, float]]:
    embeddings = [self.get_embedding(s) for s in sentences]
    boundaries = []

    for i in range(len(sentences) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:  # 默认 0.65
            boundaries.append((i + 1, sim))  # 边界位置

    return boundaries
```

### 4.3 语义切分器（SemanticChunker）

**主流程**：
```python
def chunk(self, sections: List[PaperSection]) -> List[SemanticChunk]:
    for section in sections:
        # 1. 分割成句子
        sentences = self.sentence_splitter.split(section.content)

        # 2. 检测语义边界
        boundaries = self.boundary_detector.detect_boundaries(sentences)

        # 3. 根据边界切分
        chunks = self._split_by_boundaries(sentences, boundaries)

    return chunks
```

**配置参数**：
```python
# config.py
SEMANTIC_SIMILARITY_THRESHOLD = 0.65  # 相似度阈值
MIN_SENTENCES_PER_CHUNK = 2           # 最少 2 句/块
MAX_SENTENCES_PER_CHUNK = 8           # 最多 8 句/块
```

---

## 五、两种方案对比

### 5.1 效果对比

| 维度 | 固定大小切分 | 语义切分 |
|------|------------|---------|
| **语义完整性** | ❌ 可能断裂 | ✅ 保持主题一致 |
| **块大小可控** | ✅ 精确控制 | ⚠️ 动态变化 |
| **API 调用** | ✅ 无 | ⚠️ 每句子 1 次 |
| **计算开销** | ✅ 低 | ⚠️ 中等 |
| **可解释性** | ✅ 简单 | ✅ 可解释（相似度） |
| **跨段落处理** | ❌ 忽略 | ✅ 自动检测 |
| **参数敏感** | ⚠️  chunk_size | ⚠️ 阈值 |

### 5.2 成本分析

**假设**：一篇论文 10 页，约 100 个句子

| 方案 | API 调用 | 成本估算 |
|------|---------|---------|
| 固定切分 | 0 | ￥0 |
| 语义切分 | ~100 次嵌入 | ￥0.1~0.2/篇 |

**硅基流动 API 价格**（参考）：
- BGE-large-zh: ￥0.001/次（1000 tokens 内）

### 5.3 示例对比

```
输入文本：
"深度学习是机器学习的一个子领域。它使用多层神经网络。
卷积神经网络是常用架构。它在图像处理中表现出色。

自然语言处理是另一领域。它关注人机语言交互。
Transformer 模型改变了 NLP。它引入自注意力机制。"

固定切分（512 字符）：
块 1: "深度学习...子领域。它使用...神经网络。卷积...架构。它...表现出色。自然...领域。它..."
      ↑ 问题：两个主题混在一起

语义切分（阈值 0.65）：
检测到边界：句子 4 和 5 之间（相似度 0.42）
块 1: "深度学习...表现出色。"（主题：深度学习/CNN）
块 2: "自然语言处理...自注意力机制。"（主题：NLP/Transformer）
      ↑ 优势：主题分离
```

---

## 六、实现文件清单

```
src/
├── config.py              # 新增语义切分配置
├── indexer.py             # 修改：集成语义切分
├── semantic_chunker.py    # 新增：核心语义切分模块
│   ├── SentenceSplitter          # 句子分割器
│   ├── SemanticBoundaryDetector  # 边界检测器
│   └── SemanticChunker           # 主切分器
└── parser.py              # 无修改
```

### 6.1 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 固定切分 | `indexer.py` | 16-48 |
| 智能切换 | `indexer.py` | 51-72 |
| 句子分割 | `semantic_chunker.py` | 34-80 |
| 相似度计算 | `semantic_chunker.py` | 126-145 |
| 边界检测 | `semantic_chunker.py` | 147-175 |
| 主切分逻辑 | `semantic_chunker.py` | 178-260 |

---

## 七、面试可能的问题

### Q1: 为什么不用 LLM 直接切分？

**答**：
- 成本：LLM 调用成本是嵌入模型的 10-100 倍
- 速度：LLM 响应慢，不适合批量处理
- 可控性：阈值判断比 LLM 输出更可控
- （可选）模糊边界可以调用 LLM 辅助，但默认不需要

### Q2: 阈值 0.65 怎么来的？

**答**：
- 经验值：BGE 官方推荐 0.6-0.7 范围
- 可调参数：根据实际效果调整
- 可评估：通过块内相似度分布选择最优值

### Q3: 如果 API 失败怎么办？

**答**：
- 降级：自动回退到固定切分
- 缓存：已计算的嵌入向量缓存
- 重试：httpx 超时重试机制

### Q4: 如何处理超长句子？

**答**：
- MAX_SENTENCES_PER_CHUNK 限制块大小
- 超过限制时强制分割
- 可在句子分割器中添加长度限制

### Q5: 语义切分的效果如何评估？

**答**：
- 块内平均相似度（应 > 0.7）
- 相邻块相似度（应 < 阈值）
- 检索效果对比（MRR、NDCG）
- 人工抽检主题完整性

---

## 八、使用方式

### 8.1 启用语义切分

```python
# config.py
USE_SEMANTIC_CHUNKING = True  # 设为 True 启用
SEMANTIC_SIMILARITY_THRESHOLD = 0.65
```

### 8.2 独立使用

```python
from src.semantic_chunker import SemanticChunker, PaperSection

chunker = SemanticChunker()
chunks = chunker.chunk([
    PaperSection(
        title="引言",
        content="深度学习是...",
        page_num=0,
        metadata={"source": "paper.pdf"}
    )
])

for chunk in chunks:
    print(f"内容：{chunk.content}")
    print(f"句子数：{len(chunk.sentences)}")
    print(f"内部相似度：{chunk.avg_internal_similarity:.3f}")
```

---

## 九、后续优化方向

- [ ] 公式/表格保护（特殊内容不切分）
- [ ] 跨页章节合并
- [ ] 层次化切分（章→节→段）
- [ ] 本地 BGE 模型（避免 API 调用）
- [ ] LLM 辅助模糊边界判断

---

## 十、技术亮点总结

1. **复用现有资源**：直接使用已有 BGE API，零额外成本
2. **纯本地计算**：相似度计算无需 API，仅 numpy
3. **可解释性强**：每个切分点有相似度分数
4. **可配置参数**：阈值、句子数限制均可调
5. **降级方案**：API 失败可回退到固定切分
6. **质量评估**：每个块输出内部相似度指标

---

*文档生成时间：2026-03-25*
*项目：Paper RAG - 语义切分模块*