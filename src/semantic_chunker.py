"""
语义切分模块

基于 BGE 嵌入模型的语义相似度计算，实现智能文本切分。

核心思想：
1. 将文本按句子分割
2. 计算相邻句子的语义相似度
3. 在语义突变点（相似度低）处切分
4. 合并语义连续的句子成块

面试要点：
- 为什么用语义切分：解决固定大小切分的语义断裂问题
- 如何计算相似度：BGE 嵌入向量 + 余弦相似度
- 如何决策边界：相似度阈值 + 滑动窗口验证
"""
import re
import httpx
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE,
    EMBEDDING_MODEL,
    # 语义切分配置
    SEMANTIC_SIMILARITY_THRESHOLD,
    MIN_SENTENCES_PER_CHUNK,
    MAX_SENTENCES_PER_CHUNK,
    MIN_CHUNK_LENGTH,
)
from .parser import PaperSection
from .logger import get_logger

logger = get_logger("semantic_chunker")


@dataclass
class SemanticChunk:
    """语义切块数据"""
    content: str              # 切块内容
    sentences: List[str]      # 包含的句子列表
    start_sentence_idx: int   # 起始句子索引
    end_sentence_idx: int     # 结束句子索引
    metadata: Dict            # 元数据（来源、页码等）
    avg_internal_similarity: float = 0.0  # 块内平均相似度（用于评估质量）


class SentenceSplitter:
    """
    句子分割器

    将文本分割成句子，处理特殊情况（引用、公式、缩写等）
    """

    # 中文句子结束标点
    SENTENCE_ENDINGS = re.compile(r'[。！？!.?]+')

    # 需要保护的模式（避免在错误位置切分）
    PROTECTED_PATTERNS = [
        r'\d+\.\d+',           # 小数：3.14
        r'Fig\.\s*\d+',        # 图引用：Fig. 1
        r'Tab\.\s*\d+',        # 表引用：Tab. 1
        r'et al\.',            # 等：et al.
        r'i\.e\.',             # 即
        r'e\.g\.',             # 例如
        r'vs\.',               # 对比
        r'etc\.',              # 等等
    ]

    def split(self, text: str) -> List[str]:
        """
        将文本分割成句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        if not text or not text.strip():
            return []

        # 1. 保护特殊模式，用占位符替换
        protected = {}
        protected_text = text
        for i, pattern in enumerate(self.PROTECTED_PATTERNS):
            matches = re.findall(pattern, protected_text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PROTECTED_{i}_{len(protected)}__"
                protected[placeholder] = match
                protected_text = protected_text.replace(match, placeholder, 1)

        # 2. 按句子结束符分割
        # 先按换行符分块（保持段落边界）
        paragraphs = re.split(r'\n\s*\n', protected_text)

        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 在句子边界处分割
            para_sentences = self._split_paragraph(para)
            sentences.extend(para_sentences)

        # 3. 恢复保护的内容
        sentences = [
            self._restore_protected(s, protected)
            for s in sentences
            if s.strip()
        ]

        # 4. 过滤过短的句子（< 5 字符）
        sentences = [s for s in sentences if len(s.strip()) >= 5]

        return sentences

    def _split_paragraph(self, para: str) -> List[str]:
        """分割段落成句子"""
        sentences = []
        current = []

        # 按句子结束符分割
        parts = self.SENTENCE_ENDINGS.split(para)
        endings = self.SENTENCE_ENDINGS.findall(para)

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                current.append(part)
                if i < len(endings):
                    current.append(endings[i])
                sentences.append(''.join(current))
                current = []

        return sentences

    def _restore_protected(self, text: str, protected: Dict[str, str]) -> str:
        """恢复保护的内容"""
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original)
        return text.strip()


class SemanticBoundaryDetector:
    """
    语义边界检测器

    使用 BGE 嵌入模型计算句子间相似度，检测语义边界
    """

    def __init__(self):
        self.api_base = EMBEDDING_API_BASE
        self.api_key = EMBEDDING_API_KEY
        self.model = EMBEDDING_MODEL
        self._cache: Dict[str, List[float]] = {}  # 文本缓存

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量（1024 维）
        """
        # 检查缓存
        if text in self._cache:
            return self._cache[text]

        # 调用 BGE API
        try:
            response = httpx.post(
                f"{self.api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": [text]
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]

            # 缓存结果
            self._cache[text] = embedding
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"嵌入 API 调用失败：{e}")
            raise RuntimeError(f"嵌入生成失败：{e}")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        公式：cos(θ) = (A·B) / (||A|| * ||B||)

        Args:
            vec1: 向量 1
            vec2: 向量 2

        Returns:
            相似度 [-1, 1]，越接近 1 越相似
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))

    def compute_similarity_matrix(
        self,
        sentences: List[str]
    ) -> List[List[float]]:
        """
        计算句子相似度矩阵

        Args:
            sentences: 句子列表

        Returns:
            N×N 相似度矩阵
        """
        # 批量获取嵌入向量
        embeddings = [self.get_embedding(s) for s in sentences]

        # 计算相似度矩阵
        n = len(sentences)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                sim = self.cosine_similarity(embeddings[i], embeddings[j])
                matrix[i][j] = sim
                matrix[j][i] = sim  # 对称矩阵

        return matrix

    def detect_boundaries(
        self,
        sentences: List[str],
        threshold: float = None
    ) -> List[Tuple[int, float]]:
        """
        检测语义边界

        Args:
            sentences: 句子列表
            threshold: 相似度阈值（低于此值认为是边界）

        Returns:
            边界列表 [(索引，相似度), ...]
        """
        threshold = threshold or SEMANTIC_SIMILARITY_THRESHOLD

        if len(sentences) < 2:
            return []

        # 获取所有句子的嵌入
        embeddings = [self.get_embedding(s) for s in sentences]

        # 计算相邻句子相似度
        boundaries = []
        for i in range(len(sentences) - 1):
            sim = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < threshold:
                boundaries.append((i + 1, sim))  # 边界在 i 和 i+1 之间

        return boundaries


class SemanticChunker:
    """
    语义切分器

    主入口：将 PaperSection 列表切分成语义完整的文本块
    """

    def __init__(
        self,
        similarity_threshold: float = None,
        min_sentences: int = None,
        max_sentences: int = None,
    ):
        """
        初始化切分器

        Args:
            similarity_threshold: 语义相似度阈值（默认从配置读取）
            min_sentences: 每块最少句子数
            max_sentences: 每块最多句子数
        """
        self.similarity_threshold = similarity_threshold or SEMANTIC_SIMILARITY_THRESHOLD
        self.min_sentences = min_sentences or MIN_SENTENCES_PER_CHUNK
        self.max_sentences = max_sentences or MAX_SENTENCES_PER_CHUNK

        self.sentence_splitter = SentenceSplitter()
        self.boundary_detector = SemanticBoundaryDetector()

    def chunk(self, sections: List[PaperSection]) -> List[SemanticChunk]:
        """
        将论文章节切分成语义块

        Args:
            sections: 论文章节列表

        Returns:
            语义块列表
        """
        all_chunks = []

        for section in sections:
            # 1. 分割成句子
            sentences = self.sentence_splitter.split(section.content)

            if not sentences:
                continue

            if len(sentences) <= self.min_sentences:
                # 句子太少，直接作为一个块
                all_chunks.append(SemanticChunk(
                    content=section.content,
                    sentences=sentences,
                    start_sentence_idx=0,
                    end_sentence_idx=len(sentences),
                    metadata={
                        "source": section.metadata.get("source", "unknown"),
                        "file_path": section.metadata.get("file_path", ""),
                        "page_num": section.page_num,
                        "section_title": section.title,
                    }
                ))
                continue

            # 2. 检测语义边界
            boundaries = self.boundary_detector.detect_boundaries(
                sentences,
                self.similarity_threshold
            )

            # 3. 根据边界切分
            section_chunks = self._split_by_boundaries(
                sentences,
                boundaries,
                section.metadata,
                section.page_num,
                section.title
            )

            all_chunks.extend(section_chunks)

        logger.info(f"语义切分完成：{len(all_chunks)} 个块")
        return all_chunks

    def _split_by_boundaries(
        self,
        sentences: List[str],
        boundaries: List[Tuple[int, float]],
        metadata: Dict,
        page_num: int,
        section_title: str
    ) -> List[SemanticChunk]:
        """
        根据语义边界切分

        Args:
            sentences: 句子列表
            boundaries: 边界列表 [(索引，相似度), ...]
            metadata: 元数据
            page_num: 页码
            section_title: 章节标题

        Returns:
            语义块列表
        """
        chunks = []

        if not boundaries:
            # 没有检测到边界，按最大句子数切分
            return self._split_by_max_size(
                sentences, metadata, page_num, section_title
            )

        # 添加起始和结束边界
        boundary_indices = [0] + [b[0] for b in boundaries] + [len(sentences)]

        for i in range(len(boundary_indices) - 1):
            start = boundary_indices[i]
            end = boundary_indices[i + 1]

            chunk_sentences = sentences[start:end]

            if not chunk_sentences:
                continue

            # 检查块大小
            if len(chunk_sentences) > self.max_sentences:
                # 块太大，进一步分割
                sub_chunks = self._split_by_max_size(
                    chunk_sentences, metadata, page_num, section_title, start
                )
                chunks.extend(sub_chunks)
            else:
                # 创建块
                content = ' '.join(chunk_sentences)
                chunks.append(SemanticChunk(
                    content=content,
                    sentences=chunk_sentences,
                    start_sentence_idx=start,
                    end_sentence_idx=end,
                    metadata={
                        "source": metadata.get("source", "unknown"),
                        "file_path": metadata.get("file_path", ""),
                        "page_num": page_num,
                        "section_title": section_title,
                    },
                    avg_internal_similarity=self._compute_avg_similarity(
                        chunk_sentences
                    )
                ))

        return chunks

    def _split_by_max_size(
        self,
        sentences: List[str],
        metadata: Dict,
        page_num: int,
        section_title: str,
        offset: int = 0
    ) -> List[SemanticChunk]:
        """按最大句子数分割（当块太大时）"""
        chunks = []

        for i in range(0, len(sentences), self.max_sentences):
            chunk_sentences = sentences[i:i + self.max_sentences]
            content = ' '.join(chunk_sentences)

            chunks.append(SemanticChunk(
                content=content,
                sentences=chunk_sentences,
                start_sentence_idx=offset + i,
                end_sentence_idx=offset + i + len(chunk_sentences),
                metadata={
                    "source": metadata.get("source", "unknown"),
                    "file_path": metadata.get("file_path", ""),
                    "page_num": page_num,
                    "section_title": section_title,
                },
                avg_internal_similarity=self._compute_avg_similarity(chunk_sentences)
            ))

        return chunks

    def _compute_avg_similarity(self, sentences: List[str]) -> float:
        """计算块内平均相似度（用于质量评估）"""
        if len(sentences) < 2:
            return 1.0

        embeddings = [self.boundary_detector.get_embedding(s) for s in sentences]
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = self.boundary_detector.cosine_similarity(
                embeddings[i],
                embeddings[i + 1]
            )
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 1.0


# 兼容原有接口的函数
def split_text_semantic(
    text: str,
    metadata: Optional[Dict] = None,
    page_num: int = 0,
    section_title: str = ""
) -> List[Dict]:
    """
    语义切分函数（兼容原有 split_text 接口）

    Args:
        text: 待切分文本
        metadata: 元数据
        page_num: 页码
        section_title: 章节标题

    Returns:
        文本块列表（字典格式）
    """
    if metadata is None:
        metadata = {}

    # 创建临时 PaperSection
    section = PaperSection(
        title=section_title,
        content=text,
        page_num=page_num,
        metadata=metadata
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk([section])

    # 转换为字典格式
    return [
        {
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


if __name__ == "__main__":
    # 测试
    test_text = """
    深度学习是机器学习的一个子领域。它使用多层神经网络来学习数据的层次化表示。
    卷积神经网络（CNN）是深度学习中最常用的架构之一。CNN 在图像处理任务中表现出色。

    自然语言处理是人工智能的另一个重要领域。它关注计算机和人类语言之间的交互。
    Transformer 模型彻底改变了 NLP 领域。它引入了自注意力机制。
    """

    chunker = SemanticChunker()
    chunks = chunker.chunk([
        PaperSection(
            title="测试章节",
            content=test_text,
            page_num=0,
            metadata={"source": "test.pdf"}
        )
    ])

    print(f"切分成 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i + 1}:")
        print(f"  内容：{chunk.content[:100]}...")
        print(f"  句子数：{len(chunk.sentences)}")
        print(f"  内部相似度：{chunk.avg_internal_similarity:.3f}")