"""检索模块"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
import httpx
import hashlib

from .config import CHROMA_PERSIST_DIR, TOP_K, EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL
from .logger import get_logger
from .cache import embedding_cache
from .errors import RetrievalError, handle_error

logger = get_logger("retriever")


@dataclass
class SearchResult:
    """检索结果"""
    content: str
    source: str
    page_num: int
    section_title: str
    score: float
    metadata: Dict


class Retriever:
    """向量检索器"""

    def __init__(self, persist_dir: Optional[str] = None, top_k: int = TOP_K):
        """
        初始化检索器

        Args:
            persist_dir: ChromaDB 持久化目录
            top_k: 返回的最大结果数
        """
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.top_k = top_k

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 获取 collection
        self.collection = self.client.get_collection(name="papers")

        # 配置嵌入模型（使用 OpenAI 兼容接口）
        self._embedding_fn = None
        self._configure_embedding()

    def _configure_embedding(self):
        """配置嵌入模型"""
        def embed_fn(texts: List[str]) -> List[List[float]]:
            """使用硅基流动 API 生成嵌入（带缓存）"""
            if not texts:
                return []

            logger.debug(f"生成嵌入：{len(texts)} 个文本")

            # 批量处理（每次最多 32 个文本）
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    # 检查缓存
                    cached_embeddings = []
                    uncached_indices = []

                    for j, text in enumerate(batch):
                        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
                        cached = embedding_cache.get(cache_key)
                        if cached:
                            cached_embeddings.append(cached)
                        else:
                            cached_embeddings.append(None)
                            uncached_indices.append(j)

                    # 调用 API 获取未缓存的嵌入
                    if uncached_indices:
                        uncached_batch = [batch[j] for j in uncached_indices]
                        response = httpx.post(
                            f"{EMBEDDING_API_BASE}/embeddings",
                            headers={
                                "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": EMBEDDING_MODEL,
                                "input": uncached_batch
                            },
                            timeout=30.0
                        )
                        response.raise_for_status()
                        data = response.json()
                        uncached_embeddings = [item["embedding"] for item in data["data"]]

                        # 缓存新嵌入
                        for idx, emb in zip(uncached_indices, uncached_embeddings):
                            cache_key = f"emb:{hashlib.md5(batch[idx].encode()).hexdigest()}"
                            embedding_cache.set(cache_key, emb)
                            cached_embeddings[idx] = emb
                    else:
                        uncached_embeddings = []

                    all_embeddings.extend(cached_embeddings)
                    logger.debug(f"批量 {i//batch_size + 1}: {len(cached_embeddings)} 个嵌入")

                except httpx.HTTPError as e:
                    logger.error(f"嵌入 API 调用失败：{e}")
                    raise RetrievalError(f"嵌入生成失败：{e}")

            return all_embeddings

        self._embedding_fn = embed_fn

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        try:
            k = top_k or self.top_k

            logger.info(f"检索查询：{query[:50]}... (top_k={k})")

            # 生成查询嵌入
            query_embedding = self._embedding_fn([query])[0]

            # 执行检索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # 解析结果
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # 余弦距离转相似度：similarity = 1 - distance
                    score = 1 - distance

                    search_results.append(SearchResult(
                        content=doc,
                        source=metadata.get("source", "unknown"),
                        page_num=metadata.get("page_num", 0),
                        section_title=metadata.get("section_title", ""),
                        score=score,
                        metadata=metadata
                    ))

            logger.info(f"检索到 {len(search_results)} 条结果")
            return search_results

        except Exception as e:
            logger.error(f"检索失败：{e}")
            raise RetrievalError(handle_error(e))

    def search_with_filter(
        self,
        query: str,
        source_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        带过滤条件的检索

        Args:
            query: 查询文本
            source_filter: 来源论文过滤
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        k = top_k or self.top_k

        # 生成查询嵌入
        query_embedding = self._embedding_fn([query])[0]

        # 构建过滤条件
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # 解析结果
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                search_results.append(SearchResult(
                    content=doc,
                    source=metadata.get("source", "unknown"),
                    page_num=metadata.get("page_num", 0),
                    section_title=metadata.get("section_title", ""),
                    score=score,
                    metadata=metadata
                ))

        return search_results

    def get_all_sources(self) -> List[str]:
        """获取所有已索引的论文来源"""
        results = self.collection.get(
            include=["metadatas"]
        )

        sources = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

        return sorted(list(sources))


if __name__ == "__main__":
    # 测试
    retriever = Retriever()
    results = retriever.search("论文主要贡献是什么？")
    print(f"检索到 {len(results)} 条结果")
    for r in results[:3]:
        print(f"- {r.source}: {r.content[:50]}... (score: {r.score:.3f})")
