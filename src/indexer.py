"""索引构建模块"""
from pathlib import Path
from typing import List, Optional
import chromadb
import httpx
import hashlib

from .config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL
from .parser import parse_all_papers, PaperSection
from .logger import get_logger
from .errors import IndexError, handle_error

logger = get_logger("indexer")


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    将文本分割成块

    Args:
        text: 待分割的文本
        chunk_size: 每块大小
        chunk_overlap: 块间重叠大小

    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 如果不是最后一块，尝试在句子边界处分割
        if end < text_len:
            # 寻找最近的句子结束符
            for sep in ['。', '.', '！', '!', '？', '?', '\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[:last_sep + 1]
                    break

        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap

    return chunks


class Indexer:
    """向量索引构建器"""

    def __init__(self, persist_dir: Optional[str] = None):
        """
        初始化索引构建器

        Args:
            persist_dir: 持久化目录，None 则使用默认配置
        """
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

        # 配置嵌入模型 API
        self._embedding_fn = self._configure_embedding()

    def _configure_embedding(self):
        """配置嵌入模型 API"""
        def embed_fn(texts: List[str]) -> List[List[float]]:
            """使用智谱 AI API 生成嵌入"""
            if not texts:
                return []

            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    response = httpx.post(
                        f"{EMBEDDING_API_BASE}/embeddings",
                        headers={
                            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": EMBEDDING_MODEL,
                            "input": batch
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(embeddings)
                    logger.debug(f"批量嵌入：{len(batch)} 个文本")
                except httpx.HTTPError as e:
                    logger.error(f"嵌入 API 调用失败：{e}")
                    raise IndexError(f"嵌入生成失败：{e}")

            return all_embeddings

        return embed_fn

    def build_index(self, sections: List[PaperSection]) -> int:
        """
        构建向量索引

        Args:
            sections: 论文章节列表

        Returns:
            添加到索引的文档数量
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []
        all_embeddings = []

        for i, section in enumerate(sections):
            # 分割文本
            chunks = split_text(section.content)

            for j, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_id = f"{section.metadata.get('source', 'unknown')}_{section.page_num}_{i}_{j}"
                metadata = {
                    "source": section.metadata.get("source", "unknown"),
                    "file_path": section.metadata.get("file_path", ""),
                    "page_num": section.page_num,
                    "section_title": section.title
                }

                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append(metadata)

        # 批量生成嵌入向量（使用智谱 AI API）
        if all_chunks:
            logger.info(f"正在生成 {len(all_chunks)} 个文本块的嵌入向量...")
            print(f"正在生成嵌入向量（共 {len(all_chunks)} 个文本块，可能需要几分钟）...")
            all_embeddings = self._embedding_fn(all_chunks)
            logger.info(f"嵌入向量生成完成")

        # 清空现有索引（避免重复）
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        # 批量添加到索引（带嵌入向量）
        if all_chunks and all_embeddings:
            self.collection.add(
                documents=all_chunks,
                ids=all_ids,
                metadatas=all_metadatas,
                embeddings=all_embeddings
            )

        print(f"索引构建完成，共 {len(all_chunks)} 个文本块")
        return len(all_chunks)

    def build_from_papers(self) -> int:
        """
        从论文目录构建索引

        Returns:
            添加到索引的文档数量
        """
        print("开始解析论文...")
        sections = parse_all_papers()

        if not sections:
            print("未找到任何论文")
            return 0

        print(f"共解析 {len(sections)} 个章节，开始构建索引...")
        return self.build_index(sections)

    def get_collection_stats(self) -> dict:
        """获取索引统计信息"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "persist_dir": self.persist_dir
        }


if __name__ == "__main__":
    # 测试
    indexer = Indexer()
    count = indexer.build_from_papers()
    print(f"索引构建完成：{count} 个文本块")

    stats = indexer.get_collection_stats()
    print(f"统计信息：{stats}")
