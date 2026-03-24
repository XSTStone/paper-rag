"""索引构建模块"""
from pathlib import Path
from typing import List, Optional
import chromadb
import httpx
import hashlib

from .config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL
from .parser import parse_all_papers, parse_pdf, PaperSection
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

        # 文件哈希缓存（用于增量更新）
        self._file_hashes: Dict[str, str] = {}
        self._load_file_hashes()

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

    def build_from_papers(self, force: bool = False) -> int:
        """
        从论文目录构建索引

        Args:
            force: 是否强制重新构建所有索引

        Returns:
            添加到索引的文档数量
        """
        from .config import DATA_DIR
        papers_dir = Path(DATA_DIR)

        # 获取当前文件列表
        current_files = {}
        for pdf_file in papers_dir.glob("*.pdf"):
            file_hash = self._compute_file_hash(pdf_file)
            if file_hash:
                current_files[pdf_file.name] = file_hash

        # 如果没有文件，返回 0
        if not current_files:
            print("未找到任何论文")
            return 0

        # 强制重新构建
        if force:
            print("开始重新构建索引...")
            sections = parse_all_papers()
            if sections:
                return self.build_index(sections)
            return 0

        # 检测变动
        cached_files = set(self._file_hashes.keys())
        current_file_set = set(current_files.keys())

        added = list(current_file_set - cached_files)
        removed = list(cached_files - current_file_set)
        modified = [f for f in current_file_set & cached_files if current_files[f] != self._file_hashes[f]]
        unchanged = [f for f in current_file_set & cached_files if current_files[f] == self._file_hashes[f]]

        # 如果没有变动且已有索引，跳过
        if not added and not removed and not modified:
            count = self.collection.count()
            if count > 0:
                print(f"文件无变动，索引已是最新（{count} 个文本块）")
                return 0

        # 显示变动情况
        changes = {"added": added, "removed": removed, "modified": modified, "unchanged": unchanged}
        return self.update_incremental(changes)

    def get_collection_stats(self) -> dict:
        """获取索引统计信息"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "persist_dir": self.persist_dir
        }

    def _load_file_hashes(self):
        """加载文件哈希缓存"""
        hash_file = Path(self.persist_dir) / ".file_hashes.json"
        if hash_file.exists():
            try:
                import json
                with open(hash_file, 'r', encoding='utf-8') as f:
                    self._file_hashes = json.load(f)
                logger.debug(f"已加载文件哈希缓存：{len(self._file_hashes)} 个文件")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"加载文件哈希缓存失败：{e}")
                self._file_hashes = {}

    def _save_file_hashes(self):
        """保存文件哈希缓存"""
        hash_file = Path(self.persist_dir) / ".file_hashes.json"
        try:
            import json
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, 'w', encoding='utf-8') as f:
                json.dump(self._file_hashes, f, ensure_ascii=False, indent=2)
            logger.debug("文件哈希缓存已保存")
        except IOError as e:
            logger.error(f"保存文件哈希缓存失败：{e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件的 MD5 哈希值"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")
            return ""

    def _get_indexed_files(self) -> Set[str]:
        """获取已索引的文件名"""
        files = set()
        for doc_id in self.collection.get()["ids"]:
            # ID 格式：file_path_page_section_chunk
            parts = doc_id.rsplit('_', 3)
            if parts:
                files.add(parts[0])
        return files

    def _delete_file_chunks(self, filename: str):
        """删除指定文件的索引块"""
        ids_to_delete = [
            doc_id for doc_id in self.collection.get()["ids"]
            if doc_id.startswith(f"{filename}_")
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"已删除文件 {filename} 的 {len(ids_to_delete)} 个索引块")

    def _index_single_file(self, file_path: Path) -> int:
        """
        索引单个文件

        Args:
            file_path: 论文文件路径

        Returns:
            添加的文本块数量
        """
        try:
            sections = parse_pdf(str(file_path))
            if not sections:
                logger.warning(f"文件 {file_path} 未解析出任何内容")
                return 0

            all_chunks = []
            all_ids = []
            all_metadatas = []

            filename = file_path.name
            file_hash = self._compute_file_hash(file_path)

            for i, section in enumerate(sections):
                chunks = split_text(section.content)

                for j, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    chunk_id = f"{filename}_{section.page_num}_{i}_{j}"
                    metadata = {
                        "source": filename,
                        "file_path": str(file_path),
                        "page_num": section.page_num,
                        "section_title": section.title
                    }

                    all_chunks.append(chunk)
                    all_ids.append(chunk_id)
                    all_metadatas.append(metadata)

            if all_chunks:
                logger.info(f"正在生成 {len(all_chunks)} 个文本块的嵌入向量...")
                all_embeddings = self._embedding_fn(all_chunks)

                self.collection.add(
                    documents=all_chunks,
                    ids=all_ids,
                    metadatas=all_metadatas,
                    embeddings=all_embeddings
                )

                # 更新文件哈希缓存
                self._file_hashes[filename] = file_hash
                self._save_file_hashes()

                logger.info(f"文件 {filename} 索引完成：{len(all_chunks)} 个文本块")
                return len(all_chunks)

            return 0

        except Exception as e:
            logger.error(f"索引文件 {file_path} 失败：{e}")
            return 0

    def update_incremental(self, changes: dict) -> dict:
        """
        增量更新索引

        Args:
            changes: 变动信息，格式为：
                {
                    "added": List[str],  # 新增的文件
                    "removed": List[str],  # 删除的文件
                    "modified": List[str],  # 修改的文件
                    "unchanged": List[str]  # 未变的文件
                }

        Returns:
            dict: 更新结果统计
        """
        from .config import DATA_DIR
        papers_dir = Path(DATA_DIR)

        stats = {
            "added": 0,
            "removed": 0,
            "modified": 0,
            "errors": []
        }

        # 处理新增和修改的文件
        for filename in changes.get("added", []) + changes.get("modified", []):
            file_path = papers_dir / filename
            if file_path.exists():
                try:
                    # 先删除旧的索引（如果有）
                    self._delete_file_chunks(filename)

                    # 重新索引
                    count = self._index_single_file(file_path)
                    stats["added" if filename in changes.get("added", []) else "modified"] += count
                    print(f"✓ {filename}: {count} 个文本块")
                except Exception as e:
                    error_msg = f"索引 {filename} 失败：{e}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)

        # 处理删除的文件
        for filename in changes.get("removed", []):
            try:
                self._delete_file_chunks(filename)
                stats["removed"] += 1

                # 从哈希缓存中移除
                if filename in self._file_hashes:
                    del self._file_hashes[filename]
                    self._save_file_hashes()

                print(f"✗ {filename}: 已删除索引")
            except Exception as e:
                error_msg = f"删除 {filename} 索引失败：{e}"
                stats["errors"].append(error_msg)
                logger.error(error_msg)

        # 输出统计
        total = stats["added"] + stats["modified"] + stats["removed"]
        if total > 0:
            print(f"\n索引更新完成：新增={stats['added']}, 修改={stats['modified']}, 删除={stats['removed']}")
            logger.info(f"增量更新完成：{stats}")

        return stats


if __name__ == "__main__":
    # 测试
    indexer = Indexer()
    count = indexer.build_from_papers()
    print(f"索引构建完成：{count} 个文本块")

    stats = indexer.get_collection_stats()
    print(f"统计信息：{stats}")
