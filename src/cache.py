"""缓存模块 - 性能优化"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import CHROMA_PERSIST_DIR


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class Cache:
    """内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        初始化缓存

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒），None 表示永不过期
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存"""
        # 检查是否需要清理
        if len(self._cache) >= self.max_size:
            self._cleanup()

        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl else None

        self._cache[key] = CacheEntry(
            value=value,
            expires_at=expires_at
        )

    def delete(self, key: str) -> None:
        """删除缓存"""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def _cleanup(self) -> None:
        """清理过期缓存"""
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

        # 如果仍然超出限制，删除最旧的条目
        if len(self._cache) >= self.max_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )
            for i in range(len(sorted_items) - self.max_size + 1):
                del self._cache[sorted_items[i][0]]

    def size(self) -> int:
        """返回缓存条目数"""
        return len(self._cache)


# 全局缓存实例
embedding_cache = Cache(max_size=5000, default_ttl=3600)  # 嵌入缓存，1 小时过期
query_cache = Cache(max_size=1000, default_ttl=300)  # 查询缓存，5 分钟过期


def cached_embedding(func):
    """嵌入结果缓存装饰器"""
    def wrapper(texts: List[str], *args, **kwargs):
        # 为每个文本生成缓存键
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            cached = embedding_cache.get(key)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 调用原函数获取未缓存的结果
        if uncached_texts:
            uncached_results = func(uncached_texts, *args, **kwargs)
            for i, (idx, _) in enumerate(results):
                pass  # 已有缓存

            # 填充结果并缓存
            for i, text in enumerate(uncached_texts):
                embedding = uncached_results[i]
                key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
                embedding_cache.set(key, embedding)

        # 合并结果
        final_results = [None] * len(texts)
        for idx, emb in results:
            final_results[idx] = emb

        # 添加新获取的结果
        for i, idx in enumerate(uncached_indices):
            final_results[idx] = uncached_results[i]

        return final_results

    return wrapper


class IndexStats:
    """索引统计信息"""

    def __init__(self):
        self._stats_cache: Optional[Dict] = None
        self._last_update = 0
        self._cache_ttl = 60  # 60 秒缓存

    def get_stats(self, indexer) -> Dict:
        """获取索引统计（带缓存）"""
        now = time.time()

        if self._stats_cache and (now - self._last_update) < self._cache_ttl:
            return self._stats_cache

        # 重新计算统计
        stats = indexer.get_collection_stats()
        stats["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._stats_cache = stats
        self._last_update = now

        return stats


# 全局索引统计
index_stats = IndexStats()
