"""论文文件夹监控模块 - 检测变动并触发索引更新"""
from pathlib import Path
from typing import Dict, Set, Optional, List, Callable
import hashlib
import json
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, DirDeletedEvent

from .config import DATA_DIR
from .logger import get_logger

logger = get_logger("watcher")

# 缓存文件路径
CACHE_FILE = Path(__file__).parent.parent / ".paper_cache.json"


def compute_file_hash(file_path: Path) -> str:
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


class PaperCache:
    """论文文件缓存管理"""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}  # filename -> md5 hash
        self._load()

    def _load(self):
        """从文件加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"已加载缓存：{len(self.cache)} 个文件")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"加载缓存失败：{e}")
                self.cache = {}

    def _save(self):
        """保存缓存到文件"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug("缓存已保存")
        except IOError as e:
            logger.error(f"保存缓存失败：{e}")

    def get_current_files(self) -> Dict[str, str]:
        """获取当前目录下的所有 PDF 文件及其哈希值"""
        papers_dir = Path(DATA_DIR)
        current_files = {}

        if papers_dir.exists():
            for pdf_file in papers_dir.glob("*.pdf"):
                file_hash = compute_file_hash(pdf_file)
                if file_hash:
                    current_files[pdf_file.name] = file_hash

        return current_files

    def detect_changes(self) -> dict:
        """
        检测文件变动

        Returns:
            dict: {
                "added": List[str],  # 新增的文件
                "removed": List[str],  # 删除的文件
                "modified": List[str],  # 修改的文件
                "unchanged": List[str]  # 未变的文件
            }
        """
        current_files = self.get_current_files()
        cached_files = set(self.cache.keys())
        current_file_set = set(current_files.keys())

        added = list(current_file_set - cached_files)
        removed = list(cached_files - current_file_set)

        modified = []
        unchanged = []

        for filename in current_file_set & cached_files:
            if current_files[filename] != self.cache[filename]:
                modified.append(filename)
            else:
                unchanged.append(filename)

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged
        }

    def update(self):
        """更新缓存"""
        self.cache = self.get_current_files()
        self._save()
        logger.info(f"缓存已更新：{len(self.cache)} 个文件")

    def has_any_files(self) -> bool:
        """检查是否有任何论文文件"""
        return len(self.get_current_files()) > 0


class PaperChangeHandler(FileSystemEventHandler):
    """论文文件变动处理器"""

    def __init__(self, cache: PaperCache, on_change_callback: Optional[Callable] = None):
        super().__init__()
        self.cache = cache
        self.on_change_callback = on_change_callback
        self._debounce_timer: Optional[threading.Timer] = None
        self._debounce_interval = 2.0  # 防抖间隔（秒）

    def _schedule_callback(self):
        """调度回调函数（带防抖）"""
        if self._debounce_timer:
            self._debounce_timer.cancel()

        self._debounce_timer = threading.Timer(self._debounce_interval, self._trigger_callback)
        self._debounce_timer.start()

    def _trigger_callback(self):
        """触发回调"""
        changes = self.cache.detect_changes()
        has_changes = any([changes["added"], changes["removed"], changes["modified"]])

        if has_changes:
            logger.info(f"检测到文件变动：新增={changes['added']}, 删除={changes['removed']}, 修改={changes['modified']}")
            if self.on_change_callback:
                self.on_change_callback(changes)
        else:
            logger.debug("文件变动检测：无变化")

    def _should_process(self, event) -> bool:
        """检查是否应该处理该事件"""
        # 只处理 PDF 文件
        if not event.src_path.endswith('.pdf'):
            return False
        return True

    def on_created(self, event):
        """文件创建事件"""
        if self._should_process(event):
            logger.info(f"检测到新文件：{event.src_path}")
            self._schedule_callback()

    def on_modified(self, event):
        """文件修改事件"""
        if self._should_process(event):
            logger.info(f"检测到文件修改：{event.src_path}")
            self._schedule_callback()

    def on_deleted(self, event):
        """文件删除事件"""
        if self._should_process(event):
            logger.info(f"检测到文件删除：{event.src_path}")
            self._schedule_callback()


class PaperWatcher:
    """论文文件夹监控器"""

    def __init__(self, watch_dir: Optional[str] = None, on_change_callback: Optional[Callable] = None):
        """
        初始化监控器

        Args:
            watch_dir: 监控目录，None 则使用配置中的 DATA_DIR
            on_change_callback: 文件变动回调函数，接收 changes dict 作为参数
        """
        self.watch_dir = Path(watch_dir) if watch_dir else Path(DATA_DIR)
        self.cache = PaperCache()
        self.handler = PaperChangeHandler(self.cache, on_change_callback)
        self.observer: Optional[Observer] = None
        self._running = False

    def start(self):
        """启动监控"""
        if self._running:
            logger.warning("监控已在运行")
            return

        # 确保监控目录存在
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # 初始化缓存
        self.cache.update()

        # 启动观察者
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        self._running = True

        logger.info(f"开始监控论文目录：{self.watch_dir}")

    def stop(self):
        """停止监控"""
        if self.observer and self._running:
            self.observer.stop()
            self.observer.join()
            self._running = False
            logger.info("已停止监控")

    def check_changes(self) -> dict:
        """
        手动检查文件变动

        Returns:
            dict: 变动信息（同 PaperCache.detect_changes）
        """
        return self.cache.detect_changes()

    def update_cache(self):
        """更新缓存"""
        self.cache.update()

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running


def watch_papers(callback: Optional[Callable] = None) -> PaperWatcher:
    """
    便捷函数：启动论文文件夹监控

    Args:
        callback: 文件变动回调函数

    Returns:
        PaperWatcher 实例
    """
    watcher = PaperWatcher(on_change_callback=callback)
    watcher.start()
    return watcher


if __name__ == "__main__":
    # 测试
    def on_change(changes):
        print(f"检测到变动：{changes}")

    watcher = watch_papers(on_change)
    print(f"开始监控 {DATA_DIR} 目录，按 Ctrl+C 停止...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("监控已停止")
