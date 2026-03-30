"""配置管理模块"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# ============= LLM 配置 =============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ============= 嵌入模型配置 =============
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ============= ChromaDB 配置 =============
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "storage"))

# ============= 数据目录配置 =============
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data" / "papers"))

# ============= 文档处理配置 =============
# 分块大小
CHUNK_SIZE = 512
# 分块重叠大小
CHUNK_OVERLAP = 50
# 检索返回的最大文档数
TOP_K = 10
# 最小文档长度（过滤过短的文档块）
MIN_CHUNK_LENGTH = 20

# ============= 语义切分配置 =============
# 语义相似度阈值（低于此值认为是语义边界）
SEMANTIC_SIMILARITY_THRESHOLD = 0.65
# 每块最少句子数
MIN_SENTENCES_PER_CHUNK = 2
# 每块最多句子数
MAX_SENTENCES_PER_CHUNK = 8
# 是否启用语义切分（设为 True 启用）
USE_SEMANTIC_CHUNKING = True

# ============= 验证配置 =============
def validate_config():
    """验证必要的配置是否存在"""
    errors = []

    if not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY 未配置")

    return errors
