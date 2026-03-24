"""错误处理模块 - 自定义异常和错误提示"""
from typing import Optional, Dict, Any


class PaperRAGError(Exception):
    """基础异常类"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ConfigError(PaperRAGError):
    """配置错误"""
    pass


class ParserError(PaperRAGError):
    """PDF 解析错误"""
    pass


class IndexError(PaperRAGError):
    """索引构建错误"""
    pass


class RetrievalError(PaperRAGError):
    """检索错误"""
    pass


class LLMError(PaperRAGError):
    """LLM 调用错误"""
    pass


class FileError(PaperRAGError):
    """文件操作错误"""
    pass


# 错误消息映射
ERROR_MESSAGES = {
    ConfigError: {
        "default": "配置错误",
        "DEEPSEEK_API_KEY": "DeepSeek API Key 未配置，请在 .env 文件中设置 DEEPSEEK_API_KEY",
        "EMBEDDING_API_KEY": "嵌入模型 API Key 未配置，请在 .env 文件中设置 EMBEDDING_API_KEY",
    },
    ParserError: {
        "default": "PDF 解析失败",
        "file_not_found": "文件不存在",
        "invalid_pdf": "无效的 PDF 文件",
        "parse_failed": "解析失败，文件可能已损坏",
    },
    IndexError: {
        "default": "索引构建失败",
        "no_papers": "未找到任何论文，请先上传 PDF 文件到 data/papers/ 目录",
        "build_failed": "索引构建失败",
        "empty_collection": "索引为空",
    },
    RetrievalError: {
        "default": "检索失败",
        "no_results": "未找到相关结果",
        "query_empty": "查询内容为空",
    },
    LLMError: {
        "default": "LLM 调用失败",
        "api_error": "API 调用失败，请检查网络和 API Key",
        "rate_limit": "请求频率过高，请稍后重试",
        "timeout": "请求超时，请重试",
    },
    FileError: {
        "default": "文件操作失败",
        "permission_denied": "权限不足，无法访问文件",
        "disk_full": "磁盘空间不足",
    },
}


def get_user_friendly_message(error: Exception) -> str:
    """
    获取用户友好的错误消息

    Args:
        error: 异常对象

    Returns:
        用户友好的错误消息
    """
    error_type = type(error)

    if error_type in ERROR_MESSAGES:
        messages = ERROR_MESSAGES[error_type]
        # 尝试匹配具体的错误键
        for key, msg in messages.items():
            if key == "default":
                continue
            if key.lower() in str(error).lower():
                return msg
        return messages.get("default", str(error))

    # 未知错误类型，返回通用消息
    return f"发生错误：{str(error)}"


def handle_error(error: Exception, logger=None) -> str:
    """
    统一错误处理

    Args:
        error: 异常对象
        logger: 日志记录器

    Returns:
        用户友好的错误消息
    """
    if logger:
        logger.exception(f"处理错误：{error}")

    user_message = get_user_friendly_message(error)

    # 如果是自定义异常，使用其 message
    if isinstance(error, PaperRAGError):
        user_message = error.message

    return user_message
