"""查询引擎模块"""
from typing import List, Optional, Dict
from dataclasses import dataclass
import httpx

from .config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL, TOP_K
from .retriever import Retriever, SearchResult
from .logger import get_logger
from .errors import LLMError, handle_error

logger = get_logger("query_engine")


@dataclass
class QueryResponse:
    """查询响应"""
    answer: str
    sources: List[SearchResult]
    conversation_id: Optional[str] = None


class QueryEngine:
    """查询引擎 - 整合检索和 LLM 生成"""

    def __init__(self, top_k: int = TOP_K):
        """
        初始化查询引擎

        Args:
            top_k: 检索返回的最大文档数
        """
        self.top_k = top_k
        self.retriever = Retriever(top_k=top_k)

        # 对话历史
        self._conversation_history: Dict[str, List[Dict]] = {}

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用 DeepSeek API 生成回复"""
        try:
            logger.info(f"调用 LLM: {len(prompt)} 字符")

            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            response = httpx.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.info(f"LLM 响应：{len(content)} 字符")
            return content

        except httpx.HTTPTimeoutError as e:
            logger.error(f"LLM 请求超时：{e}")
            raise LLMError(handle_error(LLMError("请求超时，请重试")))
        except httpx.HTTPError as e:
            logger.error(f"LLM 请求失败：{e}")
            raise LLMError(handle_error(LLMError("API 调用失败")))

    def _build_prompt(self, query: str, context: List[SearchResult]) -> str:
        """构建给 LLM 的提示词"""
        context_text = "\n\n".join([
            f"[来源：{r.source}, 第{r.page_num + 1}页]\n{r.content}"
            for r in context
        ])

        prompt = f"""基于以下论文内容回答问题。如果内容不足以回答问题，请说明。

相关论文内容：
{context_text}

问题：{query}

请用中文回答，并在回答中标注引用来源（使用 [来源：论文名，页码] 格式）。"""

        return prompt

    def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> QueryResponse:
        """
        执行查询

        Args:
            question: 用户问题
            conversation_id: 对话 ID（用于多轮对话）
            top_k: 检索结果数

        Returns:
            查询响应
        """
        try:
            k = top_k or self.top_k
            logger.info(f"查询：{question[:50]}...")

            # 向量检索
            search_results = self.retriever.search(question, top_k=k)

            if not search_results:
                logger.warning("未找到相关结果")
                return QueryResponse(
                    answer="未找到相关的论文内容。请先上传论文文件到 data/papers/ 目录并构建索引。",
                    sources=[]
                )

            # 构建提示词并调用 LLM
            prompt = self._build_prompt(question, search_results)

            # 如果有对话历史，添加上下文
            system_prompt = "你是一个论文检索助手，基于提供的论文内容回答用户问题。"

            if conversation_id and conversation_id in self._conversation_history:
                history = self._conversation_history[conversation_id]
                # 在提示词中添加对话历史
                history_text = "\n".join([
                    f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                    for m in history[-4:]  # 最近 4 轮对话
                ])
                prompt = f"对话历史：\n{history_text}\n\n{prompt}"

            answer = self._call_llm(prompt, system_prompt=system_prompt)

            # 更新对话历史
            if conversation_id:
                if conversation_id not in self._conversation_history:
                    self._conversation_history[conversation_id] = []
                self._conversation_history[conversation_id].append({
                    "role": "user",
                    "content": question
                })
                self._conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": answer
                })
                # 限制对话历史长度
                if len(self._conversation_history[conversation_id]) > 20:
                    self._conversation_history[conversation_id] = self._conversation_history[conversation_id][-20:]

            logger.info(f"查询完成，{len(search_results)} 条来源")
            return QueryResponse(
                answer=answer,
                sources=search_results
            )

        except LLMError as e:
            logger.error(f"LLM 错误：{e}")
            return QueryResponse(
                answer=f"生成回答失败：{e.message}",
                sources=[]
            )
        except Exception as e:
            logger.exception(f"查询失败：{e}")
            return QueryResponse(
                answer=f"查询失败：{handle_error(e)}",
                sources=[]
            )

    def clear_history(self, conversation_id: Optional[str] = None):
        """清除对话历史"""
        if conversation_id:
            self._conversation_history.pop(conversation_id, None)
        else:
            self._conversation_history.clear()

    def get_sources(self) -> List[str]:
        """获取所有已索引的论文来源"""
        return self.retriever.get_all_sources()


if __name__ == "__main__":
    # 测试
    engine = QueryEngine()
    response = engine.query("这篇论文的主要贡献是什么？")
    print(f"回答：{response.answer}")
    print(f"\n引用来源：{len(response.sources)} 条")
    for s in response.sources[:3]:
        print(f"- {s.source}: {s.content[:30]}...")
