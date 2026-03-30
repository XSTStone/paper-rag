"""论文检索 RAG 应用 - Streamlit 前端"""
import streamlit as st
import os
from pathlib import Path
import shutil
import uuid
import json

from src.config import DATA_DIR, CHROMA_PERSIST_DIR, validate_config
from src.indexer import Indexer
from src.query_engine import QueryEngine
from src.logger import setup_logger
from src.errors import handle_error
from src.watcher import PaperCache

# 初始化日志
logger = setup_logger("app")

# 页面配置
st.set_page_config(
    page_title="论文检索 RAG",
    page_icon="📚",
    layout="wide"
)

# 初始化 session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "auto_update" not in st.session_state:
    st.session_state.auto_update = False
if "rag_trace_data" not in st.session_state:
    st.session_state.rag_trace_data = None
if "show_visualizer" not in st.session_state:
    st.session_state.show_visualizer = False
if "trace_query" not in st.session_state:
    st.session_state.trace_query = ""


def init_indexer():
    """初始化索引器"""
    return Indexer()


def init_query_engine():
    """初始化查询引擎"""
    return QueryEngine()


def save_uploaded_file(uploaded_file) -> Path:
    """保存上传的文件"""
    papers_dir = Path(DATA_DIR)
    papers_dir.mkdir(parents=True, exist_ok=True)
    save_path = papers_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def build_index_callback(indexer):
    """构建索引"""
    with st.spinner("正在构建索引，请稍候..."):
        try:
            count = indexer.build_from_papers()
            st.session_state.index_built = True
            # 更新缓存
            cache = PaperCache()
            cache.update()
            st.success(f"索引构建完成！共 {count} 个文本块")
            logger.info(f"索引构建完成：{count} 个文本块")
        except Exception as e:
            error_msg = handle_error(e)
            st.error(f"构建索引失败：{error_msg}")
            logger.error(f"索引构建失败：{e}")


def clear_chat():
    """清除聊天记录"""
    st.session_state.messages = []
    st.session_state.conversation_id = str(uuid.uuid4())


def run_rag_trace(query: str) -> dict:
    """
    执行 RAG 流程追踪，收集中间数据

    Args:
        query: 用户查询

    Returns:
        包含 RAG 流程完整数据的字典
    """
    try:
        engine = init_query_engine()

        # 1. 执行检索
        search_results = engine.retriever.search(query, top_k=5)

        # 2. 构建 prompt
        context = search_results[:5]
        prompt = engine._build_prompt(query, context)

        # 3. 调用 LLM
        system_prompt = "你是一个论文检索助手，基于提供的论文内容回答用户问题。"
        answer = engine._call_llm(prompt, system_prompt=system_prompt)

        # 4. 组装追踪数据
        trace_data = {
            "query": query,
            "chunks": [
                {
                    "id": f"chunk_{i+1:03d}",
                    "content": r.content,
                    "source": r.source,
                    "page_num": r.page_num,
                    "score": round(r.score, 4)
                }
                for i, r in enumerate(search_results)
            ],
            "results": [
                {
                    "id": f"result_{i+1:03d}",
                    "content": r.content,
                    "source": r.source,
                    "page_num": r.page_num,
                    "score": round(r.score, 4)
                }
                for i, r in enumerate(search_results)
            ],
            "system_prompt": system_prompt,
            "prompt": prompt,
            "answer": answer,
            "sources": [
                {
                    "source": r.source,
                    "page_num": r.page_num,
                    "score": round(r.score, 4)
                }
                for r in search_results
            ],
            "stats": {
                "total_chunks": len(search_results),
                "top_k": 5,
                "max_score": round(search_results[0].score, 4) if search_results else 0
            }
        }

        return trace_data

    except Exception as e:
        logger.error(f"RAG 追踪失败：{e}")
        st.error(f"追踪失败：{handle_error(e)}")
        return None


# ============= 侧边栏 =============
with st.sidebar:
    st.title("📚 论文检索 RAG")

    # 配置检查
    config_errors = validate_config()
    if config_errors:
        st.warning("⚠️ 配置不完整")
        for err in config_errors:
            st.error(err)
        st.info("请在 .env 文件中配置必要的 API Key")

    st.divider()

    # 文件上传
    st.subheader("📁 文件管理")
    uploaded_files = st.file_uploader(
        "上传 PDF 论文",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传的论文将保存到 data/papers/ 目录"
    )

    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            save_path = save_uploaded_file(uploaded_file)
            saved_count += 1
        st.success(f"已保存 {saved_count} 个文件")

    # 显示已上传的论文
    papers_dir = Path(DATA_DIR)
    if papers_dir.exists():
        pdf_files = list(papers_dir.glob("*.pdf"))
        if pdf_files:
            st.write(f"**已上传的论文** ({len(pdf_files)} 篇):")
            for pdf in pdf_files:
                st.text(f"  • {pdf.name}")

    st.divider()

    # 索引管理
    st.subheader("🔧 索引管理")

    # 自动更新选项
    auto_update = st.checkbox(
        "自动更新索引",
        value=st.session_state.auto_update,
        key="auto_update_toggle",
        help="开启后，当检测到论文文件变动时自动更新索引"
    )
    st.session_state.auto_update = auto_update

    # 文件变动检测
    if auto_update:
        cache = PaperCache()
        changes = cache.detect_changes()
        has_changes = any([changes["added"], changes["removed"], changes["modified"]])

        if has_changes:
            if st.button("🔄 检测到变动，点击更新索引", type="primary", use_container_width=True):
                indexer = init_indexer()
                stats = indexer.update_incremental(changes)
                cache.update()
                st.session_state.index_built = True
                if stats["added"] + stats["modified"] > 0:
                    st.success(f"更新完成：新增 {stats['added']} 块，修改 {stats['modified']} 块")
                if stats["removed"] > 0:
                    st.info(f"删除 {stats['removed']} 个文件的索引")
                if stats["errors"]:
                    for err in stats["errors"]:
                        st.warning(err)
                st.rerun()
        else:
            st.success("✅ 文件已是最新")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 构建索引", use_container_width=True):
            indexer = init_indexer()
            build_index_callback(indexer)

    with col2:
        if st.button("🗑️ 清除索引", use_container_width=True):
            storage_dir = Path(CHROMA_PERSIST_DIR)
            if storage_dir.exists():
                shutil.rmtree(storage_dir)
                storage_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.index_built = False
                st.success("索引已清除")
                st.rerun()

    # 索引状态
    if st.session_state.index_built:
        st.success("✅ 索引已构建")
    else:
        st.info("⏳ 索引未构建")

    st.divider()

    # 聊天管理
    st.subheader("💬 聊天管理")
    if st.button("🔄 新建对话", use_container_width=True):
        clear_chat()
        st.rerun()

    st.divider()

    # 可视化追踪
    st.subheader("🔍 可视化追踪")
    st.markdown("查看 RAG 流程的内部工作原理")

    trace_query = st.text_input(
        "输入追踪查询",
        value=st.session_state.trace_query,
        placeholder="请输入问题...",
        key="trace_input"
    )

    if st.button("▶️ 开始追踪", use_container_width=True, type="primary"):
        if not st.session_state.index_built:
            st.warning("请先构建索引")
        elif not trace_query.strip():
            st.warning("请输入查询内容")
        else:
            with st.spinner("正在执行 RAG 流程追踪..."):
                trace_data = run_rag_trace(trace_query)
                if trace_data:
                    st.session_state.rag_trace_data = trace_data
                    st.session_state.show_visualizer = True
                    st.session_state.trace_query = trace_query
                    st.rerun()

    if st.session_state.show_visualizer and st.session_state.rag_trace_data:
        if st.button("👁️ 查看可视化", use_container_width=True):
            pass  # 下面会处理显示

    st.divider()

    # 信息来源
    st.subheader("📊 论文来源")
    try:
        engine = init_query_engine()
        sources = engine.get_sources()
        if sources:
            for source in sources:
                st.text(f"  • {source}")
        else:
            st.info("暂无论文")
    except Exception as e:
        logger.debug(f"获取论文来源失败：{e}")
        st.info("暂无论文")


# ============= 主界面 =============
st.title("📖 论文检索助手")
st.markdown("基于 RAG 的论文问答系统，支持自然语言查询和引用溯源")

# 检查是否显示可视化器
if st.session_state.show_visualizer and st.session_state.rag_trace_data:
    st.divider()
    st.subheader("🔍 RAG 流程可视化")

    # 读取可视化器 HTML
    visualizer_path = Path(__file__).parent / "rag-visualizer.html"
    if visualizer_path.exists():
        with open(visualizer_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 将追踪数据注入到 HTML 中
        trace_data_json = json.dumps(st.session_state.rag_trace_data, ensure_ascii=False)

        # 修改 HTML，注入初始数据
        html_content = html_content.replace(
            'window.addEventListener(\'DOMContentLoaded\', () => {',
            f'window.ragData = {trace_data_json}; console.log("注入 RAG 数据:", window.ragData); window.addEventListener(\'DOMContentLoaded\', () => {{'
        )

        # 显示可视化器
        st.components.v1.html(html_content, height=1000, scrolling=True)

        # 添加关闭按钮
        if st.button("关闭可视化"):
            st.session_state.show_visualizer = False
            st.session_state.rag_trace_data = None
            st.rerun()
    else:
        st.error(f"未找到可视化器文件：{visualizer_path}")

st.divider()

# 聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📑 查看引用来源"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source.source} (第{source.page_num + 1}页)**")
                    st.markdown(f"> {source.content[:200]}...")
                    st.markdown(f"*相似度：{source.score:.3f}*")
                    st.divider()

# 聊天输入
if prompt := st.chat_input("请输入关于论文的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检查索引
    if not st.session_state.index_built:
        with st.chat_message("assistant"):
            st.warning("⚠️ 索引尚未构建，请先上传论文并点击「构建索引」")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "请先上传 PDF 论文到左侧边栏，然后点击「构建索引」按钮。"
            })
    else:
        # 执行查询
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    engine = init_query_engine()
                    response = engine.query(
                        prompt,
                        conversation_id=st.session_state.conversation_id
                    )

                    # 显示回答
                    st.markdown(response.answer)

                    # 显示引用来源
                    if response.sources:
                        with st.expander("📑 查看引用来源"):
                            for i, source in enumerate(response.sources, 1):
                                st.markdown(f"**{i}. {source.source} (第{source.page_num + 1}页)**")
                                st.markdown(f"> {source.content[:200]}...")
                                st.markdown(f"*相似度：{source.score:.3f}*")
                                st.divider()

                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })

                except Exception as e:
                    error_msg = handle_error(e)
                    st.error(f"查询失败：{error_msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"查询失败：{error_msg}"
                    })
                    logger.error(f"查询失败：{e}")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>",
    unsafe_allow_html=True
)
st.markdown(
    "Powered by DeepSeek + LlamaIndex + ChromaDB",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
