"""论文检索 RAG 应用 - Streamlit 前端"""
import streamlit as st
import os
from pathlib import Path
import shutil
import uuid

from src.config import DATA_DIR, CHROMA_PERSIST_DIR, validate_config
from src.indexer import Indexer
from src.query_engine import QueryEngine
from src.logger import setup_logger
from src.errors import handle_error

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
