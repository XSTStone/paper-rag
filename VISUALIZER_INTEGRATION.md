# RAG 可视化器集成说明

## 集成完成

已将 RAG 可视化器集成到 paper-rag 项目中，现在可以通过可视化界面查看 RAG 流程的内部工作原理。

## 使用方法

### 1. 启动应用

```bash
cd paper-rag
streamlit run app.py
```

### 2. 构建索引

1. 在左侧边栏上传 PDF 论文
2. 点击「📦 构建索引」按钮
3. 等待索引构建完成

### 3. 使用可视化追踪

1. 在左侧边栏找到「🔍 可视化追踪」区域
2. 输入要追踪的查询问题（如："这篇论文的主要贡献是什么？"）
3. 点击「▶️ 开始追踪」按钮
4. 等待 RAG 流程执行完成
5. 点击「👁️ 查看可视化」按钮（或直接向下滚动）
6. 可视化器会显示完整的 RAG 流程

### 4. 查看各阶段详情

可视化器显示 4 个阶段，点击左侧导航或顶部流程条可切换：

| 阶段 | 内容 |
|------|------|
| 📄 文档分块 | 显示检索到的文档块及其元数据 |
| ⚡ 检索结果 | 显示向量检索结果和相似度分数 |
| 📝 Prompt 构建 | 显示组装给 LLM 的完整 Prompt |
| ✨ 最终答案 | 显示 LLM 生成的答案和引用来源 |

## 数据流说明

```
用户输入查询
    ↓
run_rag_trace() 执行追踪
    ↓
1. 检索文档 (Retriever.search)
2. 构建 Prompt (QueryEngine._build_prompt)
3. 调用 LLM (QueryEngine._call_llm)
4. 组装追踪数据
    ↓
注入到 rag-visualizer.html
    ↓
通过 st.components.v1.html 渲染
```

## 追踪数据结构

```json
{
  "query": "用户查询",
  "chunks": [
    {
      "id": "chunk_001",
      "content": "文档内容",
      "source": "论文文件名",
      "page_num": 0,
      "score": 0.95
    }
  ],
  "results": [...],  // 同 chunks
  "system_prompt": "系统指令",
  "prompt": "完整的 Prompt",
  "answer": "LLM 生成的答案",
  "sources": [...],  // 引用来源
  "stats": {
    "total_chunks": 5,
    "top_k": 5,
    "max_score": 0.95
  }
}
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit 主程序（已添加追踪功能） |
| `rag-visualizer.html` | 可视化器（简化版，4 个面板） |
| `src/query_engine.py` | 查询引擎（提供追踪数据） |
| `src/retriever.py` | 检索模块 |

## 修改内容

### app.py 新增内容

1. **Session State 扩展**
   - `rag_trace_data`: 存储追踪数据
   - `show_visualizer`: 控制可视化器显示
   - `trace_query`: 追踪查询输入

2. **新增函数 `run_rag_trace()`**
   - 执行完整的 RAG 流程
   - 收集中间数据（分块、检索结果、Prompt、答案）
   - 返回 JSON 格式的追踪数据

3. **侧边栏新增「可视化追踪」区域**
   - 查询输入框
   - 开始追踪按钮

4. **主界面新增可视化器显示区域**
   - 使用 `st.components.v1.html()` 渲染
   - 数据直接注入到 HTML 中

### rag-visualizer.html

- 简化版可视化器（4 个面板而非原版 5 个）
- 移除了复杂的动画效果
- 数据源改为从注入的 JSON 获取
- 保留了深色主题 UI 设计

## 故障排除

### 问题：点击追踪后没有反应

**解决**：
1. 检查是否已构建索引
2. 查看终端是否有错误日志
3. 确认 API Key 配置正确

### 问题：可视化器显示「加载失败」

**解决**：
1. 检查 `rag-visualizer.html` 文件是否存在
2. 确认文件编码为 UTF-8
3. 刷新页面重试

### 问题：中文显示乱码

**解决**：
1. 确保文件使用 UTF-8 编码保存
2. 检查浏览器编码设置

## 扩展建议

如需添加更多可视化功能，可以：

1. **添加 Embedding 可视化** - 在面板中展示向量分布
2. **添加 HNSW 索引动画** - 展示向量搜索过程
3. **添加多轮对话追踪** - 记录对话历史
4. **导出追踪报告** - 将追踪数据导出为 JSON

## 技术栈

- **前端**: 原生 HTML/CSS/JavaScript
- **后端**: Streamlit + Python
- **数据传递**: JSON 注入 + postMessage
