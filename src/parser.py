"""PDF 解析模块"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .config import DATA_DIR


@dataclass
class PaperSection:
    """论文章节数据"""
    title: str
    content: str
    page_num: int
    metadata: Dict


def parse_pdf(file_path: str) -> List[PaperSection]:
    """
    解析 PDF 文件，提取文本内容

    Args:
        file_path: PDF 文件路径

    Returns:
        论文章节列表
    """
    sections = []
    doc = fitz.open(file_path)
    paper_title = Path(file_path).stem

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # 跳过空页
        if not text.strip():
            continue

        # 提取文本块
        blocks = page.get_text("blocks")
        current_section = []
        current_title = ""

        for block in blocks:
            x0, y0, x1, y1, content, block_no, block_type = block[:7]
            if block_type == 0:  # 文本块
                content = content.strip()
                if content:
                    # 简单判断是否为标题（字体较大或居中）
                    if len(content) < 100 and not content.endswith(('。', '.', ';', ';')):
                        # 可能是标题
                        if current_section:
                            sections.append(PaperSection(
                                title=current_title or f"第{page_num + 1}页",
                                content="\n".join(current_section),
                                page_num=page_num,
                                metadata={
                                    "source": paper_title,
                                    "file_path": file_path
                                }
                            ))
                            current_section = []
                        current_title = content
                    else:
                        current_section.append(content)

        # 保存最后一页的内容
        if current_section:
            sections.append(PaperSection(
                title=current_title or f"第{page_num + 1}页",
                content="\n".join(current_section),
                page_num=page_num,
                metadata={
                    "source": paper_title,
                    "file_path": file_path
                }
            ))

    doc.close()
    return sections


def parse_all_papers() -> List[PaperSection]:
    """
    解析 data/papers 目录下的所有 PDF 论文

    Returns:
        所有章节列表
    """
    all_sections = []
    papers_dir = Path(DATA_DIR)

    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)
        return all_sections

    for pdf_file in papers_dir.glob("*.pdf"):
        print(f"正在解析：{pdf_file.name}")
        try:
            sections = parse_pdf(str(pdf_file))
            all_sections.extend(sections)
            print(f"  提取了 {len(sections)} 个章节")
        except Exception as e:
            print(f"  解析失败：{e}")

    return all_sections


if __name__ == "__main__":
    # 测试
    sections = parse_all_papers()
    print(f"\n共提取 {len(sections)} 个章节")
    for s in sections[:3]:
        print(f"- {s.title} (第{s.page_num + 1}页): {s.content[:50]}...")
