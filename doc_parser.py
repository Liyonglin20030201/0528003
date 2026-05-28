"""多格式文档解析模块

支持 .txt、.pdf、.docx 文件的文本提取。
"""
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """从 PDF 文件提取文本"""
    import pdfplumber

    import io
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """从 Word (.docx) 文件提取文本"""
    from docx import Document

    import io
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """根据文件扩展名提取文本内容"""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        return file_bytes.decode("utf-8")
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif suffix == ".docx":
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


SUPPORTED_EXTENSIONS = ["txt", "pdf", "docx"]
