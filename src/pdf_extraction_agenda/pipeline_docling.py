from os import PathLike

from docling.document_converter import DocumentConverter

document_converter = DocumentConverter()


def run_docling_pipeline(path: str | PathLike) -> str:
    result = document_converter.convert(path)
    return result.document.export_to_markdown()
