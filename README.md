# PDF extraction pipelines and benchmarks agenda

> [!CAUTION]
> Part of text in this repo written by ChatGPT. Also, I haven't yet run all pipelines because of lack of compute power.

This repository provides an overview of notable **pipelines** and **benchmarks** related to PDF/OCR document
processing. Each entry includes a brief description, and useful data.

## Table of contents

Did you know that GitHub supports table of
contents [by default](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/) 🤔

## Comparison

> [!IMPORTANT]
> Open [README.md in separate page](https://github.com/dantetemplar/pdf-extraction-agenda/blob/main/README.md), not in repository preview! It will look better.

<benches>

|                  Pipeline                   | [OmniDocBench](#omnidocbench) Overall ↓ | [OmniDocBench](#omnidocbench) New ↑ | [olmOCR](#olmoocr-eval) Overall ↑ |  [dp-bench](#dp-bench) NID ↑  |
|---------------------------------------------|-----------------------------------------|-------------------------------------|-----------------------------------|-------------------------------|
|              [MinerU](#MinerU)              |         0.133 <sup>[2]</sup> ⚠️         |     **90.67 <sup>[1]</sup>** ⚠️     |               61.5                |             91.18             |
|           [MonkeyOCR](#MonkeyOCR)           |         0.138 <sup>[3]</sup> ⚠️         |        88.85 <sup>[2]</sup>         |      75.8 <sup>[3]</sup> ⚠️       |                               |
|      [PP-StructureV3](#PP-StructureV3)      |                  0.145                  |                86.73                |                                   |                               |
|              [Marker](#Marker)              |                  0.296                  |                71.3                 |               70.1                |                               |
|            [Pix2Text](#Pix2Text)            |                  0.32                   |                                     |                                   |                               |
|              [olmOCR](#olmOCR)              |                  0.326                  |                81.79                |      78.5 <sup>[2]</sup> ⚠️       |                               |
|        [Unstructured](#Unstructured)        |                  0.586                  |                                     |                                   |                               |
|             [DocLing](#DocLing)             |                  0.589                  |                                     |                                   |                               |
|          [Open-Parse](#Open-Parse)          |                  0.646                  |                                     |                                   |                               |
|          [MarkItDown](#MarkItDown)          |                                         |                                     |                                   |                               |
|               [Zerox](#Zerox)               |                                         |                                     |                                   |                               |
|            [Markdrop](#Markdrop)            |                                         |                                     |                                   |                               |
|        [Vision Parse](#Vision-Parse)        |                                         |                                     |                                   |                               |
|            _↓ Specialized VLMs_             |                                         |                                     |                                   |                               |
|            [dots.ocr](#dotsocr)             |       **0.125 <sup>[1]</sup>** ⚠️       |        88.41 <sup>[3]</sup>         |    **79.1 <sup>[1]</sup>** ⚠️     |                               |
|             [Dolphin](#Dolphin)             |                  0.205                  |                74.67                |                                   |                               |
|             [OCRFlux](#OCRFlux)             |                  0.238                  |                74.82                |                                   |                               |
|        [Nanonets-OCR](#Nanonets-OCR)        |                  0.283                  |                85.59                |               64.5                |                               |
|             [GOT-OCR](#GOT-OCR)             |                  0.287                  |                                     |               48.3                |                               |
|              [Nougat](#Nougat)              |                  0.452                  |                                     |                                   |                               |
|         [SmolDocling](#SmolDocling)         |                  0.493                  |                                     |                                   |                               |
|          _↓ Proprietary pipelines_          |                                         |                                     |                                   |                               |
|             [Mathpix](#Mathpix)             |                  0.191                  |                                     |                                   |                               |
|         [Mistral OCR](#Mistral-OCR)         |                  0.268                  |                78.83                |               72.0                |                               |
|  [Google Document AI](#Google-Document-AI)  |                                         |                                     |                                   |             90.86             |
|           [Azure OCR](#Azure-OCR)           |                                         |                                     |                                   |             87.69             |
|     [Amazon Textract](#Amazon-Textract)     |                                         |                                     |                                   |     96.71 <sup>[2]</sup>      |
|          [LlamaParse](#LlamaParse)          |                                         |                                     |                                   |     92.82 <sup>[3]</sup>      |
|          [Upstage AI](#Upstage-AI)          |                                         |                                     |                                   |  **97.02 <sup>[1]</sup>** ⚠️  |
|               [doc2x](#doc2x)               |                                         |                                     |                                   |                               |
|              _↓ General VLMs_               |                                         |                                     |                                   |                               |
|               Gemini-2.5 Pro                |                  0.148                  |                88.03                |                                   |                               |
|              Gemini-2.0 Flash               |                  0.191                  |                                     |               63.8                |                               |
|               Qwen2.5-VL-72B                |                  0.214                  |                87.02                |               65.5                |                               |
|                InternVL3-78B                |                  0.218                  |                80.33                |                                   |                               |
|                    GPT4o                    |                  0.233                  |                75.02                |               69.9                |                               |
|                Qwen2-VL-72B                 |                  0.252                  |                                     |               31.5                |                               |
</benches>

- **Bold** indicates the best result for a given metric, and <sup>[2]</sup> indicates 2nd place in that benchmark.
- " " means the pipeline was not evaluated in that benchmark.
- ⚠️ means the pipeline authors are the ones who suggested the results.
- `Overall ↑` in column name means higher value is better, when `Overall ↓` - lower value is better.

## Pipelines

### [MinerU](https://github.com/opendatalab/MinerU)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/7)
[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/MinerU?label=GitHub&logo=github)](https://github.com/opendatalab/MinerU)
![License](https://img.shields.io/badge/License-AGPL--3.0-orange)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/opendatalab/MinerU)

**Primary Language:** Python

**License:** AGPL-3.0

**Description:** MinerU is an open-source tool designed to convert PDFs into machine-readable formats, such as Markdown
and JSON, facilitating seamless data extraction and further processing. Developed during the pre-training phase of
InternLM, MinerU addresses symbol conversion challenges in scientific literature, making it invaluable for research and
development in large language models. Key features include:

- **Content Cleaning**: Removes headers, footers, footnotes, and page numbers to ensure semantic coherence.
- **Structure Preservation**: Maintains the original document structure, including titles, paragraphs, and lists.
- **Multimodal Extraction**: Accurately extracts images, image descriptions, tables, and table captions.
- **Formula Recognition**: Converts recognized formulas into LaTeX format.
- **Table Conversion**: Transforms tables into LaTeX or HTML formats.
- **OCR Capabilities**: Detects scanned or corrupted PDFs and enables OCR functionality, supporting text recognition in
  84 languages.
- **Cross-Platform Compatibility**: Operates on Windows, Linux, and Mac platforms, supporting both CPU and GPU
  environments.

### [Marker](https://github.com/VikParuchuri/marker)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/8)
[![GitHub last commit](https://img.shields.io/github/last-commit/VikParuchuri/marker?label=GitHub&logo=github)](https://github.com/VikParuchuri/marker)
![License](https://img.shields.io/badge/License-GPL--3.0-yellow)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://www.datalab.to/)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://www.datalab.to/)

**Primary Language:** Python

**License:** GPL-3.0

**Description:** Marker “converts PDFs and images to markdown, JSON, and HTML quickly and accurately.” It is designed to
handle a wide range of document types in all languages and produce structured outputs.

**Benchmark Results:** https://github.com/VikParuchuri/marker?tab=readme-ov-file#performance

**API Details:**

- **API URL:** https://www.datalab.to/
- **Pricing:** https://www.datalab.to/plans
- **Average Price:** $3 per 1000 pages, at least $25 per month

**Additional Notes:**
**Demo available after registration on https://www.datalab.to/**

### [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/29)
[![GitHub last commit](https://img.shields.io/github/last-commit/rednote-hilab/dots.ocr?label=GitHub&logo=github)](https://github.com/rednote-hilab/dots.ocr)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://dotsocr.xiaohongshu.com)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://replicate.com/sljeff/dots.ocr)

**License:** MIT

**Description:** dots.ocr is a powerful, multilingual document parser that unifies layout detection and content recognition within a single vision-language model while maintaining good reading order. Despite its compact 1.7B-parameter LLM foundation, it achieves state-of-the-art (SOTA) performance on text, tables, and reading order tasks. The model supports over 100 languages and can handle various document types including PDFs, images, tables, formulas, and maintains proper reading order. It offers a significantly more streamlined architecture than conventional methods that rely on complex, multi-model pipelines, allowing users to switch between tasks simply by altering the input prompt.​

**Benchmark Results:** https://huggingface.co/rednote-hilab/dots.ocr#benchmark-results

**API Details:**
- **API URL:** https://replicate.com/sljeff/dots.ocr

**Additional Notes:**
- Built on 1.7B parameters, providing faster inference speeds than larger models
- Supports both layout detection and content recognition in a unified architecture
- Multiple deployment options including Docker, vLLM, Hugging Face Transformers, and cloud APIs
- Strong multilingual capabilities with particular strength in low-resource languages
- Can output structured data in JSON, Markdown, and HTML formats
- Includes specialized prompts for different use cases: layout detection, OCR-only, and grounding OCR
- 4-bit quantized version available for consumer-grade GPUs

### [OCRFlux](https://github.com/chatdoc-com/OCRFlux)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/31)
[![GitHub last commit](https://img.shields.io/github/last-commit/chatdoc-com/OCRFlux?label=GitHub&logo=github)](https://github.com/chatdoc-com/OCRFlux)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://ocrflux.pdfparser.io/)

**License:** Apache-2.0

**Description:** OCRFlux is a multimodal large language model based toolkit designed to convert PDFs and images into clean, readable, plain Markdown text. It excels in complex layout handling, including multi-column layouts, figures, insets, complicated tables, and equations. The system also provides automated removal of headers and footers, alongside native support for cross-page table and paragraph merging, a pioneering feature among open-source OCR tools. Built on a 3 billion parameter vision-language model, it can run efficiently on GPUs such as the GTX 3090. OCRFlux provides batch inference support for whole documents and detailed parsing quality with benchmarks demonstrating significant improvements over several leading OCR models.​

**Additional Notes:**
- Recommended GPU: 24GB or more VRAM for best performance, but supports tensor parallelism to divide workload across multiple smaller GPUs
- Includes Docker container support for easy deployment
- Supports various command-line options for customizing inference, GPU memory utilization, page merging behavior, and data type selection
- Outputs results as JSONL files convertible into Markdown documents
- Developed and maintained by ChatDOC team
- Has 2.3k stars on GitHub

### [Dolphin](https://huggingface.co/ByteDance/Dolphin)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/30)
[![GitHub last commit](https://img.shields.io/github/last-commit/bytedance/Dolphin?label=GitHub&logo=github)](https://github.com/bytedance/Dolphin)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/ByteDance/Dolphin)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://replicate.com/bytedance/dolphin)

**License:** MIT

**Description:** Dolphin is a novel multimodal document image parsing model (0.3B parameters) that follows an analyze-then-parse paradigm. It addresses complex document understanding challenges through a two-stage approach: Stage 1 performs comprehensive page-level layout analysis by generating element sequences in natural reading order, while Stage 2 enables efficient parallel parsing of document elements using heterogeneous anchors and task-specific prompts. The model handles intertwined elements such as text paragraphs, figures, formulas, and tables while maintaining superior efficiency through its lightweight architecture and parallel parsing mechanism. Built on a vision-encoder-decoder architecture using Swin Transformer for visual encoding and MBart for text decoding, Dolphin supports both page-level and element-level parsing tasks.

**API Details:**
- **API URL:** https://replicate.com/bytedance/dolphin
- **Average Price:** Approximately $17 per 1000 pages (based on Replicate pricing of $0.017 per run)​

**Additional Notes:**
- Compact 0.3B parameter model optimized for efficiency
- Supports both original config-based framework and Hugging Face integration
- Multi-page PDF document parsing capability added in June 2025
- TensorRT-LLM and vLLM support for accelerated inference
- Two parsing granularities: page-level (entire document) and element-level (individual components)
- Element-decoupled parsing strategy allows for easier data collection and training
- Natural language prompt-based interface for controlling parsing tasks
- Supports various document elements including text paragraphs, tables, formulas, and figures
- Open-source with active development and community support (7.4k GitHub stars)
- Published research paper accepted at ACL 2025 conference

### [Nanonets-OCR](https://huggingface.co/nanonets/Nanonets-OCR-s)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/32)
[![GitHub last commit](https://img.shields.io/github/last-commit/NanoNets/docstrange?label=GitHub&logo=github)](https://github.com/NanoNets/docstrange)
![License](https://img.shields.io/badge/License-Other (please specify below)-red)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/Souvik3333/Nanonets-ocr-s)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://nanonets.com/ocr-api)

**License:** Other (please specify below)

**Description:** Nanonets-OCR-s is a powerful open-source OCR model that converts images or documents into richly structured markdown with intelligent content recognition and semantic tags. Key features include automatic LaTeX equation recognition, intelligent image description, signature detection, watermark extraction, smart checkbox handling, and complex table extraction. It is designed for downstream processing by large language models for tasks like document understanding and parsing.

**API Details:**
- **API URL:** https://nanonets.com/ocr-api
- **Pricing:** https://nanonets.com/pricing

**Additional Notes:**
- The open-source model supports inference via Hugging Face transformers and vLLM server.
- It can be fine-tuned and adapted for custom datasets.
- Useful for research, experimentation, and building customized OCR pipelines without commercial restrictions.

### [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/33)
[![GitHub last commit](https://img.shields.io/github/last-commit/PaddlePaddle/PaddleOCR?label=GitHub&logo=github)](https://github.com/PaddlePaddle/PaddleOCR)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/PaddlePaddle/PP-StructureV3_Online_Demo)

**License:** Apache-2.0

**Description:** PP-StructureV3 is a multi-model pipeline for document image parsing that converts document images or PDFs into structured JSON and Markdown files. It integrates several key modules: preprocessing for image quality improvements, an OCR engine (PP-OCRv5), layout detection via PP-DocLayout-plus, document item recognition (tables, formulas, charts, seals), and post-processing to reconstruct element relationships and reading order. The pipeline is designed for high accuracy in complex layouts including multi-column texts, magazines, handwritten documents, and vertically typeset languages.

It supports comprehensive recognition with specialized models for tables (PP-TableMagic), formulas (PP-FormulaNet_plus), charts (PP-Chart2Table), and seals (PP-OCRv4_seal). It achieves state-of-the-art results on benchmarks like OmniDocBench, especially for Chinese and English documents, competing well with expert and general vision-language models.

**Additional Notes:**
-   PP-StructureV3 uses PP-OCRv5 as the OCR backbone, which includes improvements in network architecture and training, supporting vertical text, handwriting, and rare Chinese characters.
-   Preprocessing includes document orientation classification and text unwarping.
-   Layout analysis uses PP-DocLayout-plus and a region detection model to handle multiple articles per page.
-   Table recognition with PP-TableMagic outputs HTML formatted structures.
-   Formula recognition with PP-FormulaNet_plus outputs LaTeX.
-   Chart parsing converts charts into markdown tables.
-   Seal recognition handles curved text and round/oval seals.
-   Post-processing enhances reading order reconstruction especially for complex document layouts (e.g., multi-column magazines, vertical typesetting).
-  Performance is tested on NVIDIA V100/A100 GPUs with detailed resource usage statistics available.
-  The system can process PDFs and images and can save results in JSON and Markdown formats.

### [MarkItDown](https://github.com/microsoft/markitdown)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/9)
[![GitHub last commit](https://img.shields.io/github/last-commit/microsoft/markitdown?label=GitHub&logo=github)](https://github.com/microsoft/markitdown)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://mitdown.ca/)

**Primary Language:** Python

**License:** MIT

**Description:** MarkItDown is a Python-based utility developed by Microsoft for converting various file formats into
Markdown. It supports a wide range of file types, including:

- **Office Documents**: Word (.docx), PowerPoint (.pptx), Excel (.xlsx)
- **Media Files**: Images (with EXIF metadata and OCR capabilities), Audio (with speech transcription)
- **Web and Data Formats**: HTML, CSV, JSON, XML
- **Archives**: ZIP files (with recursive content parsing)
- **URLs**: YouTube links

This versatility makes MarkItDown a valuable tool for tasks such as indexing, text analysis, and preparing content for
Large Language Model (LLM) training. The utility offers both command-line and Python API interfaces, providing
flexibility for various use cases. Additionally, MarkItDown features a plugin-based architecture, allowing for easy
integration of third-party extensions to enhance its functionality.

### [olmOCR](https://olmocr.allenai.org/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/10)
[![GitHub last commit](https://img.shields.io/github/last-commit/allenai/olmocr?label=GitHub&logo=github)](https://github.com/allenai/olmocr)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://olmocr.allenai.org/)

**Primary Language:** Python

**License:** Apache-2.0

**Description:** olmOCR is an open-source toolkit developed by the Allen Institute for AI, designed to convert PDFs and
document images into clean, plain text suitable for large language model (LLM) training and other applications. Key
features include:

- **High Accuracy**: Preserves reading order and supports complex elements such as tables, equations, and handwriting.
- **Document Anchoring**: Combines text and visual information to enhance extraction accuracy.
- **Structured Content Representation**: Utilizes Markdown to represent structured content, including sections, lists,
  equations, and tables.
- **Optimized Pipeline**: Compatible with SGLang and vLLM inference engines, enabling efficient scaling from single to
  multiple GPUs.

### [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/27)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yuliang-Liu/MonkeyOCR?label=GitHub&logo=github)](https://github.com/Yuliang-Liu/MonkeyOCR)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](http://vlrlabmonkey.xyz:7685)

**License:** Apache-2.0

**Description:** MonkeyOCR is an open‑source, **layout‑aware document parsing system** developed by Yuliang‑Liu and collaborators that implements a novel **Structure‑Recognition‑Relation (SRR)** 
triplet paradigm. It decomposes document analysis into three phases—block structure detection (“Where is it?”), 
content recognition (“What is it?”), and reading‑order relation modeling (“How is it organized?”)—delivering both high 
accuracy and inference speed by avoiding heavy end‑to‑end models or brittle modular pipelines. 
Trained on the extensive **MonkeyDoc dataset** (nearly 3.9 million instances across English and Chinese, 
covering 10+ document types), MonkeyOCR achieves state‑of‑the‑art performance, including significant gains in table 
(+8.6%) and formula (+15.0%) recognition, and outperforms much larger models like Qwen2.5‑VL (72B) and Gemini 2.5 Pro. 
Remarkably, the 3B‑parameter variant runs efficiently—approximately 0.84 pages per second on multi‑page input using a single 
NVIDIA 3090 GPU—making it practical for real‑world document workloads.


**Benchmark Results:** https://github.com/Yuliang-Liu/MonkeyOCR?tab=readme-ov-file#benchmark-results

### [Mistral OCR](https://mistral.ai/news/mistral-ocr)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/20)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://docs.mistral.ai/capabilities/document/)

**License:** Proprietary

**API Details:**

- **API URL:** https://docs.mistral.ai/capabilities/document/
- **Pricing:** https://mistral.ai/products/la-plateforme#pricing
- **Average Price:** 1$ per 1000 pages

### [Google Document AI](https://cloud.google.com/document-ai)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/23)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://console.cloud.google.com/ai/document-ai)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://cloud.google.com/document-ai/docs/reference/rest)

**License:** Proprietary

**Description:** Google Document AI is a cloud-based document processing service that uses machine learning to
automatically extract structured data from documents. It supports various document types, including invoices, receipts,
forms, and identity documents. Key features include:

- **Optical Character Recognition (OCR)**: Converts scanned images and PDFs into editable text.
- **Data Extraction**: Identifies and extracts key-value pairs, tables, and other structured data.
- **Document Understanding** Classifies and understands the content of documents.
- **Customization**: Allows users to train custom models for specific document types.

**API Details:**

- **API URL:** https://cloud.google.com/document-ai/docs/reference/rest
- **Pricing:** https://cloud.google.com/document-ai/pricing
- **Average Price:** $1.50 per 1000 pages

### [Azure OCR](https://azure.microsoft.com/en-us/products/ai-services/ai-vision)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/24)
[![GitHub last commit](https://img.shields.io/github/last-commit/Azure/azure-sdk-for-python?label=GitHub&logo=github)](https://github.com/Azure/azure-sdk-for-python)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/ocr)

**License:** Proprietary

**Description:** Azure AI Vision OCR is a cloud-based service that employs advanced machine-learning algorithms to
extract printed and handwritten text from images and documents. It supports a wide array of languages and can process
various content types, including posters, street signs, product labels, and business documents. The service is designed
to detect text lines, words, and paragraphs, providing structured output suitable for integration into applications
requiring text extraction capabilities.

**API Details:**

- **API URL:** https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/ocr
- **Pricing:** https://azure.microsoft.com/en-us/pricing/details/cognitive-services/computer-vision/
- **Average Price:** $1 per 1,000 transactions

### [Amazon Textract](https://aws.amazon.com/textract/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/25)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://docs.aws.amazon.com/textract/latest/dg/API_Reference.html)

**License:** Proprietary

**Description:** Amazon Textract is a machine learning service that automatically extracts text, handwriting, and data
from scanned documents. It goes beyond simple optical character recognition (OCR) by also identifying the contents of
fields in forms, information stored in tables, and the presence of selection elements such as checkboxes. This enables
the conversion of unstructured content into structured data, facilitating integration into various applications and
workflows.

**API Details:**

- **API URL:** https://docs.aws.amazon.com/textract/latest/dg/API_Reference.html
- **Pricing:** https://aws.amazon.com/textract/pricing/
- **Average Price:** $1.50 per 1000 pages

### [LlamaParse](https://www.llamaindex.ai/llamaparse)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/6)
[![GitHub last commit](https://img.shields.io/github/last-commit/run-llama/llama_parse?label=GitHub&logo=github)](https://github.com/run-llama/llama_parse)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://api.cloud.llamaindex.ai/api/parsing/upload)

**Primary Language:** Python

**License:** Proprietary

**Description:** LlamaParse is a GenAI-native document parsing platform developed by LlamaIndex. It transforms complex
documents—including PDFs, PowerPoint presentations, Word documents, and spreadsheets—into structured, LLM-ready formats.
LlamaParse excels in accurately extracting and formatting tables, images, and other non-standard layouts, ensuring
high-quality data for downstream applications such as Retrieval-Augmented Generation (RAG) and data processing. The
platform supports over 10 file types and offers features like natural language parsing instructions, JSON output, and
multilingual support.

**API Details:**

- **API URL:** https://api.cloud.llamaindex.ai/api/parsing/upload
- **Pricing:** https://docs.cloud.llamaindex.ai/llamaparse/usage_data
- **Average Price:** **Free Plan**: 1,000 pages per day; **Paid Plan**: 7,000 pages per week, with additional pages at $
  3 per 1,000 pages

### [Mathpix](https://mathpix.com/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/5)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://docs.mathpix.com/)

**Primary Language:** Not publicly available

**License:** Proprietary

**Description:** Mathpix offers advanced Optical Character Recognition (OCR) technology tailored for STEM content. Their
services include the Convert API, which accurately digitizes images and PDFs containing complex elements such as
mathematical equations, chemical diagrams, tables, and handwritten notes. The platform supports multiple output formats,
including LaTeX, MathML, HTML, and Markdown, facilitating seamless integration into various applications and workflows.
Additionally, Mathpix provides the Snipping Tool, a desktop application that allows users to capture and convert content
from their screens into editable formats with a single keyboard shortcut.

**API Details:**

- **API URL:** https://docs.mathpix.com/
- **Pricing:** https://mathpix.com/pricing
- **Average Price:** $5 per 1000 pages

### [Upstage AI](https://upstage.ai/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/22)
[![GitHub last commit](https://img.shields.io/github/last-commit/UpstageAI/cookbook?label=GitHub&logo=github)](https://github.com/UpstageAI/cookbook)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://console.upstage.ai/docs/getting-started)

**License:** Proprietary

**Description:** The Upstage AI is a comprehensive suite of artificial intelligence solutions designed to enhance
business operations across various industries. It encompasses advanced large language models (LLMs) and document
processing engines to streamline workflows and improve efficiency.

**Benchmark Results:** https://www.upstage.ai/blog/en/icdar-win-interview

**API Details:**

- **API URL:** https://console.upstage.ai/docs/getting-started
- **Pricing:** https://upstage.ai/pricing
- **Average Price:** $10 per 1000 pages

### [Nougat](https://facebookresearch.github.io/nougat/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/4)
[![GitHub last commit](https://img.shields.io/github/last-commit/facebookresearch/nougat?label=GitHub&logo=github)](https://github.com/facebookresearch/nougat)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Primary Language:** Python

**License:** MIT

**Description:** Nougat (Neural Optical Understanding for Academic Documents) is an open-source Visual Transformer model
developed by Meta AI Research. It is designed to perform Optical Character Recognition (OCR) on scientific documents,
converting PDFs into a machine-readable markup language. Nougat simplifies the extraction of complex elements such as
mathematical expressions and tables, enhancing the accessibility of scientific knowledge. The model processes raw pixel
data from document images and outputs structured markdown text, bridging the gap between human-readable content and
machine-readable formats.

### [GOT-OCR](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/3)
[![GitHub last commit](https://img.shields.io/github/last-commit/Ucas-HaoranWei/GOT-OCR2.0?label=GitHub&logo=github)](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/ucaslcl/GOT_online)

**Primary Language:** Python

**License:** Apache-2.0

**Description:** GOT-OCR (General OCR Theory) is an open-source, unified end-to-end model designed to advance OCR to
version 2.0. It supports a wide range of tasks, including plain document OCR, scene text OCR, formatted document OCR,
and OCR for tables, charts, mathematical formulas, geometric shapes, molecular formulas, and sheet music. The model is
highly versatile, supporting various input types and producing structured outputs, making it well-suited for complex OCR
tasks.

**Benchmark Results:** https://github.com/Ucas-HaoranWei/GOT-OCR2.0#benchmarks

### [DocLing](https://github.com/DS4SD/docling)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/2)
[![GitHub last commit](https://img.shields.io/github/last-commit/DS4SD/docling?label=GitHub&logo=github)](https://github.com/DS4SD/docling)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Primary Language:** Python

**License:** MIT

**Description:** DocLing is an open-source document processing pipeline developed by IBM Research. It simplifies the
parsing of diverse document formats—including PDF, DOCX, PPTX, HTML, and images—and provides seamless integrations with
the generative AI ecosystem. Key features include advanced PDF understanding, optical character recognition (OCR)
support, and plug-and-play integrations with frameworks like LangChain and LlamaIndex.

### [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/21)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo)

**License:** Apache-2.0

**Description:** SmolDocling is a multimodal Image-Text-to-Text model designed for efficient document conversion,
developed by Docling team. It retains Docling's most popular features while ensuring full compatibility with Docling
through seamless support for DoclingDocuments.

### [Zerox](https://getomni.ai/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/12)
[![GitHub last commit](https://img.shields.io/github/last-commit/getomni-ai/zerox?label=GitHub&logo=github)](https://github.com/getomni-ai/zerox)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://getomni.ai/ocr-demo)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://getomni.ai/)

**Primary Language:** TypeScript

**License:** MIT

**Description:** Zerox is an OCR and document extraction tool that leverages vision models to convert PDFs and images
into structured Markdown format. It excels in handling complex layouts, including tables and charts, making it ideal for
AI ingestion and further text analysis.

**Benchmark Results:** https://getomni.ai/ocr-benchmark

**API Details:**

- **API URL:** https://getomni.ai/
- **Pricing:** https://getomni.ai/pricing
- **Average Price:** Extract structured data: 'Startup' plan at $225 per month with 5000 pages included, after that $2
  per 1000 pages

### [Unstructured](https://unstructured.io/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/13)
[![GitHub last commit](https://img.shields.io/github/last-commit/Unstructured-IO/unstructured?label=GitHub&logo=github)](https://github.com/Unstructured-IO/unstructured)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://demo.unstructured.io/)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://docs.unstructured.io/platform-api/overview)

**Primary Language:** Python

**License:** Apache-2.0

**Description:** Unstructured is an open-source library that provides components for ingesting and pre-processing
unstructured data, including images and text documents such as PDFs, HTML, and Word documents. It transforms complex
data into structured formats suitable for large language models and AI applications. The platform offers
enterprise-grade connectors to seamlessly integrate various data sources, making it easier to extract and transform data
for analysis and processing.

**API Details:**

- **API URL:** https://docs.unstructured.io/platform-api/overview
- **Pricing:** https://unstructured.io/developers
- **Average Price:** **Basic Strategy
  **: $2 per 1,000 pages, suitable for simple, text-only documents. **Advanced Strategy**: $20 per 1,000 pages, ideal
  for PDFs, images, and complex file types. **Platinum/VLM Strategy**: $30 per 1,000 pages, designed for challenging
  documents, including scanned and handwritten content with VLM API integration.

### [Pix2Text](https://p2t.breezedeus.com/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/14)
[![GitHub last commit](https://img.shields.io/github/last-commit/breezedeus/Pix2Text?label=GitHub&logo=github)](https://github.com/breezedeus/Pix2Text)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://p2t.breezedeus.com/)

**Primary Language:** Python

**License:** MIT

**Description:** Pix2Text (P2T) is an open-source Python3 tool designed to recognize layouts, tables, mathematical
formulas (LaTeX), and text in images, converting them into Markdown format. It serves as a free alternative to Mathpix,
supporting over 80 languages, including English, Simplified Chinese, Traditional Chinese, and Vietnamese. P2T can also
process entire PDF files, extracting content into structured Markdown, facilitating seamless conversion of visual
content into text-based representations.

### [Open-Parse](https://filimoa.github.io/open-parse/)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/15)
[![GitHub last commit](https://img.shields.io/github/last-commit/Filimoa/open-parse?label=GitHub&logo=github)](https://github.com/Filimoa/open-parse)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Primary Language:** Python

**License:** MIT

**Description:** Open Parse is a flexible, open-source library designed to enhance document chunking for
Retrieval-Augmented Generation (RAG) systems. It visually analyzes document layouts to effectively group related
content, surpassing traditional text-splitting methods. Key features include:

- **Visually-Driven Analysis**: Understands complex layouts for superior chunking.
- **Markdown Support**: Extracts headings, bold, and italic text into Markdown format.
- **High-Precision Table Extraction**: Converts tables into clean Markdown with high accuracy.
- **Extensibility**: Allows implementation of custom post-processing steps.
- **Intuitive Design**: Offers robust editor support for seamless integration.

### [Extractous](https://github.com/yobix-ai/extractous)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/16)
[![GitHub last commit](https://img.shields.io/github/last-commit/yobix-ai/extractous?label=GitHub&logo=github)](https://github.com/yobix-ai/extractous)
![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen)

**Primary Language:** Rust

**License:** Apache-2.0

**Description:** Extractous is a high-performance, open-source library designed for efficient extraction of content and
metadata from various document types, including PDF, Word, HTML, and more. Developed in Rust, it offers bindings for
multiple programming languages, starting with Python. Extractous aims to provide a comprehensive solution for
unstructured data extraction, enabling local and efficient processing without relying on external services or APIs. Key
features include:

- **High Performance**: Leveraging Rust's capabilities, Extractous achieves faster processing speeds and lower memory
  utilization compared to traditional extraction libraries.
- **Multi-Language Support**: While the core is written in Rust, bindings are available for Python, with plans to
  support additional languages like JavaScript/TypeScript.
- **Extensive Format Support**: Through integration with Apache Tika, Extractous supports a wide range of file formats,
  ensuring versatility in data extraction tasks.
- **OCR Integration**: Incorporates Tesseract OCR to extract text from images and scanned documents, enhancing its
  ability to handle diverse content types.

**Benchmark Results:** https://github.com/yobix-ai/extractous-benchmarks

### [Markdrop](https://github.com/shoryasethia/markdrop)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/18)
[![GitHub last commit](https://img.shields.io/github/last-commit/shoryasethia/markdrop?label=GitHub&logo=github)](https://github.com/shoryasethia/markdrop)
![License](https://img.shields.io/badge/License-GPL--3.0-yellow)

**Primary Language:** Python

**License:** GPL-3.0

**Description:** A Python package for converting PDFs to markdown while extracting images and tables, generate
descriptive text descriptions for extracted tables/images using several LLM clients. And many more functionalities.
Markdrop is available on PyPI.

### [Vision Parse](https://github.com/iamarunbrahma/vision-parse)

[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/19)
[![GitHub last commit](https://img.shields.io/github/last-commit/iamarunbrahma/vision-parse?label=GitHub&logo=github)](https://github.com/iamarunbrahma/vision-parse)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Primary Language:** Python

**License:** MIT

**Description:** Parse PDFs into markdown using Vision LLMs


### [doc2x](https://noedgeai.com/)
[✏️](https://github.com/dantetemplar/pdf-extraction-agenda/issues/28)
[![GitHub last commit](https://img.shields.io/github/last-commit/NoEdgeAI/pdfdeal?label=GitHub&logo=github)](https://github.com/NoEdgeAI/pdfdeal)
![License](https://img.shields.io/badge/License-Proprietary-red)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://doc2x.noedgeai.com/)
[![API](https://img.shields.io/badge/API-Available-blue?logo=swagger&logoColor=85EA2D)](https://noedgeai.github.io/pdfdeal-docs/)

**License:** Proprietary

**Description:** NoEdgeAI is an open‑source technology initiative focused on enhancing document processing in Retrieval-Augmented Generation (RAG) workflows. Their flagship library, pdfdeal, is a Python wrapper for the Doc2X API that facilitates high‑fidelity PDF-to-text conversion. It extends Doc2X’s capabilities by offering local text preprocessing, Markdown and LaTeX extraction, file splitting, image uploading, and enhancements for better recall when integrating PDFs into knowledge‑base tools like Graphrag, Dify, or FastGPT

**API Details:**
- **API URL:** https://noedgeai.github.io/pdfdeal-docs/


## Benchmarks

### [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/OmniDocBench?label=GitHub&logo=github)](https://github.com/opendatalab/OmniDocBench)
![GitHub License](https://img.shields.io/github/license/opendatalab/OmniDocBench)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

OmniDocBench is *“a benchmark for evaluating diverse document parsing in real-world scenarios”* by MinerU devs. It
establishes a
comprehensive evaluation standard for document content extraction methods.

**Notable features:** OmniDocBench covers a wide variety of document types and layouts, comprising **981 PDF pages
across 9 document types, 4 layout styles, and 3 languages**. It provides **rich annotations**: over 20k block-level
elements (paragraphs, headings, tables, etc.) and 80k+ span-level elements (lines, formulas, etc.), including reading
order and various attribute tags for pages, text, and tables. The dataset undergoes strict quality control (combining
manual annotation, intelligent assistance, and expert review for high accuracy). OmniDocBench also comes with *
*evaluation code** for fair, end-to-end comparisons of document parsing methods. It supports multiple evaluation tasks (
overall extraction, layout detection, table recognition, formula recognition, OCR text recognition) and standard
metrics (Normalized Edit Distance, BLEU, METEOR, TEDS, COCO mAP/mAR, etc.) to benchmark performance across different
aspects of document parsing.

**End-to-End Evaluation**

End-to-end evaluation assesses the model's accuracy in parsing PDF page content. The evaluation uses the model's
Markdown output of the entire PDF page parsing results as the prediction.

<table style="width: 92%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2">Method Type</th>
      <th rowspan="2">Methods</th>
      <th colspan="2">Overall<sup>Edit</sup>↓</th>
      <th colspan="2">Text<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>CDM</sup>↑</th>
      <th colspan="2">Table<sup>TEDS</sup>↑</th>
      <th colspan="2">Table<sup>Edit</sup>↓</th>
      <th colspan="2">Read Order<sup>Edit</sup>↓</th>
    </tr>
    <tr>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Pipeline Tools</td>
      <td>MinerU-0.9.3</td>
      <td>0.15</td>
      <td>0.357</td>
      <td>0.061</td>
      <td>0.215</td>
      <td>0.278</td>
      <td>0.577</td>
      <td>57.3</td>
      <td>42.9</td>
      <td>78.6</td>
      <td>62.1</td>
      <td>0.18</td>
      <td>0.344</td>
      <td>0.079</td>
      <td>0.292</td>
    </tr>
    <tr>
      <td>Marker-1.2.3</td>
      <td>0.336</td>
      <td>0.556</td>
      <td>0.08</td>
      <td>0.315</td>
      <td>0.53</td>
      <td>0.883</td>
      <td>17.6</td>
      <td>11.7</td>
      <td>67.6</td>
      <td>49.2</td>
      <td>0.619</td>
      <td>0.685</td>
      <td>0.114</td>
      <td>0.34</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td>0.191</td>
      <td>0.365</td>
      <td>0.105</td>
      <td>0.384</td>
      <td>0.306</td>
      <td>0.454</td>
      <td>62.7</td>
      <td>62.1</td>
      <td>77.0</td>
      <td>67.1</td>
      <td>0.243</td>
      <td>0.32</td>
      <td>0.108</td>
      <td>0.304</td>
    </tr>
    <tr>
      <td>Docling-2.14.0</td>
      <td>0.589</td>
      <td>0.909</td>
      <td>0.416</td>
      <td>0.987</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>61.3</td>
      <td>25.0</td>
      <td>0.627</td>
      <td>0.810</td>
      <td>0.313</td>
      <td>0.837</td>
    </tr>
    <tr>
      <td>Pix2Text-1.1.2.3</td>
      <td>0.32</td>
      <td>0.528</td>
      <td>0.138</td>
      <td>0.356</td>
      <td><strong>0.276</strong></td>
      <td>0.611</td>
      <td>78.4</td>
      <td>39.6</td>
      <td>73.6</td>
      <td>66.2</td>
      <td>0.584</td>
      <td>0.645</td>
      <td>0.281</td>
      <td>0.499</td>
    </tr>
    <tr>
      <td>Unstructured-0.17.2</td>
      <td>0.586</td>
      <td>0.716</td>
      <td>0.198</td>
      <td>0.481</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0.064</td>
      <td>1</td>
      <td>0.998</td>
      <td>0.145</td>
      <td>0.387</td>
    </tr>
    <tr>
      <td>OpenParse-0.7.0</td>
      <td>0.646</td>
      <td>0.814</td>
      <td>0.681</td>
      <td>0.974</td>
      <td>0.996</td>
      <td>1</td>
      <td>0.106</td>
      <td>0</td>
      <td>64.8</td>
      <td>27.5</td>
      <td>0.284</td>
      <td>0.639</td>
      <td>0.595</td>
      <td>0.641</td>
    </tr>
    <tr>
      <td rowspan="5">Expert VLMs</td>
      <td>GOT-OCR</td>
      <td>0.287</td>
      <td>0.411</td>
      <td>0.189</td>
      <td>0.315</td>
      <td>0.360</td>
      <td>0.528</td>
      <td>74.3</td>
      <td>45.3</td>
      <td>53.2</td>
      <td>47.2</td>
      <td>0.459</td>
      <td>0.52</td>
      <td>0.141</td>
      <td>0.28</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.452</td>
      <td>0.973</td>
      <td>0.365</td>
      <td>0.998</td>
      <td>0.488</td>
      <td>0.941</td>
      <td>15.1</td>
      <td>16.8</td>
      <td>39.9</td>
      <td>0.0</td>
      <td>0.572</td>
      <td>1.000</td>
      <td>0.382</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>Mistral OCR</td>
      <td>0.268</td>
      <td>0.439</td>
      <td>0.072</td>
      <td>0.325</td>
      <td>0.318</td>
      <td>0.495</td>
      <td>64.6</td>
      <td>45.9</td>
      <td>75.8</td>
      <td>63.6</td>
      <td>0.6</td>
      <td>0.65</td>
      <td>0.083</td>
      <td>0.284</td>
    </tr>
    <tr>
      <td>OLMOCR-sglang</td>
      <td>0.326</td>
      <td>0.469</td>
      <td>0.097</td>
      <td>0.293</td>
      <td>0.455</td>
      <td>0.655</td>
      <td>74.3</td>
      <td>43.2</td>
      <td>68.1</td>
      <td>61.3</td>
      <td>0.608</td>
      <td>0.652</td>
      <td>0.145</td>
      <td>0.277</td>
    </tr>
    <tr>
      <td>SmolDocling-256M_transformer</td>
      <td>0.493</td>
      <td>0.816</td>
      <td>0.262</td>
      <td>0.838</td>
      <td>0.753</td>
      <td>0.997</td>
      <td>32.1</td>
      <td>0.551</td>
      <td>44.9</td>
      <td>16.5</td>
      <td>0.729</td>
      <td>0.907</td>
      <td>0.227</td>
      <td>0.522</td>
    </tr>
    <tr>
      <td rowspan="8">General VLMs</td>
    <tr>
      <td>Gemini2.0-flash</td>
      <td>0.191</td>
      <td>0.264</td>
      <td>0.091</td>
      <td>0.139</td>
      <td>0.389</td>
      <td>0.584</td>
      <td>77.6</td>
      <td>43.6</td>
      <td>79.7</td>
      <td>78.9</td>
      <td>0.193</td>
      <td>0.206</td>
      <td>0.092</td>
      <td>0.128</td>
    </tr>
    <tr>
      <td>Gemini2.5-Pro</td>
      <td><strong>0.148</strong></td>
      <td><strong>0.212</strong></td>
      <td><strong>0.055</strong></td>
      <td><strong>0.168</strong></td>
      <td>0.356</td>
      <td>0.439</td>
      <td>80.0</td>
      <td><strong>69.4</strong></td>
      <td><strong>85.8</strong></td>
      <td><strong>86.4</strong></td>
      <td><strong>0.13</strong></td>
      <td><strong>0.119</strong></td>
      <td><strong>0.049</strong></td>
      <td><strong>0.121</strong></td>
    </tr>
    <tr>
      <td>GPT4o</td>
      <td>0.233</td>
      <td>0.399</td>
      <td>0.144</td>
      <td>0.409</td>
      <td>0.425</td>
      <td>0.606</td>
      <td>72.8</td>
      <td>42.8</td>
      <td>72.0</td>
      <td>62.9</td>
      <td>0.234</td>
      <td>0.329</td>
      <td>0.128</td>
      <td>0.251</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>0.252</td>
      <td>0.327</td>
      <td>0.096</td>
      <td>0.218</td>
      <td>0.404</td>
      <td>0.487</td>
      <td><strong>82.2</strong></td>
      <td>61.2</td>
      <td>76.8</td>
      <td>76.4</td>
      <td>0.387</td>
      <td>0.408</td>
      <td>0.119</td>
      <td>0.193</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>0.214</td>
      <td>0.261</td>
      <td>0.092</td>
      <td>0.18</td>
      <td>0.315</td>
      <td><strong>0.434</strong></td>
      <td>68.8</td>
      <td>62.5</td>
      <td>82.9</td>
      <td>83.9</td>
      <td>0.341</td>
      <td>0.262</td>
      <td>0.106</td>
      <td>0.168</td>
    </tr>
    <tr>
      <td>InternVL2-76B</td>
      <td>0.44</td>
      <td>0.443</td>
      <td>0.353</td>
      <td>0.290</td>
      <td>0.543</td>
      <td>0.701</td>
      <td>67.4</td>
      <td>44.1</td>
      <td>63.0</td>
      <td>60.2</td>
      <td>0.547</td>
      <td>0.555</td>
      <td>0.317</td>
      <td>0.228</td>
    </tr>
  </tbody>
</table>


<table style="width: 92%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2">Method Type</th>
      <th rowspan="2">Methods</th>
      <th colspan="2">Overall<sup>Edit</sup>↓</th>
      <th colspan="2">Text<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>CDM</sup>↑</th>
      <th colspan="2">Table<sup>TEDS</sup>↑</th>
      <th colspan="2">Table<sup>Edit</sup>↓</th>
      <th colspan="2">Read Order<sup>Edit</sup>↓</th>
    </tr>
    <tr>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Pipeline Tools</td>
      <td>MinerU-0.9.3</td>
      <td>0.15</td>
      <td>0.357</td>
      <td>0.061</td>
      <td>0.215</td>
      <td>0.278</td>
      <td>0.577</td>
      <td>57.3</td>
      <td>42.9</td>
      <td>78.6</td>
      <td>62.1</td>
      <td>0.18</td>
      <td>0.344</td>
      <td>0.079</td>
      <td>0.292</td>
    </tr>
    <tr>
      <td>Marker-1.2.3</td>
      <td>0.336</td>
      <td>0.556</td>
      <td>0.08</td>
      <td>0.315</td>
      <td>0.53</td>
      <td>0.883</td>
      <td>17.6</td>
      <td>11.7</td>
      <td>67.6</td>
      <td>49.2</td>
      <td>0.619</td>
      <td>0.685</td>
      <td>0.114</td>
      <td>0.34</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td>0.191</td>
      <td>0.365</td>
      <td>0.105</td>
      <td>0.384</td>
      <td>0.306</td>
      <td>0.454</td>
      <td>62.7</td>
      <td>62.1</td>
      <td>77.0</td>
      <td>67.1</td>
      <td>0.243</td>
      <td>0.32</td>
      <td>0.108</td>
      <td>0.304</td>
    </tr>
    <tr>
      <td>Docling-2.14.0</td>
      <td>0.589</td>
      <td>0.909</td>
      <td>0.416</td>
      <td>0.987</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>61.3</td>
      <td>25.0</td>
      <td>0.627</td>
      <td>0.810</td>
      <td>0.313</td>
      <td>0.837</td>
    </tr>
    <tr>
      <td>Pix2Text-1.1.2.3</td>
      <td>0.32</td>
      <td>0.528</td>
      <td>0.138</td>
      <td>0.356</td>
      <td><strong>0.276</strong></td>
      <td>0.611</td>
      <td>78.4</td>
      <td>39.6</td>
      <td>73.6</td>
      <td>66.2</td>
      <td>0.584</td>
      <td>0.645</td>
      <td>0.281</td>
      <td>0.499</td>
    </tr>
    <tr>
      <td>Unstructured-0.17.2</td>
      <td>0.586</td>
      <td>0.716</td>
      <td>0.198</td>
      <td>0.481</td>
      <td>0.999</td>
      <td>1</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0.064</td>
      <td>1</td>
      <td>0.998</td>
      <td>0.145</td>
      <td>0.387</td>
    </tr>
    <tr>
      <td>OpenParse-0.7.0</td>
      <td>0.646</td>
      <td>0.814</td>
      <td>0.681</td>
      <td>0.974</td>
      <td>0.996</td>
      <td>1</td>
      <td>0.106</td>
      <td>0</td>
      <td>64.8</td>
      <td>27.5</td>
      <td>0.284</td>
      <td>0.639</td>
      <td>0.595</td>
      <td>0.641</td>
    </tr>
    <tr>
      <td rowspan="5">Expert VLMs</td>
      <td>GOT-OCR</td>
      <td>0.287</td>
      <td>0.411</td>
      <td>0.189</td>
      <td>0.315</td>
      <td>0.360</td>
      <td>0.528</td>
      <td>74.3</td>
      <td>45.3</td>
      <td>53.2</td>
      <td>47.2</td>
      <td>0.459</td>
      <td>0.52</td>
      <td>0.141</td>
      <td>0.28</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.452</td>
      <td>0.973</td>
      <td>0.365</td>
      <td>0.998</td>
      <td>0.488</td>
      <td>0.941</td>
      <td>15.1</td>
      <td>16.8</td>
      <td>39.9</td>
      <td>0.0</td>
      <td>0.572</td>
      <td>1.000</td>
      <td>0.382</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>Mistral OCR</td>
      <td>0.268</td>
      <td>0.439</td>
      <td>0.072</td>
      <td>0.325</td>
      <td>0.318</td>
      <td>0.495</td>
      <td>64.6</td>
      <td>45.9</td>
      <td>75.8</td>
      <td>63.6</td>
      <td>0.6</td>
      <td>0.65</td>
      <td>0.083</td>
      <td>0.284</td>
    </tr>
    <tr>
      <td>OLMOCR-sglang</td>
      <td>0.326</td>
      <td>0.469</td>
      <td>0.097</td>
      <td>0.293</td>
      <td>0.455</td>
      <td>0.655</td>
      <td>74.3</td>
      <td>43.2</td>
      <td>68.1</td>
      <td>61.3</td>
      <td>0.608</td>
      <td>0.652</td>
      <td>0.145</td>
      <td>0.277</td>
    </tr>
    <tr>
      <td>SmolDocling-256M_transformer</td>
      <td>0.493</td>
      <td>0.816</td>
      <td>0.262</td>
      <td>0.838</td>
      <td>0.753</td>
      <td>0.997</td>
      <td>32.1</td>
      <td>0.551</td>
      <td>44.9</td>
      <td>16.5</td>
      <td>0.729</td>
      <td>0.907</td>
      <td>0.227</td>
      <td>0.522</td>
    </tr>
    <tr>
      <td rowspan="8">General VLMs</td>
    <tr>
      <td>Gemini2.0-flash</td>
      <td>0.191</td>
      <td>0.264</td>
      <td>0.091</td>
      <td>0.139</td>
      <td>0.389</td>
      <td>0.584</td>
      <td>77.6</td>
      <td>43.6</td>
      <td>79.7</td>
      <td>78.9</td>
      <td>0.193</td>
      <td>0.206</td>
      <td>0.092</td>
      <td>0.128</td>
    </tr>
    <tr>
      <td>Gemini2.5-Pro</td>
      <td><strong>0.148</strong></td>
      <td><strong>0.212</strong></td>
      <td><strong>0.055</strong></td>
      <td><strong>0.168</strong></td>
      <td>0.356</td>
      <td>0.439</td>
      <td>80.0</td>
      <td><strong>69.4</strong></td>
      <td><strong>85.8</strong></td>
      <td><strong>86.4</strong></td>
      <td><strong>0.13</strong></td>
      <td><strong>0.119</strong></td>
      <td><strong>0.049</strong></td>
      <td><strong>0.121</strong></td>
    </tr>
    <tr>
      <td>GPT4o</td>
      <td>0.233</td>
      <td>0.399</td>
      <td>0.144</td>
      <td>0.409</td>
      <td>0.425</td>
      <td>0.606</td>
      <td>72.8</td>
      <td>42.8</td>
      <td>72.0</td>
      <td>62.9</td>
      <td>0.234</td>
      <td>0.329</td>
      <td>0.128</td>
      <td>0.251</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>0.252</td>
      <td>0.327</td>
      <td>0.096</td>
      <td>0.218</td>
      <td>0.404</td>
      <td>0.487</td>
      <td><strong>82.2</strong></td>
      <td>61.2</td>
      <td>76.8</td>
      <td>76.4</td>
      <td>0.387</td>
      <td>0.408</td>
      <td>0.119</td>
      <td>0.193</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>0.214</td>
      <td>0.261</td>
      <td>0.092</td>
      <td>0.18</td>
      <td>0.315</td>
      <td><strong>0.434</strong></td>
      <td>68.8</td>
      <td>62.5</td>
      <td>82.9</td>
      <td>83.9</td>
      <td>0.341</td>
      <td>0.262</td>
      <td>0.106</td>
      <td>0.168</td>
    </tr>
    <tr>
      <td>InternVL2-76B</td>
      <td>0.44</td>
      <td>0.443</td>
      <td>0.353</td>
      <td>0.290</td>
      <td>0.543</td>
      <td>0.701</td>
      <td>67.4</td>
      <td>44.1</td>
      <td>63.0</td>
      <td>60.2</td>
      <td>0.547</td>
      <td>0.555</td>
      <td>0.317</td>
      <td>0.228</td>
    </tr>
  </tbody>
</table>

<p style="text-align: center; margin-top: -4pt;">
  Comprehensive evaluation of document parsing algorithms on OmniDocBench: performance metrics for text, formula, table, and reading order extraction, with overall scores derived from ground truth comparisons.
</p>

### [olmoOCR eval](https://github.com/allenai/olmocr/tree/main/olmocr/bench)

[![GitHub last commit](https://img.shields.io/github/last-commit/allenai/olmocr?label=GitHub&logo=github)](https://github.com/allenai/olmocr)
![GitHub License](https://img.shields.io/github/license/allenai/olmocr)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/allenai/olmOCR-bench)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

olmOCR-Bench works by testing various "facts" about document pages at the PDF-level. Our intention is that each "fact" is very simple, 
unambiguous, and machine-checkable, similar to a unit test. For example, once your document has been OCRed, we may check that a
 particular sentence appears exactly somewhere on the page.

Dataset Link: https://huggingface.co/datasets/allenai/olmOCR-bench


<table>
  <thead>
    <tr>
      <th align="left"><strong>Model</strong></th>
      <th align="center">ArXiv</th>
      <th align="center">Old Scans Math</th>
      <th align="center">Tables</th>
      <th align="center">Old Scans</th>
      <th align="center">Headers and Footers</th>
      <th align="center">Multi column</th>
      <th align="center">Long tiny text</th>
      <th align="center">Base</th>
      <th align="center">Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">GOT OCR</td>
      <td align="center">52.7</td>
      <td align="center">52.0</td>
      <td align="center">0.20</td>
      <td align="center">22.1</td>
      <td align="center">93.6</td>
      <td align="center">42.0</td>
      <td align="center">29.9</td>
      <td align="center">94.0</td>
      <td align="center">48.3 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">Marker v1.7.5 (base, force_ocr)</td>
      <td align="center">76.0</td>
      <td align="center">57.9</td>
      <td align="center">57.6</td>
      <td align="center">27.8</td>
      <td align="center">84.9</td>
      <td align="center">72.9</td>
      <td align="center">84.6</td>
      <td align="center">99.1</td>
      <td align="center">70.1 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">MinerU v1.3.10</td>
      <td align="center">75.4</td>
      <td align="center">47.4</td>
      <td align="center">60.9</td>
      <td align="center">17.3</td>
      <td align="center"><strong>96.6</strong></td>
      <td align="center">59.0</td>
      <td align="center">39.1</td>
      <td align="center">96.6</td>
      <td align="center">61.5 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">Mistral OCR API</td>
      <td align="center"><strong>77.2</strong></td>
      <td align="center">67.5</td>
      <td align="center">60.6</td>
      <td align="center">29.3</td>
      <td align="center">93.6</td>
      <td align="center">71.3</td>
      <td align="center">77.1</td>
      <td align="center"><strong>99.4</strong></td>
      <td align="center">72.0 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">Nanonets OCR</td>
      <td align="center">67.0</td>
      <td align="center">68.6</td>
      <td align="center"><strong>77.7</strong></td>
      <td align="center">39.5</td>
      <td align="center">40.7</td>
      <td align="center">69.9</td>
      <td align="center">53.4</td>
      <td align="center">99.3</td>
      <td align="center">64.5 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">GPT-4o (No Anchor)</td>
      <td align="center">51.5</td>
      <td align="center"><strong>75.5</strong></td>
      <td align="center">69.1</td>
      <td align="center">40.9</td>
      <td align="center">94.2</td>
      <td align="center">68.9</td>
      <td align="center">54.1</td>
      <td align="center">96.7</td>
      <td align="center">68.9 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">GPT-4o (Anchored)</td>
      <td align="center">53.5</td>
      <td align="center">74.5</td>
      <td align="center">70.0</td>
      <td align="center">40.7</td>
      <td align="center">93.8</td>
      <td align="center">69.3</td>
      <td align="center">60.6</td>
      <td align="center">96.8</td>
      <td align="center">69.9 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">Gemini Flash 2 (No Anchor)</td>
      <td align="center">32.1</td>
      <td align="center">56.3</td>
      <td align="center">61.4</td>
      <td align="center">27.8</td>
      <td align="center">48.0</td>
      <td align="center">58.7</td>
      <td align="center"><strong>84.4</strong></td>
      <td align="center">94.0</td>
      <td align="center">57.8 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">Gemini Flash 2 (Anchored)</td>
      <td align="center">54.5</td>
      <td align="center">56.1</td>
      <td align="center"><strong>72.1</strong></td>
      <td align="center">34.2</td>
      <td align="center">64.7</td>
      <td align="center">61.5</td>
      <td align="center">71.5</td>
      <td align="center">95.6</td>
      <td align="center">63.8 ± 1.2</td>
    </tr>
    <tr>
      <td align="left">Qwen 2 VL (No Anchor)</td>
      <td align="center">19.7</td>
      <td align="center">31.7</td>
      <td align="center">24.2</td>
      <td align="center">17.1</td>
      <td align="center">88.9</td>
      <td align="center">8.3</td>
      <td align="center">6.8</td>
      <td align="center">55.5</td>
      <td align="center">31.5 ± 0.9</td>
    </tr>
    <tr>
      <td align="left">Qwen 2.5 VL (No Anchor)</td>
      <td align="center">63.1</td>
      <td align="center">65.7</td>
      <td align="center">67.3</td>
      <td align="center">38.6</td>
      <td align="center">73.6</td>
      <td align="center">68.3</td>
      <td align="center">49.1</td>
      <td align="center">98.3</td>
      <td align="center">65.5 ± 1.2</td>
    </tr>
    <tr>
      <td align="left">olmOCR v0.1.75 (No Anchor)</td>
      <td align="center">71.5</td>
      <td align="center">71.4</td>
      <td align="center">71.4</td>
      <td align="center"><strong>42.8</strong></td>
      <td align="center">94.1</td>
      <td align="center">77.7</td>
      <td align="center">71.0</td>
      <td align="center">97.8</td>
      <td align="center">74.7 ± 1.1</td>
    </tr>
    <tr>
      <td align="left">olmOCR v0.1.75 (Anchored)</td>
      <td align="center">74.9</td>
      <td align="center">71.2</td>
      <td align="center">71.0</td>
      <td align="center">42.2</td>
      <td align="center">94.5</td>
      <td align="center"><strong>78.3</strong></td>
      <td align="center">73.3</td>
      <td align="center">98.3</td>
      <td align="center"><strong>75.5 ± 1.0</strong></td>
    </tr>
  </tbody>
</table>


Also, the olmOCR project provides an **evaluation toolkit** (`runeval.py`) for side-by-side comparison of PDF conversion
pipeline outputs. This tool allows researchers to directly compare text extraction results from different pipeline
versions against a gold-standard reference. Also olmoOCR authors made some evalutions in
their [technical report](https://olmocr.allenai.org/papers/olmocr.pdf).


> We then sampled 2,000 comparison pairs (same PDF, different tool). We asked 11 data researchers and
> engineers at Ai2 to assess which output was the higher quality representation of the original PDF, focusing on
> reading order, comprehensiveness of content and representation of structured information. The user interface
> used is similar to that in Figure 5. Exact participant instructions are listed in Appendix B.

**Bootstrapped Elo Ratings (95% CI)**

| Model   | Elo Rating ± CI | 95% CI Range     |
| ------- | --------------- | ---------------- |
| olmoOCR | 1813.0 ± 84.9   | [1605.9, 1930.0] |
| MinerU  | 1545.2 ± 99.7   | [1336.7, 1714.1] |
| Marker  | 1429.1 ± 100.7  | [1267.6, 1645.5] |
| GOTOCOR | 1212.7 ± 82.0   | [1097.3, 1408.3] |

<br/>

> Table 7: Pairwise Win/Loss Statistics Between Models

| Model Pair         | Wins    | Win Rate (%) |
| ------------------ | ------- | ------------ |
| olmOCR vs. Marker  | 49/31   | **61.3**     |
| olmOCR vs. GOTOCOR | 41/29   | **58.6**     |
| olmOCR vs. MinerU  | 55/22   | **71.4**     |
| Marker vs. MinerU  | 53/26   | 67.1         |
| Marker vs. GOTOCOR | 45/26   | 63.4         |
| GOTOCOR vs. MinerU | 38/37   | 50.7         |
| **Total**          | **452** |              |

### [Marker benchmarks](https://github.com/VikParuchuri/marker?tab=readme-ov-file#benchmarks)

[![GitHub last commit](https://img.shields.io/github/last-commit/VikParuchuri/marker?label=GitHub&logo=github)](https://github.com/VikParuchuri/marker?tab=readme-ov-file#benchmarks)
![GitHub License](https://img.shields.io/github/license/VikParuchuri/marker)
<!--- 
License: GPL 3.0
Primary language: Python
-->

The Marker repository provides benchmark results comparing various PDF processing methods, scored based on a heuristic
that aligns text with ground truth text segments, and an LLM as a judge scoring method.

| Method     | Avg Time | Heuristic Score | LLM Score |
| ---------- | -------- | --------------- | --------- |
| marker     | 2.83837  | 95.6709         | 4.23916   |
| llamaparse | 23.348   | 84.2442         | 3.97619   |
| mathpix    | 6.36223  | 86.4281         | 4.15626   |
| docling    | 3.69949  | 86.7073         | 3.70429   |

### [READoc](https://arxiv.org/abs/2409.05137)

[![GitHub last commit](https://img.shields.io/github/last-commit/icip-cas/READoc?label=GitHub&logo=github)](https://github.com/icip-cas/READoc)
[![arXiv](https://img.shields.io/badge/arXiv-2409.05137-b31b1b)](https://arxiv.org/abs/2409.05137)

| Methods                    | Text (Concat) | Text (Vocab) | Heading (Concat) | Heading (Tree) | Formula (Embed) | Formula (Isolate) | Table (Concat) | Table (Tree) | Reading Order (Block) | Reading Order (Token) | Average |
| -------------------------- | ------------- | ------------ | ---------------- | -------------- | --------------- | ----------------- | -------------- | ------------ | --------------------- | --------------------- | ------- |
| **Baselines**              |               |              |                  |                |                 |                   |                |              |                       |                       |         |
| PyMuPDF4LLM                | 66.66         | 74.27        | 27.86            | 20.77          | 0.07            | 0.02              | 23.27          | 15.83        | 87.70                 | 89.09                 | 40.55   |
| Tesseract OCR              | 78.85         | 76.51        | 1.26             | 0.30           | 0.12            | 0.00              | 0.00           | 0.00         | 96.70                 | 97.59                 | 35.13   |
| **Pipeline Tools**         |               |              |                  |                |                 |                   |                |              |                       |                       |         |
| MinerU                     | 84.15         | 84.76        | 62.89            | 39.15          | 62.97           | 71.02             | 0.00           | 0.00         | 98.64                 | 97.72                 | 60.17   |
| Pix2Text                   | 85.85         | 83.72        | 63.23            | 34.53          | 43.18           | 37.45             | 54.08          | 47.35        | 97.68                 | 96.78                 | 64.39   |
| Marker                     | 83.58         | 81.36        | 68.78            | 54.82          | 5.07            | 56.26             | 47.12          | 43.35        | 98.08                 | 97.26                 | 63.57   |
| **Expert Visual Models**   |               |              |                  |                |                 |                   |                |              |                       |                       |         |
| Nougat-small               | 87.35         | 92.00        | 86.40            | 87.88          | 76.52           | 79.39             | 55.63          | 52.35        | 97.97                 | 98.36                 | 81.38   |
| Nougat-base                | 88.03         | 92.29        | 86.60            | 88.50          | 76.19           | 79.47             | 54.40          | 52.30        | 97.98                 | 98.41                 | 81.42   |
| **Vision-Language Models** |               |              |                  |                |                 |                   |                |              |                       |                       |         |
| DeepSeek-VL-7B-Chat        | 31.89         | 39.96        | 23.66            | 12.53          | 17.01           | 16.94             | 22.96          | 16.47        | 88.76                 | 66.75                 | 33.69   |
| MiniCPM-Llama3-V2.5        | 58.91         | 70.87        | 26.33            | 7.68           | 16.70           | 17.90             | 27.89          | 24.91        | 95.26                 | 93.02                 | 43.95   |
| LLaVa-1.6-Vicuna-13B       | 27.51         | 37.09        | 8.92             | 6.27           | 17.80           | 11.68             | 23.78          | 16.23        | 76.63                 | 51.68                 | 27.76   |
| InternVL-Chat-V1.5         | 53.06         | 68.44        | 25.03            | 13.57          | 33.13           | 24.37             | 40.44          | 34.35        | 94.61                 | 91.31                 | 47.83   |
| GPT-4o-mini                | 79.44         | 84.37        | 31.77            | 18.65          | 42.23           | 41.67             | 47.81          | 39.85        | 97.69                 | 96.35                 | 57.98   |

**Table 3:** Evaluation of various Document Structured Extraction systems on READOC-arXiv.

### [Mistral-OCR benchmarks](https://mistral.ai/news/mistral-ocr)

| Model                | Overall   | Math      | Multilingual | Scanned   | Tables    |
| -------------------- | --------- | --------- | ------------ | --------- | --------- |
| Google Document AI   | 83.42     | 80.29     | 86.42        | 92.77     | 78.16     |
| Azure OCR            | 89.52     | 85.72     | 87.52        | 94.65     | 89.52     |
| Gemini-1.5-Flash-002 | 90.23     | 89.11     | 86.76        | 94.87     | 90.48     |
| Gemini-1.5-Pro-002   | 89.92     | 88.48     | 86.33        | 96.15     | 89.71     |
| Gemini-2.0-Flash-001 | 88.69     | 84.18     | 85.80        | 95.11     | 91.46     |
| GPT-4o-2024-11-20    | 89.77     | 87.55     | 86.00        | 94.58     | 91.70     |
| Mistral OCR 2503     | **94.89** | **94.29** | **89.55**    | **98.96** | **96.12** |

### [dp-bench](https://huggingface.co/datasets/upstage/dp-bench)

| Source       | Request date | TEDS ↑ | TEDS-S ↑ | NID ↑ | Avg. Time (secs) ↓ |
| ------------ | ------------ | ------ | -------- | ----- | ------------------ |
| upstage      | 2024-10-24   | 93.48  | 94.16    | 97.02 | 3.79               |
| aws          | 2024-10-24   | 88.05  | 90.79    | 96.71 | 14.47              |
| llamaparse   | 2024-10-24   | 74.57  | 76.34    | 92.82 | 4.14               |
| unstructured | 2024-10-24   | 65.56  | 70.00    | 91.18 | 13.14              |
| google       | 2024-10-24   | 66.13  | 71.58    | 90.86 | 5.85               |
| microsoft    | 2024-10-24   | 87.19  | 89.75    | 87.69 | 4.44               |

### [Actualize pro](https://www.actualize.pro/recourses/unlocking-insights-from-pdfs-a-comparative-study-of-extraction-tools)

[![GitHub last commit](https://img.shields.io/github/last-commit/actualize-ae/pdf-benchmarking?label=GitHub&logo=github)](https://github.com/actualize-ae/pdf-benchmarking)

> In the digital age, PDF documents remain a cornerstone for disseminating and archiving information.
> However, extracting meaningful data from these structured and unstructured formats continues to challenge modern AI
> systems.
> Our recent benchmarking study evaluated seven prominent PDF extraction tools to determine their capabilities across
> diverse document types and applications.

| PDF Parser   | Overall Score (out of 10) | Text Extraction Accuracy (Score out of 10) | Table Extraction Accuracy (Score out of 10) | Reading Order Accuracy (Score out of 10) | Markdown Conversion Accuracy (Score out of 10) | Code and Math Equations Extraction (Score out of 10) | Image Extraction Accuracy (Score out of 10) |
| ------------ | ------------------------- | ------------------------------------------ | ------------------------------------------- | ---------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| MinerU       | 8                         | 9.3                                        | 7.3                                         | 8.7                                      | 8.3                                            | 6.5                                                  | 7                                           |
| Xerox        | 7.9                       | 8.7                                        | 7.7                                         | 9                                        | 8.7                                            | 7                                                    | 6                                           |
| MarkItdown   | 7.78                      | 9                                          | 6.83                                        | 9                                        | 7.67                                           | 7.83                                                 | 5.83                                        |
| Docling      | 7.3                       | 8.7                                        | 6.3                                         | 9                                        | 8                                              | 6.5                                                  | 5                                           |
| Llama parse  | 7.1                       | 7.3                                        | 7.7                                         | 8.7                                      | 7.3                                            | 6                                                    | 5.3                                         |
| Marker       | 6.5                       | 7.3                                        | 5.7                                         | 7.3                                      | 6.7                                            | 4.5                                                  | 6.7                                         |
| Unstructured | 6.2                       | 7.3                                        | 5                                           | 8.3                                      | 6.7                                            | 5                                                    | 4.7                                         |

### [liduos.com](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)

| Function                                          | MinerU | PaddleOCR | Marker | Unstructured | gptpdf | Zerox | Chunkr | pdf-extract-api | Sparrow | LlamaParse | DeepDoc | MegaParse |
| ------------------------------------------------- | ------ | --------- | ------ | ------------ | ------ | ----- | ------ | --------------- | ------- | ---------- | ------- | --------- |
| PDF and Image Parsing                             | ✓      | ✓         | ✓      | ✓            | ✓      | ✓     | ✓      | ✓               | ✓       | ✓          | ✓       | ✓         |
| Parsing of Other Formats (PPT, Excel, DOCX, etc.) | ✓      | -         | -      | ✓            | -      | ✓     | ✓      | -               | ✓       | ✓          | ✓       | ✓         |
| Layout Analysis                                   | ✓      | ✓         | ✓      | -            | ✓      | -     | ✓      | -               | -       | ✓          | ✓       | -         |
| Text Recognition                                  | ✓      | ✓         | ✓      | ✓            | ✓      | ✓     | ✓      | ✓               | ✓       | ✓          | ✓       | ✓         |
| Image Recognition                                 | ✓      | ✓         | ✓      | ✓            | ✓      | ✓     | ✓      | ✓               | ✓       | ✓          | ✓       | ✓         |
| Simple (Vertical/Horizontal/Hierarchical) Tables  | ✓      | ✓         | ✓      | ✓            | ✓      | ✓     | ✓      | ✓               | ✓       | ✓          | ✓       | ✓         |
| Complex Tables                                    | -      | -         | -      | -            | -      | -     | -      | -               | -       | -          | -       | -         |
| Formula Recognition                               | -      | -         | -      | -            | -      | -     | -      | -               | -       | -          | -       | -         |
| HTML Output                                       | ✓      | -         | ✓      | ✓            | -      | -     | ✓      | -               | -       | -          | ✓       | -         |
| Markdown Output                                   | ✓      | ✓         | ✓      | -            | ✓      | ✓     | ✓      | ✓               | ✓       | ✓          | -       | ✓         |
| JSON Output                                       | ✓      | -         | ✓      | ✓            | -      | -     | ✓      | ✓               | -       | ✓          | ✓       | -         |

### [Omni OCR Benchmark](https://getomni.ai/ocr-benchmark)

[![GitHub last commit](https://img.shields.io/github/last-commit/yobix-ai/extractous-benchmarks?label=GitHub&logo=github)](https://github.com/getomni-ai/benchmark)
![GitHub License](https://img.shields.io/github/license/getomni-ai/benchmark)

**JSON Accuracy**

| Model Provider     | JSON Accuracy (%) |
| ------------------ | ----------------- |
| OmniAI             | 91.7%             |
| Gemini 2.0 Flash   | 86.1%             |
| Azure              | 85.1%             |
| GPT-4o             | 75.5%             |
| AWS Textract       | 74.3%             |
| Claude Sonnet 3.5  | 69.3%             |
| Google Document AI | 67.8%             |
| GPT-4o Mini        | 64.8%             |
| Unstructured       | 50.8%             |

**Cost per 1,000 Pages**

| Model Provider     | Cost per 1,000 Pages ($) |
| ------------------ | ------------------------ |
| GPT-4o Mini        | 0.97                     |
| Gemini 2.0 Flash   | 1.12                     |
| Google Document AI | 1.50                     |
| AWS Textract       | 4.00                     |
| OmniAI             | 10.00                    |
| Azure              | 10.00                    |
| GPT-4o             | 18.37                    |
| Claude Sonnet 3.5  | 19.93                    |
| Unstructured       | 20.00                    |

**Processing Time per Page**

| Model Provider     | Average Latency (seconds) |
| ------------------ | ------------------------- |
| Google Document AI | 3.19                      |
| Azure              | 4.40                      |
| AWS Textract       | 4.86                      |
| Unstructured       | 7.99                      |
| OmniAI             | 9.69                      |
| Gemini 2.0 Flash   | 10.71                     |
| Claude Sonnet 3.5  | 18.42                     |
| GPT-4o Mini        | 22.73                     |
| GPT-4o             | 24.85                     |

### [Extractous benchmarks](https://github.com/yobix-ai/extractous-benchmarks/tree/main/docs)

[![GitHub last commit](https://img.shields.io/github/last-commit/yobix-ai/extractous-benchmarks?label=GitHub&logo=github)](https://github.com/yobix-ai/extractous-benchmarks/tree/main/docs)
![GitHub License](https://img.shields.io/github/license/yobix-ai/extractous-benchmarks)

[`extractous`](https://github.com/yobix-ai/extractous) speedup relative to [
`unstructured-io`](https://github.com/Unstructured-IO/unstructured)

![image](https://github.com/user-attachments/assets/6d9bc6ba-8e1a-4083-9d6f-864adf854e2f)

[`extractous`](https://github.com/yobix-ai/extractous) memory efficiency relative to [
`unstructured-io`](https://github.com/Unstructured-IO/unstructured)

![image](https://github.com/user-attachments/assets/e6236232-4fa3-4cd0-8cfa-0bfcd5bc18e3)

