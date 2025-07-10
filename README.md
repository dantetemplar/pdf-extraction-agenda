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

| Pipeline                                  | [OmniDocBench](#omnidocbench) Overall ↓ | [olmOCR](#olmoocr-eval) Overall ↑ | [Omni OCR](#omni-ocr-benchmark) Accuracy ↑ | [Marker](#marker-benchmarks) Overall ↓ | [Mistral](#mistral-ocr-benchmarks) Overall ↑ | [dp-bench](#dp-bench) NID ↑ | [READoc](#readoc) Overall ↑ | [Actualize.pro](#actualize-pro) Overall ↑ |
| ----------------------------------------- | --------------------------------------- | --------------------------------- | :----------------------------------------- | -------------------------------------- | :------------------------------------------- | --------------------------- | --------------------------- | ----------------------------------------- |
| [MinerU](#MinerU)                         | 0.150 <sup>[3]</sup> ⚠️                 | 61.5                              |                                            |                                        |                                              |                             | 60.17                       | **8**                                     |
| [Marker](#Marker)                         | 0.336                                   | 70.1                              |                                            | **4.24** ⚠️                            |                                              |                             | 63.57                       | 6.5                                       |
| [DocLing](#DocLing)                       | 0.589                                   |                                   |                                            | 3.70                                   |                                              |                             |                             | 7.3                                       |
| [MonkeyOCR (pro-3B)](#MonkeyOCR)          | **0.138 <sup>[1]</sup>** ⚠️             | **75.8 <sup>[1]</sup>** ⚠️        |                                            |                                        |                                              |                             |                             |                                           |
| [MarkItDown](#MarkItDown)                 |                                         |                                   |                                            |                                        |                                              |                             |                             | 7.78                                      |
| [Zerox (OmniAI)](#Zerox)                  |                                         |                                   | **91.7 <sup>[1]</sup>** ⚠️                 |                                        |                                              |                             |                             | 7.9                                       |
| [Unstructured](#Unstructured)             | 0.586                                   |                                   | 50.8                                       |                                        |                                              | 91.18                       |                             | 6.2                                       |
| [Pix2Text](#Pix2Text)                     | 0.32                                    |                                   |                                            |                                        |                                              |                             | 64.39                       |                                           |
| [open-parse](#open-parse)                 | 0.646                                   |                                   |                                            |                                        |                                              |                             |                             |                                           |
| [Markdrop](#markdrop)                     |                                         |                                   |                                            |                                        |                                              |                             |                             |                                           |
| [Vision Parse](#Vision-Parse)             |                                         |                                   |                                            |                                        |                                              |                             |                             |                                           |
| [olmOCR](#olmOCR)                         | 0.326                                   | 75.5 <sup>[2]</sup> ⚠️            |                                            |                                        |                                              |                             |                             |                                           |
| _↓ Proprietary pipelines_                 |                                         |                                   |                                            |                                        |                                              |                             |                             |                                           |
| [Mistral OCR](#MistralOCR)                | 0.268                                   | 72.0 <sup>[3]</sup>               |                                            |                                        | **94.89 ⚠️**                                 |                             |                             |                                           |
| [Google Document AI](#Google-Document-AI) |                                         |                                   | 67.8                                       |                                        | 83.42                                        | 90.86                       |                             |                                           |
| [Azure OCR](#Azure-OCR)                   |                                         |                                   | 85.1                                       |                                        | 89.52                                        | 87.69                       |                             |                                           |
| [Amazon Textract](#Amazon-Textract)       |                                         |                                   | 74.3                                       |                                        |                                              | 96.71                       |                             |                                           |
| [LlamaParse](#LlamaParse)                 |                                         |                                   |                                            | 3.98                                   |                                              | 92.82                       |                             | 7.1                                       |
| [Mathpix](#Mathpix)                       | 0.191                                   |                                   |                                            | 4.16                                   |                                              |                             |                             |                                           |
| [upstage](#upstage-ai)                    |                                         |                                   |                                            |                                        |                                              | **97.02**  ⚠️               |                             |                                           |
| _↓ Expert VLMs_                           |                                         |                                   |                                            |                                        |                                              |                             |                             |                                           |
| [Nougat](#Nougat)                         | 0.452                                   |                                   |                                            |                                        |                                              |                             | **81.42**                   |                                           |
| [GOT-OCR](#GOT-OCR)                       | 0.287                                   | 48.3                              |                                            |                                        |                                              |                             |                             |                                           |
| [SmolDocling](#SmolDocling)               | 0.493                                   |                                   |                                            |                                        |                                              |                             |                             |                                           |
| Nanonets-OCR                              |                                         | 64.5                              |                                            |                                        |                                              |                             |                             |                                           |
| _↓ General VLMs_                          |                                         |                                   |                                            |                                        |                                              |                             |                             |                                           |
| Gemini-1.5 Flash                          |                                         |                                   |                                            |                                        | 90.23                                        |                             |                             |                                           |
| Gemini-1.5 Pro                            |                                         |                                   |                                            |                                        | 89.92                                        |                             |                             |                                           |
| Gemini-2.0 Flash                          | 0.191                                   | 63.8                              | 86.1 <sup>[2]</sup>                        |                                        | 88.69                                        |                             |                             |                                           |
| Gemini-2.5 Pro                            | 0.148 <sup>[2]</sup>                    |                                   |                                            |                                        |                                              |                             |                             |                                           |
| GPT4o                                     | 0.233                                   | 69.9                              | 75.5                                       |                                        | 89.77                                        |                             |                             |                                           |
| Claude Sonnet 3.5                         |                                         |                                   | 69.3                                       |                                        |                                              |                             |                             |                                           |
| Qwen2-VL-72B                              | 0.252                                   |                                   |                                            |                                        |                                              |                             |                             |                                           |
| Qwen2.5-VL-72B                            | 0.214                                   | 65.5                              |                                            |                                        |                                              |                             |                             |                                           |
| InternVL2-76B                             | 0.44                                    |                                   |                                            |                                        |                                              |                             |                             |                                           |

- **Bold** indicates the best result for a given metric, and <sup>[2]</sup> indicates 2nd place in that benchmark.
- " " means the pipeline was not evaluated in that benchmark.
- ⚠️ means the pipeline authors are the ones who did the benchmark.
- `Overall ↑` in column name means higher value is better, when `Overall ↓` - lower value is better.

> [!NOTE]
> I'm working on implementing an easy-to-repeat benchmarking (just run notebook on colab to repeat results, or extend
> them), but for now I'm struggling with finding suitable dataset.

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

### [MistralOCR](https://mistral.ai/news/mistral-ocr)

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

