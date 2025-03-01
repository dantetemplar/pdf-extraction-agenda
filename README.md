# PDF extraction pipelines and benchmarks agenda

> [!CAUTION]
> Part of text in this repo written by ChatGPT. Also, I haven't yet run all pipelines because of lack of compute power.

This repository provides an overview of notable **pipelines** and **benchmarks** related to PDF/OCR document
processing. Each entry includes a brief description, and useful data.

## Table of contents

Did you know that GitHub supports table of
contents [by default](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/) 🤔

## Comparison

| Pipeline                  | [OmniDocBench](#omnidocbench) Overall Edit ↓ | [olmOCR Eval](#olmoocr-eval) Elo Rating ↑ | [Marker Bench](#marker-benchmarks) Overall Edit ↓ | [READoc](#readoc) Overall Score ↑ | [Actualize.pro](#actualize-pro) Overall Score ↑ |
|---------------------------|----------------------------------------------|-------------------------------------------|---------------------------------------------------|-----------------------------------|-------------------------------------------------|
| [MinerU](#MinerU)         | **0.150**    ⚠️                              | 1545.2                                    | -                                                 | 60.17                             | **8**                                           |
| [Marker](#Marker)         | 0.336                                        | 1429.1                                    | **4.23916**  ⚠️                                   | 63.57                             | 6.5                                             |
| [Mathpix](#Mathpix)       | 0.189                                        | -                                         | 4.15626                                           | -                                 | -                                               |
| [DocLing](#DocLing)       | 0.589                                        | -                                         | 3.70429                                           | -                                 | 7.3                                             |
| [GOT-OCR](#GOT-OCR)       | 0.289                                        | 1212.7                                    | -                                                 | -                                 | -                                               |
| [olmOCR](#olmOCR)         | -                                            | **1813.0**   ⚠️                           | -                                                 | -                                 | -                                               |
| [LlamaParse](#LlamaParse) | -                                            | -                                         | 3.97619                                           | -                                 | 7.1                                             |
| [MarkItDown](#MarkItDown) | -                                            | -                                         | -                                                 | -                                 | 7.78                                            |
| Xerox                     | -                                            | -                                         | -                                                 | -                                 | 7.9                                             |
| Unstructured              | -                                            | -                                         | -                                                 | -                                 | 6.2                                             |
| Pix2Text                  | -                                            | -                                         | -                                                 | **64.39**                         | -                                               |

- **Bold** indicates the best result for a given metric.
- "-" means the pipeline was not evaluated in that benchmark.
- ⚠️ means the pipeline authors are the ones who did the benchmark.

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
      <th colspan="2">Text<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>CDM</sup>↑</th>
      <th colspan="2">Table<sup>TEDS</sup>↑</th>
      <th colspan="2">Table<sup>Edit</sup>↓</th>
      <th colspan="2">Read Order<sup>Edit</sup>↓</th>
      <th colspan="2">Overall<sup>Edit</sup>↓</th>
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
      <td rowspan="4">Pipeline Tools</td>
      <td>MinerU-0.9.3</td>
      <td><strong>0.061</strong></td>
      <td><strong>0.211</strong></td>
      <td><strong>0.278</strong></td>
      <td>0.577</td>
      <td>66.9</td>
      <td>49.5</td>
      <td><strong>78.6</strong></td>
      <td>62.1</td>
      <td><strong>0.180</strong></td>
      <td>0.344</td>
      <td><strong>0.079</strong></td>
      <td>0.288</td>
      <td><strong>0.150</strong></td>
      <td><u>0.355</u></td>
    </tr>
    <tr>
      <td>Marker-1.2.3</td>
      <td><u>0.080</u></td>
      <td>0.315</td>
      <td>0.530</td>
      <td>0.883</td>
      <td>20.1</td>
      <td>16.8</td>
      <td>67.6</td>
      <td>49.2</td>
      <td>0.619</td>
      <td>0.685</td>
      <td>0.114</td>
      <td>0.340</td>
      <td>0.336</td>
      <td>0.556</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td>0.101</td>
      <td>0.358</td>
      <td><u>0.306</u></td>
      <td><strong>0.454</strong></td>
      <td>71.4</td>
      <td><strong>72.7</strong></td>
      <td><u>77.0</u></td>
      <td><strong>67.1</strong></td>
      <td>0.243</td>
      <td><strong>0.320</strong></td>
      <td><u>0.105</u></td>
      <td>0.275</td>
      <td><u>0.189</u></td>
      <td><strong>0.352</strong></td>
    </tr>
    <tr>
      <td>Docling</td>
      <td>0.416</td>
      <td>0.987</td>
      <td>0.999</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>61.3</td>
      <td>25.0</td>
      <td>0.627</td>
      <td>0.810</td>
      <td>0.313</td>
      <td>0.837</td>
      <td>0.589</td>
      <td>0.909</td>
    </tr>
    <tr>
      <td rowspan="2">Expert VLMs</td>
      <td>GOT-OCR</td>
      <td>0.191</td>
      <td>0.315</td>
      <td>0.360</td>
      <td><u>0.528</u></td>
      <td><strong>81.8</strong></td>
      <td>51.4</td>
      <td>53.2</td>
      <td>47.2</td>
      <td>0.459</td>
      <td>0.520</td>
      <td>0.143</td>
      <td>0.280</td>
      <td>0.289</td>
      <td>0.411</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.367</td>
      <td>0.998</td>
      <td>0.488</td>
      <td>0.941</td>
      <td>17.4</td>
      <td>16.9</td>
      <td>39.9</td>
      <td>0</td>
      <td>0.572</td>
      <td>1</td>
      <td>0.384</td>
      <td>0.954</td>
      <td>0.453</td>
      <td>0.973</td>
    </tr>
    <tr>
      <td rowspan="3">General VLMs</td>
      <td>GPT4o</td>
      <td>0.146</td>
      <td>0.409</td>
      <td>0.425</td>
      <td>0.606</td>
      <td><u>76.4</u></td>
      <td>48.2</td>
      <td>72.0</td>
      <td>62.9</td>
      <td><u>0.234</u></td>
      <td><u>0.329</u></td>
      <td>0.128</td>
      <td>0.251</td>
      <td>0.233</td>
      <td>0.399</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>0.253</td>
      <td><u>0.251</u></td>
      <td>0.468</td>
      <td>0.572</td>
      <td>54.9</td>
      <td><u>60.9</u></td>
      <td>59.5</td>
      <td><u>66.4</u></td>
      <td>0.551</td>
      <td>0.518</td>
      <td>0.254</td>
      <td><strong>0.223</strong></td>
      <td>0.381</td>
      <td>0.391</td>
    </tr>
    <tr>
      <td>InternVL2-76B</td>
      <td>0.353</td>
      <td>0.29</td>
      <td>0.543</td>
      <td>0.701</td>
      <td>69.8</td>
      <td>49.6</td>
      <td>63.0</td>
      <td>60.2</td>
      <td>0.547</td>
      <td>0.555</td>
      <td>0.317</td>
      <td><u>0.228</u></td>
      <td>0.440</td>
      <td>0.443</td>
    </tr>
  </tbody>
</table>
<p style="text-align: center; margin-top: -4pt;">
  Comprehensive evaluation of document parsing algorithms on OmniDocBench: performance metrics for text, formula, table, and reading order extraction, with overall scores derived from ground truth comparisons.
</p>

### [olmoOCR eval](https://github.com/allenai/olmocr)

[![GitHub last commit](https://img.shields.io/github/last-commit/allenai/olmocr?label=GitHub&logo=github)](https://github.com/allenai/olmocr)
![GitHub License](https://img.shields.io/github/license/allenai/olmocr)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

The olmOCR project provides an **evaluation toolkit** (`runeval.py`) for side-by-side comparison of PDF conversion
pipeline outputs. This tool allows researchers to directly compare text extraction results from different pipeline
versions against a gold-standard reference. Also olmoOCR authors made some evalutions in
their [technical report](https://olmocr.allenai.org/papers/olmocr.pdf).


> We then sampled 2,000 comparison pairs (same PDF, different tool). We asked 11 data researchers and
> engineers at Ai2 to assess which output was the higher quality representation of the original PDF, focusing on
> reading order, comprehensiveness of content and representation of structured information. The user interface
> used is similar to that in Figure 5. Exact participant instructions are listed in Appendix B.

**Bootstrapped Elo Ratings (95% CI)**

| Model   | Elo Rating ± CI | 95% CI Range     |
|---------|-----------------|------------------|
| olmoOCR | 1813.0 ± 84.9   | [1605.9, 1930.0] |
| MinerU  | 1545.2 ± 99.7   | [1336.7, 1714.1] |
| Marker  | 1429.1 ± 100.7  | [1267.6, 1645.5] |
| GOTOCOR | 1212.7 ± 82.0   | [1097.3, 1408.3] |

<br/>

> Table 7: Pairwise Win/Loss Statistics Between Models

| Model Pair         | Wins    | Win Rate (%) |
|--------------------|---------|--------------|
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
|------------|----------|-----------------|-----------|
| marker     | 2.83837  | 95.6709         | 4.23916   |
| llamaparse | 23.348   | 84.2442         | 3.97619   |
| mathpix    | 6.36223  | 86.4281         | 4.15626   |
| docling    | 3.69949  | 86.7073         | 3.70429   |

### [READoc](https://arxiv.org/abs/2409.05137)

[![GitHub last commit](https://img.shields.io/github/last-commit/icip-cas/READoc?label=GitHub&logo=github)](https://github.com/icip-cas/READoc)
[![arXiv](https://img.shields.io/badge/arXiv-2409.05137-b31b1b)](https://arxiv.org/abs/2409.05137)

| Methods                    | Text (Concat) | Text (Vocab) | Heading (Concat) | Heading (Tree) | Formula (Embed) | Formula (Isolate) | Table (Concat) | Table (Tree) | Reading Order (Block) | Reading Order (Token) | Average |
|----------------------------|---------------|--------------|------------------|----------------|-----------------|-------------------|----------------|--------------|-----------------------|-----------------------|---------|
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

### [Actualize pro](https://www.actualize.pro/recourses/unlocking-insights-from-pdfs-a-comparative-study-of-extraction-tools)

[![GitHub last commit](https://img.shields.io/github/last-commit/actualize-ae/pdf-benchmarking?label=GitHub&logo=github)](https://github.com/actualize-ae/pdf-benchmarking)

> In the digital age, PDF documents remain a cornerstone for disseminating and archiving information.
> However, extracting meaningful data from these structured and unstructured formats continues to challenge modern AI
> systems.
> Our recent benchmarking study evaluated seven prominent PDF extraction tools to determine their capabilities across
> diverse document types and applications.

| PDF Parser   | Overall Score (out of 10) | Text Extraction Accuracy (Score out of 10) | Table Extraction Accuracy (Score out of 10) | Reading Order Accuracy (Score out of 10) | Markdown Conversion Accuracy (Score out of 10) | Code and Math Equations Extraction (Score out of 10) | Image Extraction Accuracy (Score out of 10) |
|--------------|---------------------------|--------------------------------------------|---------------------------------------------|------------------------------------------|------------------------------------------------|------------------------------------------------------|---------------------------------------------|
| MinerU       | 8                         | 9.3                                        | 7.3                                         | 8.7                                      | 8.3                                            | 6.5                                                  | 7                                           |
| Xerox        | 7.9                       | 8.7                                        | 7.7                                         | 9                                        | 8.7                                            | 7                                                    | 6                                           |
| MarkItdown   | 7.78                      | 9                                          | 6.83                                        | 9                                        | 7.67                                           | 7.83                                                 | 5.83                                        |
| Docling      | 7.3                       | 8.7                                        | 6.3                                         | 9                                        | 8                                              | 6.5                                                  | 5                                           |
| Llama parse  | 7.1                       | 7.3                                        | 7.7                                         | 8.7                                      | 7.3                                            | 6                                                    | 5.3                                         |
| Marker       | 6.5                       | 7.3                                        | 5.7                                         | 7.3                                      | 6.7                                            | 4.5                                                  | 6.7                                         |
| Unstructured | 6.2                       | 7.3                                        | 5                                           | 8.3                                      | 6.7                                            | 5                                                    | 4.7                                         |

### [liduos.com](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)

| Function                                          | MinerU | PaddleOCR | Marker | Unstructured | gptpdf | Zerox | Chunkr | pdf-extract-api | Sparrow | LlamaParse | DeepDoc | MegaParse |
|---------------------------------------------------|--------|-----------|--------|--------------|--------|-------|--------|-----------------|---------|------------|---------|-----------|
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

<!---
## Extraction Properties

This section defines the properties available for configuring the extraction process. Each property governs a specific aspect of how PDFs are processed.


### PDF Processing Type

This property determines how the PDF content is processed based on its format and structure.

#### Available Options

- **Native PDF**  
  Extracts text directly from machine-readable PDFs without any conversion. This method is the fastest and retains the highest accuracy for structured content.

- **Converting to Image**  
  Converts PDF pages into images before applying OCR. This method is used for scanned or image-based PDFs but may introduce OCR-related errors.

- **Virtually Hybrid**  
  Routes direct text extraction and image conversion with OCR. If a page contains selectable text, it is extracted natively; otherwise, the page is converted to an image and processed with OCR.

- **Hybrid**  
  Processes both text and visual elements in PDFs simultaneously (f.e. sending anchored text and page image to VLLM), leveraging both native extraction and OCR-based or VLLM image analysis in a unified approach.

#### Choosing the Right Option
- If working with digital PDFs that contain selectable text and no images of figures, **Native PDF** provides the best results.
- For scanned documents, at least **Converting to Image** is required to retrieve text using OCR.
- If PDFs contain a mix of text-based pages and image-based pages, **Virtually Hybrid** ensures maximum text extraction accuracy.
- If both text and visual data are crucial for analysis, **Hybrid** is the most comprehensive option, and actually it is rare approach.

### Embedded Graphics Handling

When processing PDFs, embedded graphics such as vector diagrams, charts, and images can be handled in different ways depending on the extraction strategy. This property determines how these graphics are treated during extraction and whether additional processing, such as OCR, is applied.

#### Available Options

- **Ignore**  
  The extraction process does not process embedded graphics. They remain in the PDF, and no additional files or links are generated.

- **Replace with OCR**  
  Embedded graphics are removed from the output, and OCR is applied to any raster images within the PDF. The extracted text replaces the original graphic content where possible.

- **Extract to Folder with Markdown Link**  
  Vector-based graphics and embedded images are extracted to a separate folder. In the Markdown output, a reference is added using a standard image link format:  
  ```markdown
  ![Extracted Graphic](path/to/image.png)
  ```

- **Extract to Folder with Markdown Link and OCR Comment**  
  This mode extracts graphics to a folder, inserts a Markdown link to the extracted image, and appends any OCR-extracted text as a comment below the reference. Example:
  ```markdown
  ![Extracted Graphic](path/to/image.png)
  
  [comment]: # (OCR: "Figure 2 - Sales Growth Trends")
  ```

#### Choosing the Right Option
- If the PDF contains machine-readable text and graphics are not required, **Ignore** is the most efficient choice.
- When working with scanned documents containing important visual text, **Replace with OCR** ensures that no information is lost.
- For preserving visual elements while keeping Markdown structured, **Extract to Folder with Markdown Link** is recommended.
- If both graphics and extracted text are needed, **Extract to Folder with Markdown Link and OCR Comment** provides the most comprehensive output.
-->
