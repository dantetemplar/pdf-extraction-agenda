# PDF extraction pipelines and benchmarks agenda

This repository provides an overview of selected **pipeline** and **benchmark** repositories related to PDF/OCR document processing. Each entry includes a brief description, latest commit date, contributor count, license, primary language, and notable features for quick reference.

## Pipelines

### MinerU
[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/MinerU?label=GitHub&logo=github)](https://github.com/opendatalab/MinerU)
![GitHub License](https://img.shields.io/github/license/opendatalab/MinerU)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=huggingface)](https://huggingface.co/spaces/opendatalab/MinerU)
<!--- 
License: AGPL-3.0 
Primary language: Python
-->

MinerU is described as *“a high-quality tool for convert PDF to Markdown and JSON”*, serving as a one-stop open-source solution for high-quality data extraction from PDFs. It supports conversion of PDFs into machine-readable formats (Markdown, JSON) for easy data extraction.

**Notable features:** Provides a new API with composable processing stages and a `Dataset` class supporting multiple document formats (images, PDFs, Word, PPT, etc.). It includes advanced capabilities like automatic language identification for OCR (selecting the appropriate model from 84 supported languages). MinerU has a focus on performance and compatibility (optimized for ARM Linux and integrated with Huawei Ascend NPU for acceleration) and implements robust layout analysis (e.g. reading order with `layoutreader`) and table recognition modules to improve parsing accuracy.

### Marker
[![GitHub last commit](https://img.shields.io/github/last-commit/VikParuchuri/marker?label=GitHub&logo=github)](https://github.com/VikParuchuri/marker)
![GitHub License](https://img.shields.io/github/license/VikParuchuri/marker)
[![Demo](https://img.shields.io/badge/DEMO%20after%20registration-black?logo=awwwards)](https://olmocr.allenai.org/)
<!--- 
License: GPL 3.0
Primary language: Python
-->

Marker *“converts PDFs and images to markdown, JSON, and HTML quickly and accurately.”* It is designed to handle a wide range of document types in all languages and produce structured outputs.

**Demo available after registration on https://www.datalab.to/**

**Notable features:** Marker supports complex document elements: it properly formats tables, forms, equations, inline math, links, references, and code blocks. It also extracts and saves images and removes common artifacts (headers, footers, etc.). The tool is extensible with user-defined formatting logic, and it offers an optional hybrid mode that uses LLM assistance (`--use_llm`) to boost accuracy (e.g. merging tables across pages, handling complex math). Marker is flexible in execution, working on GPU, CPU, or Apple’s MPS, and provides high throughput in batch processing scenarios.

### markitdown by Microsoft
[![GitHub last commit](https://img.shields.io/github/last-commit/microsoft/markitdown?label=GitHub&logo=github)](https://github.com/microsoft/markitdown)
![GitHub License](https://img.shields.io/github/license/microsoft/markitdown)
<!--- 
License: MIT
Primary language: Python
-->

MarkItDown is a Python-based utility for converting various files to Markdown. *“It supports: PDF, PowerPoint, Word, Excel, images (with EXIF metadata and OCR), audio (with speech transcription), HTML, text formats (CSV, JSON, XML), ZIP archives, YouTube URLs, ... and more!”*. This breadth makes it useful for indexing and text analysis across diverse content types.

**Notable features:** The tool is currently in alpha (v0.0.2a1) and recently introduced a plugin-based architecture for extensibility. Despite its early stage, MarkItDown emphasizes broad format coverage, allowing conversion of Office documents, PDFs, images, and even audio to Markdown in a single workflow. It supports third-party plugins (disabled by default) which can be enabled via command-line (`--use-plugins`), and it provides a mechanism to discover plugins (search by `#markitdown-plugin`) for extending its capabilities.

### olmoOCR
[![GitHub last commit](https://img.shields.io/github/last-commit/allenai/olmocr?label=GitHub&logo=github)](https://github.com/allenai/olmocr)
![GitHub License](https://img.shields.io/github/license/allenai/olmocr)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](https://olmocr.allenai.org/)

<!--- 
License: Apache 2.0 
Primary language: Python
-->

olmOCR is a **toolkit for linearizing PDFs for LLM datasets/training**. In other words, it streamlines PDFs into text to facilitate large language model training on document data.

**Notable features:** This toolkit includes multiple components for high-quality PDF-to-text conversion. Key features outlined by the project include a prompting strategy for superior natural text extraction using ChatGPT-4, a **side-by-side evaluation toolkit** to compare different pipeline versions, language filtering and spam removal, and fine-tuning code for vision-language models. It supports large-scale processing, capable of converting millions of PDFs in parallel using a distributed pipeline (with integration of *sglang* for efficient GPU inference). Additionally, olmOCR provides a viewer to inspect extracted results in context (Dolma JSONL outputs with an HTML preview), facilitating easier validation of the conversion results.

## Benchmarks
### OmniDocBench by MinerU devs
[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/OmniDocBench?label=GitHub&logo=github)](https://github.com/opendatalab/OmniDocBench)
![GitHub License](https://img.shields.io/github/license/opendatalab/OmniDocBench)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

OmniDocBench is *“a benchmark for evaluating diverse document parsing in real-world scenarios”*. It establishes a comprehensive evaluation standard for document content extraction methods.

**Notable features:** OmniDocBench covers a wide variety of document types and layouts, comprising **981 PDF pages across 9 document types, 4 layout styles, and 3 languages**. It provides **rich annotations**: over 20k block-level elements (paragraphs, headings, tables, etc.) and 80k+ span-level elements (lines, formulas, etc.), including reading order and various attribute tags for pages, text, and tables. The dataset undergoes strict quality control (combining manual annotation, intelligent assistance, and expert review for high accuracy). OmniDocBench also comes with **evaluation code** for fair, end-to-end comparisons of document parsing methods. It supports multiple evaluation tasks (overall extraction, layout detection, table recognition, formula recognition, OCR text recognition) and standard metrics (Normalized Edit Distance, BLEU, METEOR, TEDS, COCO mAP/mAR, etc.) to benchmark performance across different aspects of document parsing.

### olmoOCR eval
[![GitHub last commit](https://img.shields.io/github/last-commit/allenai/olmocr?label=GitHub&logo=github)](https://github.com/allenai/olmocr)
![GitHub License](https://img.shields.io/github/license/allenai/olmocr)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

The olmOCR project provides an **evaluation toolkit** (`runeval.py`) for side-by-side comparison of PDF conversion pipeline outputs. This tool allows researchers to directly compare text extraction results from different pipeline versions against a gold-standard reference.

**Notable features:** This evaluation script generates a set of accuracy scores by comparing OCR outputs to ground-truth data. It is designed for easy **side-by-side evaluation**, producing metrics and even visual HTML reports for qualitative review of differences. By providing a standardized way to assess OCR pipeline performance, *olmOCR eval* helps validate improvements and ensures fair comparisons between different OCR approaches. (Note: *olmOCR eval* is part of the `allenai/olmocr` repository, not a standalone project.)

### Marker benchmarks
[![GitHub last commit](https://img.shields.io/github/last-commit/VikParuchuri/marker?label=GitHub&logo=github)](https://github.com/VikParuchuri/marker?tab=readme-ov-file#benchmarks)
![GitHub License](https://img.shields.io/github/license/VikParuchuri/marker)
<!--- 
License: GPL 3.0
Primary language: Python
-->

The Marker repository provides benchmark results comparing various PDF processing methods based on structured data extraction accuracy, processing time, and layout preservation.

**Notable features:** Benchmarks include comparison against open-source OCR solutions (Tesseract, EasyOCR, Adobe Extract API) and proprietary tools. Evaluates document type coverage, layout reconstruction accuracy, and OCR text correctness. Uses standardized metrics such as BLEU score, METEOR, and Levenshtein distance to assess output quality. Includes dataset-based testing to track improvements in text extraction accuracy over time.

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
  
  <!-- OCR Extracted Text: "Figure 2 - Sales Growth Trends" -->
  ```

#### Choosing the Right Option
- If the PDF contains machine-readable text and graphics are not required, **Ignore** is the most efficient choice.
- When working with scanned documents containing important visual text, **Replace with OCR** ensures that no information is lost.
- For preserving visual elements while keeping Markdown structured, **Extract to Folder with Markdown Link** is recommended.
- If both graphics and extracted text are needed, **Extract to Folder with Markdown Link and OCR Comment** provides the most comprehensive output.
