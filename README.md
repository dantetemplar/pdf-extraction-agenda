# PDF extraction pipelines and benchmarks agenda

> [!CAUTION]
> Part of text in this repo written by ChatGPT. Also, I haven't yet run all pipelines because of lack of compute power.

This repository provides an overview of selected **pipeline** and **benchmark** repositories related to PDF/OCR document processing. Each entry includes a brief description, latest commit date, contributor count, license, primary language, and notable features for quick reference.

## Table of contents

Did you know that GitHub supports table of
contents [by default](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/) 🤔

## Pipelines

### [MinerU](https://github.com/opendatalab/MinerU)
[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/MinerU?label=GitHub&logo=github)](https://github.com/opendatalab/MinerU)
![GitHub License](https://img.shields.io/github/license/opendatalab/MinerU)
[![Demo](https://img.shields.io/badge/DEMO-black?logo=huggingface)](https://huggingface.co/spaces/opendatalab/MinerU)
<!--- 
License: AGPL-3.0 
Primary language: Python
-->

MinerU is described as *“a high-quality tool for convert PDF to Markdown and JSON”*, serving as a one-stop open-source solution for high-quality data extraction from PDFs. It supports conversion of PDFs into machine-readable formats (Markdown, JSON) for easy data extraction.

**Notable features:** Provides a new API with composable processing stages and a `Dataset` class supporting multiple document formats (images, PDFs, Word, PPT, etc.). It includes advanced capabilities like automatic language identification for OCR (selecting the appropriate model from 84 supported languages). MinerU has a focus on performance and compatibility (optimized for ARM Linux and integrated with Huawei Ascend NPU for acceleration) and implements robust layout analysis (e.g. reading order with `layoutreader`) and table recognition modules to improve parsing accuracy.

### [Marker](https://github.com/VikParuchuri/marker)
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

### [markitdown by Microsoft](https://github.com/microsoft/markitdown)
[![GitHub last commit](https://img.shields.io/github/last-commit/microsoft/markitdown?label=GitHub&logo=github)](https://github.com/microsoft/markitdown)
![GitHub License](https://img.shields.io/github/license/microsoft/markitdown)
<!--- 
License: MIT
Primary language: Python
-->

MarkItDown is a Python-based utility for converting various files to Markdown. *“It supports: PDF, PowerPoint, Word, Excel, images (with EXIF metadata and OCR), audio (with speech transcription), HTML, text formats (CSV, JSON, XML), ZIP archives, YouTube URLs, ... and more!”*. This breadth makes it useful for indexing and text analysis across diverse content types.

**Notable features:** The tool is currently in alpha (v0.0.2a1) and recently introduced a plugin-based architecture for extensibility. Despite its early stage, MarkItDown emphasizes broad format coverage, allowing conversion of Office documents, PDFs, images, and even audio to Markdown in a single workflow. It supports third-party plugins (disabled by default) which can be enabled via command-line (`--use-plugins`), and it provides a mechanism to discover plugins (search by `#markitdown-plugin`) for extending its capabilities.

### [olmoOCR](https://github.com/allenai/olmocr)
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
### [OmniDocBench by MinerU devs](https://github.com/opendatalab/OmniDocBench)
[![GitHub last commit](https://img.shields.io/github/last-commit/opendatalab/OmniDocBench?label=GitHub&logo=github)](https://github.com/opendatalab/OmniDocBench)
![GitHub License](https://img.shields.io/github/license/opendatalab/OmniDocBench)
<!--- 
License: Apache 2.0 
Primary language: Python
-->

OmniDocBench is *“a benchmark for evaluating diverse document parsing in real-world scenarios”*. It establishes a comprehensive evaluation standard for document content extraction methods.

**Notable features:** OmniDocBench covers a wide variety of document types and layouts, comprising **981 PDF pages across 9 document types, 4 layout styles, and 3 languages**. It provides **rich annotations**: over 20k block-level elements (paragraphs, headings, tables, etc.) and 80k+ span-level elements (lines, formulas, etc.), including reading order and various attribute tags for pages, text, and tables. The dataset undergoes strict quality control (combining manual annotation, intelligent assistance, and expert review for high accuracy). OmniDocBench also comes with **evaluation code** for fair, end-to-end comparisons of document parsing methods. It supports multiple evaluation tasks (overall extraction, layout detection, table recognition, formula recognition, OCR text recognition) and standard metrics (Normalized Edit Distance, BLEU, METEOR, TEDS, COCO mAP/mAR, etc.) to benchmark performance across different aspects of document parsing.


**End-to-End Evaluation**

End-to-end evaluation assesses the model's accuracy in parsing PDF page content. The evaluation uses the model's Markdown output of the entire PDF page parsing results as the prediction.

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

The olmOCR project provides an **evaluation toolkit** (`runeval.py`) for side-by-side comparison of PDF conversion pipeline outputs. This tool allows researchers to directly compare text extraction results from different pipeline versions against a gold-standard reference. Also olmoOCR authors made some evalutions in their [technical report](https://olmocr.allenai.org/papers/olmocr.pdf).


> We then sampled 2,000 comparison pairs (same PDF, different tool). We asked 11 data researchers and
engineers at Ai2 to assess which output was the higher quality representation of the original PDF, focusing on
reading order, comprehensiveness of content and representation of structured information. The user interface
used is similar to that in Figure 5. Exact participant instructions are listed in Appendix B.
> 
> Figure 6 ELO ranking of olmOCR vs other popular PDF content extraction tools.

| OCR Tool  | Median Score | Interquartile Range |
|-----------|--------------|---------------------|
| olmOCR    | **~1850**    | **~1750 - 1900**    |
| MinerU    | ~1550        | ~1400 - 1650        |
| Marker    | ~1400        | ~1300 - 1500        |
| GOTOCOR   | ~1250        | ~1150 - 1350        |

<br/>

> Table 7: Pairwise Win/Loss Statistics Between Models

| Model Pair              | Wins   | Win Rate (%) |
|-------------------------|--------|-------------|
| olmOCR vs. Marker      | 49/31  | **61.3**    |
| olmOCR vs. GOTOCOR     | 41/29  | **58.6**    |
| olmOCR vs. MinerU      | 55/22  | **71.4**    |
| Marker vs. MinerU      | 53/26  | 67.1        |
| Marker vs. GOTOCOR     | 45/26  | 63.4        |
| GOTOCOR vs. MinerU     | 38/37  | 50.7        |
| **Total**              | **452** |             |




### [Marker benchmarks](https://github.com/VikParuchuri/marker?tab=readme-ov-file#benchmarks)
[![GitHub last commit](https://img.shields.io/github/last-commit/VikParuchuri/marker?label=GitHub&logo=github)](https://github.com/VikParuchuri/marker?tab=readme-ov-file#benchmarks)
![GitHub License](https://img.shields.io/github/license/VikParuchuri/marker)
<!--- 
License: GPL 3.0
Primary language: Python
-->

The Marker repository provides benchmark results comparing various PDF processing methods, scored based on a heuristic that aligns text with ground truth text segments, and an LLM as a judge scoring method.

| Method     | Avg Time | Heuristic Score | LLM Score |
|------------|----------|-----------------|-----------|
| marker     | 2.83837  | 95.6709         | 4.23916   |
| llamaparse | 23.348   | 84.2442         | 3.97619   |
| mathpix    | 6.36223  | 86.4281         | 4.15626   |
| docling    | 3.69949  | 86.7073         | 3.70429   |

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
