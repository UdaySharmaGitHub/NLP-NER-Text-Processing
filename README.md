# NLP | NER | Text-Processing | Data Extraction

A Python project for Natural Language Processing (NLP), Named Entity Recognition (NER), and Optical Character Recognition (OCR) using state-of-the-art transformer models. The focus is on extracting structured fields from medical certificates and sick notes, with an exploration and benchmarking of multiple model architectures.

---

## 🎯 Project Overview

This project investigates different approaches to intelligent data extraction from document images and PDFs — comparing OCR-based NER pipelines, layout-aware transformers, and OCR-free vision models.

**Core tasks:**
- **NER**: Extract named entities (persons, dates, diagnoses, medications) from text
- **Document QA**: Answer field-extraction questions about document images
- **OCR**: Convert scanned documents and PDFs to searchable text
- **Layout Understanding**: Reason about where text sits on the page (bounding boxes)

---

## 🤗 Implemented Models

All implementations live in [`Hugging face model/`](Hugging%20face%20model/) as self-contained Jupyter notebooks.

| Model | Task | Approach | Notebook | Docs |
|-------|------|----------|----------|------|
| **d4data/biomedical-ner-all** | Biomedical NER (107 entity types) | OCR → token classification | [📓 Notebook](Hugging%20face%20model/d4data%20----%20biomedical-ner-all/Clinical%20Entity%20Extraction%20using%20Biomedical%20NER.ipynb) | [📄 README](Hugging%20face%20model/d4data%20----%20biomedical-ner-all/README.md) |
| **Davlan/xlm-roberta-large-ner-hrl** | Multilingual NER (10 languages) | OCR + Regex + NER hybrid | [📓 Notebook](Hugging%20face%20model/Davlan%20----%20xlm-roberta-large-ner-hrl/Text%20Extraction%20using%20OCR%2BRegex%2BNER%20for%20main%20data%20field.ipynb) | [📄 README](Hugging%20face%20model/Davlan%20----%20xlm-roberta-large-ner-hrl/README.md) |
| **impira/layoutlm-document-qa** | Document QA — layout-aware | OCR + bounding boxes → QA | [📓 Notebook](Hugging%20face%20model/impira%20----%20layoutlm-document-qa/Document%20QA%20Field%20Extraction%20using%20LayoutLM.ipynb) | [📄 README](Hugging%20face%20model/impira%20----%20layoutlm-document-qa/README.md) |
| **microsoft/layoutlmv3-base** | Document field extraction | OCR + layout encoding heuristic | [📓 Notebook](Hugging%20face%20model/microsoft%20----%20layoutlmv3-base/Layout-Aware%20Field%20Extraction%20using%20LayoutLMv3.ipynb) | [📄 README](Hugging%20face%20model/microsoft%20----%20layoutlmv3-base/README.md) |
| **microsoft/layoutxlm-base** | Layout-aware embeddings (multilingual) | OCR + SentencePiece + spatial coords | [📓 Notebook](Hugging%20face%20model/Embedding%20Models/microsoft%20----%20layoutxlm-base%20Embedding%20models%20for%20OCR%20output/Layout-Aware%20Field%20Extraction%20using%20LayoutXLM%20%28Multilingual%29.ipynb) | [📄 README](Hugging%20face%20model/Embedding%20Models/microsoft%20----%20layoutxlm-base%20Embedding%20models%20for%20OCR%20output/README.md) |
| **naver-clova-ix/donut-base-finetuned-docvqa** | OCR-free Document QA | Raw pixels → Swin encoder → BART decoder | [📓 Notebook](Hugging%20face%20model/naver-clova-ix%20----%20donut-base-finetuned-docvqa/Document%20Field%20Extraction%20using%20Donut%20DocVQA.ipynb) | [📄 README](Hugging%20face%20model/naver-clova-ix%20----%20donut-base-finetuned-docvqa/README.md) |

See also: [Hugging Face Models for Text & Data Extraction.md](Hugging%20face%20model/Hugging%20Face%20Models%20for%20Text%20%26%20Data%20Extraction.md) — a broader guide to free open-source models for NER, OCR, and Document AI.

---

## 📁 Project Structure

```
NLP-NER-Text-Processing/
├── README.md                                    # This file
├── LICENSE
├── requirements.in                              # Python dependencies
├── requirements.txt                             # Locked dependencies (auto-generated)
├── unlabelled_data_to_labelled_data.py          # Batch OCR → labelled CSV for ground-truth eval
├── test_data/
│   └── sick_notes/                              # Sample document images used for testing
├── Hugging face model/
│   ├── README.md                                # Model index
│   ├── Hugging Face Models for Text & Data Extraction.md
│   ├── d4data ---- biomedical-ner-all/
│   ├── Davlan ---- xlm-roberta-large-ner-hrl/
│   ├── impira ---- layoutlm-document-qa/
│   ├── microsoft ---- layoutlmv3-base/
│   ├── naver-clova-ix ---- donut-base-finetuned-docvqa/
│   └── Embedding Models/
│       └── microsoft ---- layoutxlm-base Embedding models for OCR output/
└── SpaCy/                                       # spaCy experiments (in progress)
```

---

## 🔬 Model Approach Comparison

| Approach | OCR Required | Confidence Score | DocVQA F1 | Best For |
|---|---|---|---|---|
| OCR + Regex + NER (`Davlan`) | ✅ Tesseract | Per-entity score | — | Multilingual documents |
| Biomedical NER (`d4data`) | ✅ Tesseract | Per-entity score | — | Clinical/medical text |
| Layout QA v1 (`impira`) | ✅ Tesseract | ✅ Numeric 0–1 | ~70% | Forms with clear labels |
| Layout encoding (`layoutlmv3`) | ✅ Tesseract | Proximity heuristic | — | Layout-dependent fields |
| Layout embeddings (`layoutxlm`) | ✅ Tesseract | Token embeddings | — | Multilingual layout analysis |
| OCR-free QA (`donut`) | ❌ None | ❌ Binary | ~84% | Clean scanned documents |

---

## 📦 Installation

**Prerequisites:** Python 3.8+, macOS / Linux / Windows

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Tesseract OCR binary (required by most notebooks)
brew install tesseract            # macOS
sudo apt-get install tesseract-ocr   # Ubuntu/Debian
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 🚀 Quick Start

### Ground-truth label generation

`unlabelled_data_to_labelled_data.py` scans `test_data/sick_notes/`, runs Tesseract OCR on each file, and writes a CSV with blank label columns for manual annotation:

```bash
python unlabelled_data_to_labelled_data.py
# → labelled_data.csv with columns: patient_name, issued_date, doctor_name, diagnosis, issuing_organization
```

### Run a model notebook

```bash
jupyter notebook
# Open Hugging face model/ and choose a notebook
```

### NER quick example

```python
from transformers import pipeline

ner = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="simple")
results = ner("John Smith was treated at City Hospital on 12 May 2024.")
for ent in results:
    print(ent["word"], "→", ent["entity_group"], f"({ent['score']:.2f})")
```

### Document QA quick example

```python
from transformers import pipeline
from PIL import Image

qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
image = Image.open("test_data/sick_notes/example.jpg")
print(qa(image, "What is the patient's name?"))
# → [{"answer": "John Smith", "score": 0.95, ...}]
```

---

## 🗺️ Roadmap

### NER & Text Processing
- [x] XLM-RoBERTa multilingual NER (`Davlan/xlm-roberta-large-ner-hrl`)
- [x] Biomedical NER 107 entities (`d4data/biomedical-ner-all`)
- [ ] spaCy pipeline experiments
- [ ] Fine-tuned NER on sick-note ground-truth labels

### Document AI & Layout Understanding
- [x] LayoutLM v1 Document QA (`impira/layoutlm-document-qa`)
- [x] LayoutLMv3 layout encoding (`microsoft/layoutlmv3-base`)
- [x] LayoutXLM multilingual embeddings (`microsoft/layoutxlm-base`)
- [x] Donut OCR-free DocVQA (`naver-clova-ix/donut-base-finetuned-docvqa`)
- [ ] Nougat (PDF → Markdown)
- [ ] Table detection and extraction

### Benchmarking
- [x] Ground-truth label generation script (`unlabelled_data_to_labelled_data.py`)
- [ ] Populate benchmark tables in each model README
- [ ] Cross-model comparison report

---

## 🔧 Dependencies

### Core
| Library | Purpose |
|---------|---------|
| `transformers` | Hugging Face transformer models |
| `torch` | Deep learning framework |
| `spacy` | Industrial NLP |
| `pandas` / `numpy` | Data manipulation |

### OCR & Vision
| Library | Purpose |
|---------|---------|
| `pytesseract` | OCR via Tesseract |
| `pillow` | Image processing |
| `opencv-python` | Image preprocessing (CLAHE, deskew) |
| `PyMuPDF` | PDF rendering |
| `paddleocr` | Multilingual OCR (planned) |

See [requirements.in](requirements.in) for the full list.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🔗 References

- [Hugging Face Model Hub](https://huggingface.co/models)
- [LayoutLM paper](https://arxiv.org/abs/1912.13318) — Xu et al., 2019
- [Donut paper](https://arxiv.org/abs/2111.15664) — Kim et al., 2021
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) — Conneau et al., 2019
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
