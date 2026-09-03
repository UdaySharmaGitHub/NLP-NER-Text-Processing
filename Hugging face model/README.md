# 🤗 Hugging Face Models

This folder contains Jupyter notebooks demonstrating various Hugging Face models for **NLP**, **NER**, and **Document AI** tasks.

## 📋 Model Notebooks

| Model | Hugging Face Hub | Use Case | Notebook | Documentation |
|-------|------------------|----------|----------|---------------|
| **d4data/biomedical-ner-all** | [🔗 Hub](https://huggingface.co/d4data/biomedical-ner-all) | Biomedical NER (107 entity types) | [Clinical Entity Extraction using Biomedical NER.ipynb](d4data%20----%20biomedical-ner-all/Clinical%20Entity%20Extraction%20using%20Biomedical%20NER.ipynb) | [📄 README](d4data%20----%20biomedical-ner-all/README.md) |
| **Davlan/xlm-roberta-large-ner-hrl** | [🔗 Hub](https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl) | Multilingual NER (10 languages) | [Text Extraction using OCR+Regex+NER for main data field.ipynb](Davlan%20----%20xlm-roberta-large-ner-hrl/Text%20Extraction%20using%20OCR%2BRegex%2BNER%20for%20main%20data%20field.ipynb) | [📄 README](Davlan%20----%20xlm-roberta-large-ner-hrl/README.md) |
| **impira/layoutlm-document-qa** | [🔗 Hub](https://huggingface.co/impira/layoutlm-document-qa) | Document QA (invoice, forms) | [Document QA Field Extraction using LayoutLM.ipynb](impira%20----%20layoutlm-document-qa/Document%20QA%20Field%20Extraction%20using%20LayoutLM.ipynb) | [📄 README](impira%20----%20layoutlm-document-qa/README.md) |
| **microsoft/layoutlmv3-base** | [🔗 Hub](https://huggingface.co/microsoft/layoutlmv3-base) | Layout-aware document understanding | [Layout-Aware Field Extraction using LayoutLMv3.ipynb](microsoft%20----%20layoutlmv3-base/Layout-Aware%20Field%20Extraction%20using%20LayoutLMv3.ipynb) | [📄 README](microsoft%20----%20layoutlmv3-base/README.md) |
| **microsoft/layoutxlm-base** | [🔗 Hub](https://huggingface.co/microsoft/layoutxlm-base) | Layout-aware embeddings for multilingual OCR output | [Layout-Aware Field Extraction using LayoutXLM (Multilingual).ipynb](Embedding%20Models/microsoft%20----%20layoutxlm-base%20Embedding%20models%20for%20OCR%20output/Layout-Aware%20Field%20Extraction%20using%20LayoutXLM%20%28Multilingual%29.ipynb) | [📄 README](Embedding%20Models/microsoft%20----%20layoutxlm-base%20Embedding%20models%20for%20OCR%20output/README.md) |
| **naver-clova-ix/donut-base-finetuned-docvqa** | [🔗 Hub](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) | OCR-free Document QA (sick notes, forms) | [Document Field Extraction using Donut DocVQA.ipynb](naver-clova-ix%20----%20donut-base-finetuned-docvqa/Document%20Field%20Extraction%20using%20Donut%20DocVQA.ipynb) | [📄 README](naver-clova-ix%20----%20donut-base-finetuned-docvqa/README.md) |

## 📚 Additional Resources

- [Model Comparison — Which model wins for zero-shot sick-note extraction?](MODEL_COMPARISON.md) — Side-by-side benchmark of all 6 models + spaCy, with a detailed explanation of why `impira/layoutlm-document-qa` is the best zero-shot choice
- [Hugging Face Models for Text & Data Extraction](Hugging%20Face%20Models%20for%20Text%20%26%20Data%20Extraction.md) — Comprehensive guide to free, open-source models for NER, OCR, and Document AI
