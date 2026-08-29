# 🤗 Hugging Face Models for Text & Data Extraction

A curated list of **free, open-source models** on the Hugging Face Hub that can be used for **Text Extraction, Named Entity Recognition (NER), Document AI, OCR, and structured data extraction** — as alternatives / complements to spaCy.

All models listed here are freely available on the [Hugging Face Model Hub](https://huggingface.co/models). Each entry includes the model ID, task, license, and a short use-case description.

> ⚠️ **License note**: Prefer permissive licenses (**Apache 2.0**, **MIT**, **BSD**) for commercial use. Models marked **CC BY-NC** or **CC BY-NC-SA** are for **non-commercial / research use only**.

---

## 📑 Table of Contents

1. [Named Entity Recognition (NER)](#1-named-entity-recognition-ner)
2. [Multilingual NER](#2-multilingual-ner)
3. [Domain-Specific NER (Biomedical, Legal, Finance)](#3-domain-specific-ner)
4. [Document AI & Layout Understanding](#4-document-ai--layout-understanding)
5. [OCR — Transformer-based Text Recognition](#5-ocr--transformer-based-text-recognition)
6. [Question Answering (Extractive QA)](#6-question-answering-extractive-qa)
7. [Table Detection & Extraction](#7-table-detection--extraction)
8. [Zero-Shot Classification](#8-zero-shot-classification)
9. [Summarization & Text Understanding](#9-summarization--text-understanding)
10. [Sentence Embeddings for Semantic Search](#10-sentence-embeddings-for-semantic-search)
11. [Recommended Model Combinations](#-recommended-model-combinations-by-use-case)

---

## 1. Named Entity Recognition (NER)

General-purpose English NER — extract PERSON, ORG, LOC, MISC, DATE, MONEY, etc.

| Model ID | License | Size | Description |
|---|---|---|---|
| [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) | MIT | 110M | BERT-base fine-tuned on CoNLL-2003. Fast, accurate baseline. |
| [`dslim/bert-large-NER`](https://huggingface.co/dslim/bert-large-NER) | MIT | 340M | BERT-large variant, higher accuracy. |
| [`Jean-Baptiste/roberta-large-ner-english`](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | MIT | 355M | RoBERTa-large fine-tuned on CoNLL-2003 + additional data. Strong performance. |
| [`elastic/distilbert-base-uncased-finetuned-conll03-english`](https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english) | Apache 2.0 | 66M | Lightweight, fast — great for production. |
| [`flair/ner-english-ontonotes-large`](https://huggingface.co/flair/ner-english-ontonotes-large) | MIT | ~430M | Trained on OntoNotes (18 entity types). Use via the Flair library. |

---

## 2. Multilingual NER

For processing text in many languages simultaneously.

| Model ID | License | Languages | Description |
|---|---|---|---|
| [`Davlan/xlm-roberta-large-ner-hrl`](https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl) | Apache 2.0 | 10 (Ar, De, En, Es, Fr, It, Lv, Nl, Pt, Zh) | **Already in this project.** XLM-R large for high-resource languages. |
| [`Davlan/bert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) | Apache 2.0 | 10 | Lighter mBERT variant of the above. |
| [`Davlan/distilbert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl) | Apache 2.0 | 10 | Distilled — fastest multilingual NER. |
| [`Babelscape/wikineural-multilingual-ner`](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | CC BY-NC-SA 4.0 | 9 | Trained on WikiNEuRal. **Non-commercial only.** |
| [`Jean-Baptiste/camembert-ner`](https://huggingface.co/Jean-Baptiste/camembert-ner) | MIT | French | Best-in-class French NER. |

---

## 3. Domain-Specific NER

### Biomedical / Clinical
| Model ID | License | Description |
|---|---|---|
| [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) | Apache 2.0 | Recognizes 107 biomedical entities (diseases, drugs, symptoms, etc.). |
| [`dmis-lab/biobert-v1.1`](https://huggingface.co/dmis-lab/biobert-v1.1) | Apache 2.0 | Pre-trained on PubMed abstracts — fine-tune for NER. |
| [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | MIT | Pre-trained on MIMIC-III clinical notes. |
| [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased) | Apache 2.0 | Scientific text; ideal for research paper extraction. |

### Financial / Legal
| Model ID | License | Description |
|---|---|---|
| [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) | Apache 2.0 | Financial sentiment + entity understanding. |
| [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased) | CC BY-SA 4.0 | Pre-trained on legal corpora. Fine-tune for legal NER. |

---

## 4. Document AI & Layout Understanding

Extract structured data from **PDFs, invoices, receipts, forms, and scanned documents**.

| Model ID | License | Task | Description |
|---|---|---|---|
| [`microsoft/layoutlm-base-uncased`](https://huggingface.co/microsoft/layoutlm-base-uncased) | MIT | Layout-aware LM | Combines text + 2D position. Excellent for forms. |
| [`microsoft/layoutlmv2-base-uncased`](https://huggingface.co/microsoft/layoutlmv2-base-uncased) | CC BY-NC-SA 4.0 | Layout + Vision | Adds image features. **Non-commercial.** |
| [`microsoft/layoutlmv3-base`](https://huggingface.co/microsoft/layoutlmv3-base) | CC BY-NC-SA 4.0 | Layout + Vision | State-of-the-art on FUNSD, CORD. **Non-commercial.** |
| [`impira/layoutlm-document-qa`](https://huggingface.co/impira/layoutlm-document-qa) | MIT | Document QA | Ask questions on documents ("What is the invoice total?"). Commercial-friendly. |
| [`naver-clova-ix/donut-base`](https://huggingface.co/naver-clova-ix/donut-base) | MIT | End-to-end Doc-AI | OCR-free — parses images directly to JSON. |
| [`naver-clova-ix/donut-base-finetuned-cord-v2`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) | MIT | Receipt parsing | Extracts items, prices, totals from receipts as JSON. |
| [`naver-clova-ix/donut-base-finetuned-docvqa`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) | MIT | Doc-VQA | Question-answering on document images. |
| [`facebook/nougat-base`](https://huggingface.co/facebook/nougat-base) | CC BY-NC 4.0 | PDF → Markdown | Converts scientific PDFs (with equations, tables) to Markdown. **Non-commercial.** |

---

## 5. OCR — Transformer-based Text Recognition

Alternatives to Tesseract / PaddleOCR for higher accuracy on modern documents.

| Model ID | License | Best For |
|---|---|---|
| [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) | MIT | Printed text (base size). |
| [`microsoft/trocr-large-printed`](https://huggingface.co/microsoft/trocr-large-printed) | MIT | Printed text (higher accuracy). |
| [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) | MIT | Handwritten text. |
| [`microsoft/trocr-large-handwritten`](https://huggingface.co/microsoft/trocr-large-handwritten) | MIT | Handwritten text (best accuracy). |
| [`microsoft/trocr-base-stage1`](https://huggingface.co/microsoft/trocr-base-stage1) | MIT | Pre-trained checkpoint for fine-tuning. |

> 💡 **Pipeline tip**: Combine TrOCR with a **text-detection model** (e.g., CRAFT, DBNet, or PaddleOCR's detector) — TrOCR itself only recognizes text within a cropped line region.

---

## 6. Question Answering (Extractive QA)

Turn any text into a **queryable knowledge source** — great for extracting fields from unstructured text.

| Model ID | License | Description |
|---|---|---|
| [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) | CC BY 4.0 | RoBERTa fine-tuned on SQuAD 2.0. Handles "no answer" cases. |
| [`deepset/tinyroberta-squad2`](https://huggingface.co/deepset/tinyroberta-squad2) | CC BY 4.0 | Distilled — 4x faster, minimal accuracy loss. |
| [`distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert-base-cased-distilled-squad) | Apache 2.0 | Fast, lightweight extractive QA. |
| [`deepset/xlm-roberta-large-squad2`](https://huggingface.co/deepset/xlm-roberta-large-squad2) | CC BY 4.0 | Multilingual extractive QA. |
| [`google/electra-base-discriminator`](https://huggingface.co/google/electra-base-discriminator) | Apache 2.0 | Efficient; fine-tune for QA. |

---

## 7. Table Detection & Extraction

Extract tables from images / PDFs and understand tabular data.

| Model ID | License | Task |
|---|---|---|
| [`microsoft/table-transformer-detection`](https://huggingface.co/microsoft/table-transformer-detection) | MIT | Detects table regions in document images. |
| [`microsoft/table-transformer-structure-recognition`](https://huggingface.co/microsoft/table-transformer-structure-recognition) | MIT | Recognizes rows, columns, cells within a table. |
| [`microsoft/table-transformer-structure-recognition-v1.1-all`](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all) | MIT | Updated version with better accuracy. |
| [`google/tapas-base-finetuned-wtq`](https://huggingface.co/google/tapas-base-finetuned-wtq) | Apache 2.0 | Question-answering over tables. |

---

## 8. Zero-Shot Classification

Classify text into **arbitrary user-defined categories** without training data.

| Model ID | License | Description |
|---|---|---|
| [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | MIT | Popular baseline for zero-shot classification. |
| [`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | MIT | State-of-the-art zero-shot. |
| [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | MIT | Multilingual zero-shot (100 languages). |

---

## 9. Summarization & Text Understanding

Reduce long documents to key facts before extraction.

| Model ID | License | Description |
|---|---|---|
| [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) | MIT | Abstractive summarization (CNN/DailyMail). |
| [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | Apache 2.0 | Distilled BART — faster summarization. |
| [`google/pegasus-xsum`](https://huggingface.co/google/pegasus-xsum) | Apache 2.0 | Concise, extreme summarization. |
| [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | Apache 2.0 | Instruction-tuned T5. Great general-purpose model. |
| [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) | Apache 2.0 | Larger variant, better reasoning. |

---

## 10. Sentence Embeddings for Semantic Search

Vector representations for **retrieval, deduplication, RAG pipelines**.

| Model ID | License | Description |
|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0 | Fast, small (80MB), high quality. |
| [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Apache 2.0 | Best general-purpose English embeddings. |
| [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache 2.0 | Multilingual (50+ languages). |
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | MIT | Top of MTEB leaderboard for its size. |

---

## 🎯 Recommended Model Combinations by Use Case

### 📄 Invoice / Receipt Extraction
```
Image → Donut (naver-clova-ix/donut-base-finetuned-cord-v2)
         └── outputs structured JSON directly
```

### 📋 Form Understanding (FUNSD-style)
```
Image → PaddleOCR (text detection)
      → LayoutLMv3 / LayoutLM (extract key-value pairs)
```

### 📚 Scientific PDF Parsing
```
PDF → pdf2image → Nougat (facebook/nougat-base)
      └── outputs Markdown with equations & tables preserved
```

### 🏥 Medical Records
```
OCR (TrOCR / PaddleOCR)
  → d4data/biomedical-ner-all (extract diseases, drugs, symptoms)
  → deepset/roberta-base-squad2 (answer specific queries)
```

### 🌐 Multilingual News / Web Content
```
Text → Davlan/xlm-roberta-large-ner-hrl (entities)
     → MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (topic classification)
```

### 🧾 General Document QA
```
Document image → impira/layoutlm-document-qa
                 └── ask any question: "Who is the vendor?"
```

### 🖋️ Handwritten Notes
```
Image → CRAFT / DBNet (text detection)
      → microsoft/trocr-large-handwritten (recognition)
      → dslim/bert-large-NER (entity extraction)
```

---

## 🚀 Quick Start Examples

### Load a NER model
```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="dslim/bert-large-NER",
    aggregation_strategy="simple",
)
print(ner("Barack Obama was born in Hawaii and worked at Microsoft."))
```

### Extract data from a document image (Donut)
```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)

image = Image.open("receipt.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
print(processor.batch_decode(outputs, skip_special_tokens=True))
```

### OCR with TrOCR
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

image = Image.open("line_of_text.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Document Question Answering
```python
from transformers import pipeline

doc_qa = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)
result = doc_qa(image="invoice.png", question="What is the total amount?")
print(result)
```

---

## 📊 Model Comparison — Choosing the Right One

| Need | Best Choice | Why |
|---|---|---|
| **Fast English NER on CPU** | `elastic/distilbert-...conll03` | Lightweight, fast inference |
| **Best English NER accuracy** | `Jean-Baptiste/roberta-large-ner-english` | RoBERTa-large, strong benchmarks |
| **Multilingual NER (commercial)** | `Davlan/xlm-roberta-large-ner-hrl` | Apache 2.0, covers 10 major languages |
| **Receipt/invoice parsing** | `donut-base-finetuned-cord-v2` | End-to-end, no OCR needed |
| **Question-answering on PDFs/images** | `impira/layoutlm-document-qa` | MIT license, production-ready |
| **Handwritten text OCR** | `microsoft/trocr-large-handwritten` | State-of-the-art on IAM benchmark |
| **Scientific paper → Markdown** | `facebook/nougat-base` | Preserves equations & tables |
| **Semantic search / retrieval** | `sentence-transformers/all-mpnet-base-v2` | Best MTEB-tier general embeddings |

---

## 🔗 Useful Links

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Task-based Model Search](https://huggingface.co/tasks)
- [`transformers` Documentation](https://huggingface.co/docs/transformers/index)
- [`datasets` Library](https://huggingface.co/docs/datasets/index)
- [Papers With Code — NER Leaderboard](https://paperswithcode.com/task/named-entity-recognition-ner)
- [MTEB Leaderboard (Embeddings)](https://huggingface.co/spaces/mteb/leaderboard)

---

## 📝 Notes

- All models can be downloaded with `AutoModel.from_pretrained("model-id")` and cached locally in `~/.cache/huggingface/`.
- For **GPU inference**, install PyTorch with CUDA support; most models fit on ≥8 GB VRAM.
- For **production**, consider converting to **ONNX** or using **`optimum`** for faster inference.
- Always verify a model's license on its Hub page before commercial use — licenses can change.

---

## 🎯 Quick Picks — Which Model for Which Job?

### 1. Extracting names / entities from plain text

Best pick depends on language and speed needs:

| Need | Model | Why |
|---|---|---|
| **Best English accuracy** | [`Jean-Baptiste/roberta-large-ner-english`](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | RoBERTa-large fine-tuned on CoNLL-2003 + extra data. Excellent on PERSON, ORG, LOC, MISC. MIT license. |
| **Fast / CPU-friendly English** | [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) or [`elastic/distilbert-base-uncased-finetuned-conll03-english`](https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english) | 66–110M params, near-instant inference. |
| **More entity types (DATE, MONEY, PERCENT, GPE…)** | [`flair/ner-english-ontonotes-large`](https://huggingface.co/flair/ner-english-ontonotes-large) | Trained on OntoNotes (18 types) — useful when you also want dates and money. |
| **Multilingual (already in your project)** | [`Davlan/xlm-roberta-large-ner-hrl`](https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl) | 10 languages, Apache 2.0, commercial-safe. |

**Recommendation:** start with `Jean-Baptiste/roberta-large-ner-english` for accuracy, or `flair/ner-english-ontonotes-large` if you also want dates / money / GPE out of the box.

---

### 2. Medical certificates (patient name, doctor's name, issue date, hospital, diagnosis…)

A medical certificate is a **document image / PDF with a layout** (headers, fields, signatures). One model alone isn't enough — you need a small pipeline. Recommended stack from the lists above:

```text
Certificate (image/PDF)
   │
   ├── (A) OCR-free field extraction
   │       impira/layoutlm-document-qa   ← ask "Patient name?", "Doctor name?", "Date of issue?"
   │
   └── (B) OCR + layout + NER (if you need structured fields at scale)
           PaddleOCR (already in your requirements)
             → microsoft/layoutlmv3-base   (layout-aware key/value extraction)
             → d4data/biomedical-ner-all   (extract diagnosis, drugs, symptoms)
```

**Recommendation:** start with `impira/layoutlm-document-qa` for zero-shot field extraction, then add `d4data/biomedical-ner-all` on top of the OCR text for medical entities. Move to a fine-tuned `naver-clova-ix/donut-base` once you have labelled samples.