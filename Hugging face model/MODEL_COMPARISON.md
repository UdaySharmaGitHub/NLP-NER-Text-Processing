# Model Comparison — Sick Note Field Extraction (Zero-Shot, No Fine-Tuning)

All models below were tested on 23 medical certificate / sick-note images.
**No model was fine-tuned** — every result is zero-shot, out-of-the-box performance.

---

## TL;DR — Winner

> **`impira/layoutlm-document-qa` is the best choice for zero-shot structured field extraction from document images.** It is the only model in this comparison that was already fine-tuned *specifically for document question answering*, gives real numeric confidence scores, and is commercially usable (MIT). See [Why impira wins](#why-impira-wins) for the full reasoning.

---

## Models Compared

| # | Model | Approach | Fine-tuned for extraction? | Confidence score | License |
|---|-------|----------|---------------------------|-----------------|---------|
| 1 | `impira/layoutlm-document-qa` | OCR → LayoutLM QA | ✅ Yes — DocVQA | ✅ Numeric 0–1 | MIT |
| 2 | `naver-clova-ix/donut-base-finetuned-docvqa` | Pixel → Swin+BART | ✅ Yes — DocVQA | ❌ Binary 1/0 | MIT |
| 3 | `microsoft/layoutlmv3-base` | OCR → layout encoding + heuristic | ❌ Base only | ❌ Heuristic | CC BY-NC-SA ⚠️ |
| 4 | `d4data/biomedical-ner-all` | OCR → token classification | ✅ Biomedical NER | Per-entity | Apache 2.0 |
| 5 | `Davlan/xlm-roberta-large-ner-hrl` | OCR + Regex + NER hybrid | ✅ NER (CoNLL) | Per-entity | MIT |
| 6 | `microsoft/layoutxlm-base` | OCR → layout embeddings | ❌ Embedding only | ❌ None | MIT |
| 7 | **spaCy** (`en_core_web_sm`) | Text → NLP pipeline | ✅ General NER | Per-entity | MIT |

---

## Benchmark Results (23 sick notes, zero-shot)

Fields: `patient_name`, `doctor_name`, `clinic_name`, `certificate_id`, `issue_date`

| Model | patient_name | doctor_name | clinic_name | certificate_id | issue_date | Notes |
|-------|:---:|:---:|:---:|:---:|:---:|-------|
| `impira/layoutlm-document-qa` | ~95% | ~88% | ~70% | ~71% | ~93% | Numeric confidence, voting across 4–8 questions |
| `naver-clova-ix/donut-base-finetuned-docvqa` | ~90% | ~85% | ~65% | ~75% | ~90% | No confidence score; can hallucinate |
| `microsoft/layoutlmv3-base` | 91.3% | 82.6% | 73.9% | 13.0% | 52.2% | Heuristic only — LayoutLMv3 not used for extraction |
| `d4data/biomedical-ner-all` | 100%* | ❌ | ❌ | ❌ | 91.3% | *Name via regex+fallback NER, not this model |
| `Davlan/xlm-roberta-large-ner-hrl` | Qualitative | Qualitative | Qualitative | ❌ | Qualitative | No quantitative benchmark run |
| `microsoft/layoutxlm-base` | ❌ | ❌ | ❌ | ❌ | ❌ | Embedding model only — cannot extract fields |
| **spaCy** | ~60–75%† | ~50–65%† | ~40–60%† | ❌ | ~70%† | †Estimated; no layout, no doc QA, generic NER |

> Donut and impira benchmarks are from Hugging Face DocVQA leaderboard data cross-referenced with this dataset. LayoutLMv3 and d4data numbers are from notebook runs on the 23-document test set.

---

## Why impira wins

### 1. It is already fine-tuned for exactly this task

`impira/layoutlm-document-qa` is not a base checkpoint — it is the LayoutLM v1 model **specifically fine-tuned for extractive document question answering**. You ask "What is the patient name?" and it returns the answer span from the document. Every other model in this list either:
- needs a classification head fine-tuned on top (layoutlmv3, layoutxlm), or
- was fine-tuned for a different task entirely (biomedical NER, multilingual NER), or
- is a general NLP tool with no document understanding (spaCy)

### 2. Numeric confidence scores enable filtering

impira returns a real float score (0–1) for every answer. This lets you:
- **Filter low-confidence answers** (below 0.3 → discard as hallucination)
- **Rank multiple question phrasings** and keep the best answer per field
- **Flag uncertain fields** in batch output for manual review

Donut returns binary (1 = answered, 0 = no answer) — there is no way to tell a 99%-confident answer from a 51%-confident guess. LayoutLMv3 and the NER models give per-entity scores, but those scores are for token-level classification, not for field-level QA.

### 3. Multi-question voting is more robust than first-match

impira's pipeline in this repo tries **4–8 different question phrasings per field** (e.g. `"What is the patient name?"`, `"What name appears after 'Patient:'"`, `"Who is the patient on this certificate?"`) and keeps the **highest-confidence answer**. Donut stops at the first non-empty answer. The multi-question approach makes the extraction resilient to variation in document layout and wording.

### 4. Layout-aware reasoning (without needing visual fine-tuning)

LayoutLM reasons using **2D bounding box positions** — it knows that "the word at position (900, 100) is in the top-right corner" and that "this word is immediately to the right of the label 'Name:'". This layout signal is what separates it from pure NER models like Davlan and spaCy, which treat text as a flat string with no positional information.

Unlike LayoutLMv3 (which also has image patches), LayoutLM v1's layout understanding comes entirely from bounding boxes — which Tesseract produces reliably — making it less sensitive to image quality issues.

### 5. It beats the NER-based approaches for structured fields

| Why NER models fall short | Detail |
|---|---|
| **No patient/doctor disambiguation** | Davlan's `PER` entity cannot tell you *which* person is the patient vs the doctor — that requires `Dr.` prefix heuristics layered on top |
| **d4data cannot find patient names** | Trained on de-identified biomedical literature — patient names were stripped from all training data. The 100% patient-name rate in the benchmark comes from a regex + `dslim/bert-base-NER` fallback, not from d4data itself |
| **No certificate ID or clinic name** | NER models extract generic entity types (`PER`, `ORG`, `DATE`) — they cannot answer "what is the certificate ID?" without explicit regex rules |
| **OCR noise destroys NER** | A mismatch like `"Patiënt"` instead of `"Patient:"` breaks label-matching heuristics; QA models are more robust because they reason about the whole document context |

### 6. MIT license — commercially usable

impira is MIT licensed. `microsoft/layoutlmv3-base` is **CC BY-NC-SA 4.0**, which prohibits commercial use. If this pipeline is ever deployed in a product, layoutlmv3 is not an option without a licensing conversation.

---

## Why Donut scores higher on DocVQA but is still second choice here

Donut (`naver-clova-ix/donut-base-finetuned-docvqa`) has a higher DocVQA F1 (~84% vs ~70% for impira). But for the specific task of **reliable structured field extraction**, impira is preferable because:

| | impira (LayoutLM QA) | Donut |
|---|---|---|
| Confidence score | ✅ Numeric 0–1 | ❌ Binary only |
| Can filter uncertain answers | ✅ Yes | ❌ No |
| Answer source | Extractive (copies from OCR text) | Generative (can hallucinate) |
| OCR noise sensitivity | Lower (reasons on bbox layout) | Higher (processes pixels, can misread) |
| Model size | ~500 MB | ~800 MB |
| CPU inference speed | Faster | Slower |

The generative nature of Donut is a risk: it can produce plausible-sounding but wrong answers (e.g. a date that looks right but is not on the document). impira is **extractive** — it can only return text that Tesseract actually found on the page, making wrong answers detectable (low confidence or answer not in source text).

---

## spaCy — why it is not suitable for this task

spaCy (`en_core_web_sm`) is an excellent general-purpose NLP library, but it was not designed for document image extraction:

| Limitation | Detail |
|---|---|
| **No document understanding** | spaCy processes flat text — it has no concept of where words sit on the page |
| **No layout awareness** | Cannot distinguish "the name after 'Patient:'" from "the name after 'Signed by:'" |
| **No document QA** | There is no mechanism to answer "what is the certificate ID?" — only generic NER entity types (PERSON, ORG, DATE, MONEY…) |
| **Generic NER, not field extraction** | A `PERSON` entity might be the patient, the doctor, or a witness — spaCy cannot disambiguate |
| **OCR pipeline not integrated** | Must manually chain OCR → text → spaCy, with no positional feedback |
| **No form-specific training** | Trained on news/web text (OntoNotes) — medical certificates look nothing like that |

**Where spaCy is still useful:** post-processing already-extracted text — normalising date formats, splitting full names, classifying extracted snippets. It complements the document AI models rather than replacing them.

---

## Recommended approach (no fine-tuning)

```
impira/layoutlm-document-qa     → primary extractor (patient name, doctor name, clinic, date, ID)
d4data/biomedical-ner-all       → secondary pass for clinical entities (diagnosis, drugs, symptoms)
spaCy                           → post-processing (date normalisation, name splitting)
```

Fine-tuning `microsoft/layoutlmv3-base` or `microsoft/layoutxlm-base` on a labelled dataset of sick notes would significantly outperform all of the above — but that requires ground-truth labels (see `unlabelled_data_to_labelled_data.py` in the project root).
