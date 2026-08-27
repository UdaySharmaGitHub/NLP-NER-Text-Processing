# NLP | NER | Text-Processing | Data Extraction

A comprehensive Python project for Natural Language Processing (NLP), Named Entity Recognition (NER), and Optical Character Recognition (OCR) using state-of-the-art transformer models and machine learning techniques.

## 🎯 Project Overview

This is a comprehensive exploration and implementation project that investigates **all different types of state-of-the-art models available on the internet** for advanced text processing and intelligent data extraction from various document formats.

### Core Functionalities:
- **NER (Named Entity Recognition)**: Extract entities (persons, locations, organizations, etc.) from text
- **NLP (Natural Language Processing)**: Text classification, sentiment analysis, and text understanding
- **OCR (Optical Character Recognition)**: Extract text from images using advanced OCR models
- **Document Processing**: Extract structured data from PDFs, scanned documents, and complex layouts
- **Multimodal Analysis**: Process images, PDFs, and other document types with vision-language models

### Models & Technologies to Explore:
This repository will feature implementations and comparisons of:
- **Transformer Models**: BERT, RoBERTa, XLM-RoBERTa, DistilBERT, ELECTRA
- **Vision Models**: ViT (Vision Transformer), CLIP, LayoutLM, DocTR
- **Language Models**: GPT-based models, T5, BART, Seq2Seq architectures
- **OCR Solutions**: Tesseract, PaddleOCR, EasyOCR, Keras-OCR
- **Document AI**: Donut (Document Understanding Transformer), Nougat (PDF to Markdown)
- **Multimodal Models**: BLIP, LLaVA for image-text understanding
- **Specialized Models**: Domain-specific models for legal docs, invoices, forms, tables

## ✨ Features

- 🤗 **Hugging Face Transformers**: Pre-trained transformer models (XLM-RoBERTa, BERT, etc.)
- 🔤 **spaCy Integration**: Industrial-strength NLP processing
- 👁️ **Multiple OCR Engines**: pytesseract, PaddleOCR for multilingual support
- 📊 **Data Science Stack**: NumPy, Pandas, Scikit-learn for data analysis
- 📈 **Visualization**: Matplotlib, Seaborn for insights
- 🧪 **Jupyter Support**: Interactive notebooks for experimentation

## 📄 Supported Document & Input Formats

### Document Types
- **Text Documents**: .txt, .docx, .doc
- **Images**: .png, .jpg, .jpeg, .gif, .bmp, .tiff
- **PDFs**: Single-page and multi-page PDFs, scanned documents
- **Structured Documents**: Forms, invoices, receipts, tables
- **Web Content**: HTML, URLs with text extraction

### Data Extraction Capabilities
- 🔤 **Text Extraction**: Convert images/PDFs to searchable text
- 📋 **Table Recognition**: Extract structured data from tables
- 📑 **Layout Analysis**: Understand document structure and sections
- 🏷️ **Entity Extraction**: Identify key information (names, dates, amounts, etc.)
- 🔍 **Form Field Detection**: Extract values from forms and templates

## 🔬 Model Exploration & Benchmarking

This repository is designed to explore and benchmark multiple model architectures:

| Model Category | Examples | Use Case |
|---|---|---|
| **NER Models** | BERT-NER, XLM-RoBERTa, Flair | Extract named entities from text |
| **OCR Models** | Tesseract, PaddleOCR, EasyOCR, Keras-OCR | Text recognition from images |
| **Document AI** | LayoutLM, Donut, Nougat | Understand document structure and content |
| **Vision Transformers** | ViT, CLIP, DINOv2 | Image understanding and analysis |
| **LLMs** | GPT, LLaMA, Mistral | Advanced text understanding and generation |
| **Multimodal** | BLIP, LLaVA, GPT-4V | Combine image and text understanding |



## 📦 Installation

### Prerequisites
- Python 3.8+
- macOS / Linux / Windows
- pip or conda

### Setup Instructions

1. **Clone the repository**
   ```bash
   cd /Uday Sharma/Desktop/My\ Workspace/NLP-NER-Text-Processing
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Option A: Install from requirements.in
   pip install -r requirement.in

   # Option B: Install locked requirements (recommended)
   pip install pip-tools
   pip-compile requirement.in
   pip install -r requirements.txt
   ```

4. **Download spaCy model (optional)**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 📁 Project Structure

```
NLP-NER-Text-Processing/
├── README.md                          # This file
├── LICENSE                            # Project license
├── requirement.in                     # Python dependencies
├── requirements.txt                   # Locked dependencies (auto-generated)
├── .gitignore                         # Git ignore rules
├── .venv/                             # Virtual environment
├── Hugging face model/                # Pre-trained transformer models
│   └── Davlan/
│       └── xlm-roberta-large-ner-hrl/  # XLM-RoBERTa NER model
├── SpaCy/                             # spaCy models directory
├── models/                            # Model exploration directory
│   ├── ner_models/                    # NER model implementations
│   ├── ocr_models/                    # OCR model implementations
│   ├── document_ai/                   # Document AI models (LayoutLM, Donut, etc.)
│   └── vision_models/                 # Vision transformer models
├── data/                              # Dataset directory
│   ├── raw/                           # Raw text/images/PDFs
│   ├── processed/                     # Processed outputs
│   ├── sample_pdfs/                   # Sample PDF documents
│   └── sample_images/                 # Sample images for OCR
├── notebooks/                         # Jupyter notebooks for experimentation
│   ├── model_exploration/             # Notebooks comparing different models
│   ├── ocr_comparison.ipynb           # OCR models benchmarking
│   ├── ner_comparison.ipynb           # NER models comparison
│   └── document_extraction.ipynb      # Document processing examples
├── src/                               # Source code modules
│   ├── ner/                           # NER pipeline & model implementations
│   ├── nlp/                           # NLP utilities
│   ├── ocr/                           # OCR processing module
│   ├── document_ai/                   # Document AI utilities
│   └── utils/                         # Helper functions
├── benchmarks/                        # Model benchmarking & results
│   ├── results/                       # Benchmark results
│   └── comparison_reports/            # Model comparison reports
└── output/                            # Results and outputs
```

## 🎓 Research Goals

This repository aims to:
- ✅ Explore and compare **10+ different NER models** across multiple languages
- ✅ Benchmark **5+ OCR engines** on accuracy and speed
- ✅ Implement **Document AI solutions** for PDF extraction and form understanding
- ✅ Create a **unified API** to work with different models interchangeably
- ✅ Provide comprehensive **comparison benchmarks** and recommendations
- ✅ Document best practices for **production deployments**
- ✅ Build practical examples for **real-world use cases**

## 🗺️ Model Exploration Roadmap

### Phase 1: NER & Text Processing Models
- [ ] BERT-based NER models
- [ ] RoBERTa and XLM-RoBERTa variants
- [ ] Flair NER framework
- [ ] BioBERT for biomedical NER
- [ ] DistilBERT (lightweight alternative)

### Phase 2: OCR & Character Recognition
- [ ] Tesseract + OpenCV pipeline
- [ ] PaddleOCR (multilingual)
- [ ] EasyOCR comparison
- [ ] Keras-OCR implementation
- [ ] Handwriting recognition models

### Phase 3: Document AI & Layout Understanding
- [ ] LayoutLM for document understanding
- [ ] Donut (Document Understanding Transformer)
- [ ] Nougat (PDF to Markdown)
- [ ] Table detection and extraction
- [ ] Form field recognition

### Phase 4: Vision & Multimodal Models
- [ ] Vision Transformers (ViT)
- [ ] CLIP for image-text matching
- [ ] BLIP for image captioning
- [ ] LLaVA for multimodal understanding
- [ ] GPT-4V API integration

### Phase 5: Specialized & Domain Models
- [ ] Invoice/receipt recognition
- [ ] Contract document analysis
- [ ] Medical document processing
- [ ] Financial document extraction
- [ ] Custom fine-tuned models



## 🚀 Quick Start

### 1. Named Entity Recognition (NER)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model
model_name = "Davlan/xlm-roberta-large-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Process text
text = "John Smith works at Google in New York."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### 2. Text Processing with spaCy

```python
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Extract entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
```

### 3. OCR from Images

```python
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Extract text from image
result = ocr.ocr("path/to/image.jpg", cls=True)
for line in result:
    for word_info in line:
        print(word_info[1][0])  # Print recognized text
```

### 4. Document Data Extraction from PDFs

```python
from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image

# Document QA model for form understanding
processor = AutoProcessor.from_pretrained("microsoft/layoutlm-base-uncased")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")

# Process document images
image = Image.open("document.png")
question = "What is the total amount?"
encoding = processor(image, question, return_tensors="pt")
outputs = model(**encoding)
```

## 💼 Real-World Use Cases

This repository explores solutions for:

1. **Invoice Processing**: Extract vendor info, amounts, dates from invoices
2. **Resume Parsing**: Extract education, experience, skills from documents
3. **Form Automation**: Process filled forms and extract structured data
4. **Contract Analysis**: Identify key clauses and entities in contracts
5. **Receipt Recognition**: Extract items and totals from receipts
6. **Medical Records**: Extract patient info and medical codes from documents
7. **Legal Document Review**: Identify relevant sections and extract citations
8. **Web Scraping & OCR**: Convert web content and images to structured data


## 📊 Data Format

### Input
- **Text**: Plain text files (.txt), CSV files with text columns
- **Images**: PNG, JPG, JPEG formats for OCR

### Output
- **NER Results**: JSON with entity types and positions
- **Text Analysis**: CSV with classifications and scores
- **OCR Results**: Extracted text and coordinates

## 🔧 Dependencies

### Core Libraries
| Library | Purpose |
|---------|---------|
| `transformers` | Hugging Face transformer models |
| `torch` | Deep learning framework |
| `spacy` | Industrial NLP |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |

### OCR & Vision
| Library | Purpose |
|---------|---------|
| `pytesseract` | OCR via Tesseract |
| `paddleocr` | Chinese/multilingual OCR |
| `pillow` | Image processing |

### Advanced Models & Document AI
| Library | Purpose |
|---------|---------|
| `layoutlm` | Layout-aware language models |
| `detectron2` | Object detection (Tesseract alternative) |
| `timm` | Vision transformers |
| `pdf2image` | PDF to image conversion |

See [requirement.in](requirement.in) for the complete dependency list.

## 📝 Usage Examples

### Running Notebooks
```bash
jupyter notebook
# Open notebooks/ folder and select desired notebook
```

### Command Line
```bash
python src/ner/extract_entities.py --input data/raw/text.txt --output output/entities.json
```

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Commit changes (`git commit -m 'Add feature'`)
3. Push to branch (`git push origin feature/your-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References & Resources

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Complete transformer library
- [spaCy Documentation](https://spacy.io/) - Industrial NLP toolkit
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library

### Model Collections
- [Hugging Face Model Hub](https://huggingface.co/models) - 100K+ models
- [Paperspace Models](https://models.paperspace.com/) - Community models
- [Model Zoo](https://modelzoo.co/) - Deep learning model collection

### OCR & Document AI
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - General-purpose OCR
- [Donut Paper](https://arxiv.org/abs/2111.15664) - Document understanding transformer
- [LayoutLM](https://arxiv.org/abs/1912.13318) - Multimodal pre-training for layout

### Research Papers
- "Attention is All You Need" - Transformers (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" - (Devlin et al., 2018)
- "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" - (Xu et al., 2019)


## ❓ FAQs

**Q: How do I use a different transformer model?**  
A: Replace the model name in code with any model from [Hugging Face Hub](https://huggingface.co/models).

**Q: Can I use GPU acceleration?**  
A: Yes! PyTorch automatically uses GPU if available. Install CUDA-compatible PyTorch for best performance.

**Q: How do I add my custom data?**  
A: Place data in `data/raw/` and create preprocessing scripts in `src/utils/`.

## 📞 Support

For issues and questions, please open an issue on the repository or contact the maintainers.

---

**Happy NLP Processing! 🎉**