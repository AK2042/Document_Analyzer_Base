# Document Analyzer with LLaMA, Groq, Google Generative AI Embeddings & OCR

A document analysis app that allows you to upload PDF files, extract and summarize their content, and interactively ask questions about them. It leverages the Meta LLaMA 4 model, Google Generative AI embeddings for vector search, and uses OCR only when necessary for scanned or image-based PDFs.

---

## Features

- Upload PDFs and extract text directly or via OCR if needed
- Summarize documents automatically
- Interactive Q&A based on the document content
- Vector search powered by Google Generative AI embeddings and FAISS
- Uses OCR (Tesseract) when text extraction fails or is insufficient
- Stateless processing (no local file storage)

---

## Getting Started

### Prerequisites

- API keys for:
  - **Groq API** (`GROQ_API_KEY`)
  - **Google Cloud API** (`GOOGLE_API_KEY`)

### Environment Variables

Create a `.env` file in the root folder with:

```bash
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
````

---

## Installation (Local Development)

1. Clone this repository:

   ```bash
   git clone https://github.com/AK2042/Document_Analyzer_Base.git
   cd Document_Analyzer_Base
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies (for OCR and PDF processing):

   On Debian/Ubuntu:

   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr poppler-utils libsm6 libxext6 libxrender-dev
   ```

   On Arch Linux:

   ```bash
   sudo pacman -S tesseract poppler tesseract-data
   ```

4. Run the app locally:

   ```bash
   python app/Document_Analyzer.py
   ```

---

## Usage

* Upload a PDF file via the web UI.
* The app extracts text, uses OCR if necessary, and generates a summary.
* Ask questions related to the document interactively.
* Use the **Reset** button to clear the session and upload a new file.

---

## Project Structure

```
document-analyzer/
├── app/
│   ├── Document_Analyzer.py       # Main application logic
│   ├── ocr.py                    # OCR helper functions
├──.env
├── requirements.txt             # Python dependencies
└── README.md                   # This file(You are here)
```

---

## Notes

* OCR is only triggered if the initial PDF text extraction fails or returns insufficient text.
* Tesseract must be installed with the correct language data files.
* No files are permanently stored on disk; processing is done in-memory.
* Make sure your API keys have appropriate permissions and quota.
* The tesseract has been set to locate the language data in the `tessdata` folder.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

* [Meta LLaMA](https://github.com/facebookresearch/llama)
* [Groq API](https://www.groq.com/)
* [Google Generative AI Embeddings](https://developers.generativeai.google/)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [Gradio](https://gradio.app/)

---

Feel free to submit issues or pull requests for improvements!

```
