# Document Analyzer with OCR

A document analysis app that allows you to upload PDF files, extract and summarize their content, and interactively ask questions about them. It leverages the Meta LLaMA 4 model, Google Generative AI embeddings for vector search, and uses OCR only when necessary for scanned or image-based PDFs.

---

### Live Demo

[Hugging Face Space](https://ak2042-document-analyzer.hf.space/)

---

## Features

- Upload PDFs and extract text directly or via OCR
- Summarize documents automatically
- Interactive Q&A on your document content
- Vector search with Google Generative AI embeddings + FAISS
- OCR fallback using Tesseract when PDFs are image-based
- Stateless processing (no file storage on disk)

---

## Getting Started

### Prerequisites

- API keys for:
  - **Groq API** (`GROQ_API_KEY`)
  - **Google Cloud API** (`GOOGLE_API_KEY`)

### Environment Variables

Create a `.env` file in the root folder:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
````

---

## Local Development (Without Docker)

1. Clone the repository:

```bash
git clone https://github.com/AK2042/Document_Analyzer_Base.git
cd Document_Analyzer_Base
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install system dependencies:

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libsm6 libxext6 libxrender-dev
```

4. Run the app:

```bash
python app/Document_Analyzer.py
```

---

## Running via Docker

You can also run this app using Docker (great for Hugging Face Spaces or isolated environments).

### Build the Docker Image

```bash
docker build -t document-analyzer .
```

### Run the Container

```bash
docker run --env-file .env -p 7860:7860 document-analyzer
```

This will launch the Gradio UI at `http://localhost:7860`.

> Tesseract is automatically installed and configured inside the Docker container.

---

## Project Structure

```
document-analyzer/
├── Document_Analyzer.py           # Main application logic
├── ocr.py                         # OCR helper functions
├── .env                           # API keys
├── Dockerfile                     # For containerized deployment
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Notes

* OCR is triggered only if the PDF contains no extractable text.
* Tesseract OCR is pre-installed in the Docker image with `eng.traineddata`.
* No user data is stored on disk — everything is processed in-memory.
* Make sure your API keys are valid and have quota.
* Use the **Reset** button in the UI to start fresh with a new document.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* [Meta LLaMA](https://github.com/facebookresearch/llama)
* [Groq API](https://www.groq.com/)
* [Google Generative AI Embeddings](https://developers.generativeai.google/)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [Gradio](https://gradio.app/)

---

Feel free to open issues or submit PRs to improve this project!

```
