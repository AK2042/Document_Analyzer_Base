from PIL import Image
import pytesseract
import cv2
import numpy as np
import fitz 
from langchain_core.documents import Document
import os

custom_config = r'--tessdata-dir "/usr/share/tessdata"'

def extract_text_from_image(image_path, lang='eng'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised_image = cv2.medianBlur(binary_image, 3)

    scale_percent = 150
    width = int(denoised_image.shape[1] * scale_percent / 100)
    height = int(denoised_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_LINEAR)

    pil_image = Image.fromarray(resized_image)
    return pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)

def extract_text_from_pdf_ocr(file_obj, lang='eng'):
    docs = []

    if hasattr(file_obj, "read"):
        pdf_doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    elif isinstance(file_obj, str):
        pdf_doc = fitz.open(file_obj)
    else:
        raise ValueError("Unsupported file type")

    for i in range(len(pdf_doc)):
        page = pdf_doc.load_page(i)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.medianBlur(binary, 3)
        resized = cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        text = pytesseract.image_to_string(resized, lang=lang, config=custom_config)
        if text.strip():
            docs.append(Document(page_content=text.strip(), metadata={"page": i + 1}))

    return docs
