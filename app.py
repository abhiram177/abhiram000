import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import re

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    return denoised

def postprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text.strip()

def perform_ocr(image_path):
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
    preprocessed_image = preprocess_image(image_path)
    
    # PaddleOCR detection
    paddle_result = paddle_ocr.ocr(image_path, cls=True)
    paddle_text_lines = [line[1][0] for line in paddle_result[0]]
    
    # Tesseract OCR line by line
    tesseract_image = Image.fromarray(preprocessed_image)
    tesseract_lines = pytesseract.image_to_string(
        tesseract_image, 
        lang='eng'
    ).splitlines()
    
    # Post-process texts
    paddle_processed = [postprocess_text(line) for line in paddle_text_lines]
    tesseract_processed = [postprocess_text(line) for line in tesseract_lines if line.strip()]
    
    return paddle_processed, tesseract_processed

def main():
    image_path = r"/workspaces/abhiram177/images/hand1.jpeg"
    
    paddle_texts, tesseract_lines = perform_ocr(image_path)
    
    # Write results
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("PaddleOCR Detected Text:\n")
        f.write("\n".join(paddle_texts) + "\n\n")
        f.write("Tesseract OCR Text (Line by Line):\n")
        f.write("\n".join(tesseract_lines))
    
    # Print results
    print("PaddleOCR Text:")
    print("\n".join(paddle_texts))
    print("\nTesseract Text:")
    print("\n".join(tesseract_lines))

if __name__ == "__main__":
    main()