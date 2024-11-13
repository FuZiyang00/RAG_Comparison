import io
import logging
from typing import Optional
from .utils import setup_logging

import pytesseract
from PIL import Image
from PyPDF2 import PageObject, PdfReader

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class OCR:
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
     
    def extract_text_from_pdf(self) -> str:
        """
        Extracts text from a PDF file. Uses OCR if text extraction fails for any page.

        Returns:
            str: Extracted and cleaned text from the PDF.
        """
        text = ""
        try:
            with open(self.pdf_path, "rb") as f:
                pdf_reader = PdfReader(f)
                logger.info(f"Opened PDF file for text extraction: {self.pdf_path}")

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                            logger.info(f"Extracted text from page {page_num} without OCR.")
                        else:
                            logger.info(f"No text found on page {page_num}; attempting OCR.")
        
                            text += OCR.extract_text_from_images(page)
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
                        continue  # Skip to the next page on error
                        
        except FileNotFoundError:
            logger.error(f"File not found: {self.pdf_path}")
            raise 

        # Custom assertion with error message if no text was extracted
        assert len(text) > 0, f"Blank PDF: {self.pdf_path}"

        logger.info(f"Completed text extraction for {self.pdf_path}")
        return text
    
    @staticmethod
    def extract_text_from_images(page: PageObject) -> str:
        """
        Extracts text from images on a page using OCR.

        Args:
            page (PageObject): The PDF page object containing images.

        Returns:
            str: Extracted text from images using OCR.
        """
        text = ""
        for image_file_object in page.images:
            try:
                image = Image.open(io.BytesIO(image_file_object.data))
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text
                logger.info("Extracted text from image using OCR.")
            except Exception as e:
                print(f"Error processing image for OCR: {e}")
                logger.error(f"Error processing image for OCR: {e}")
        return text
