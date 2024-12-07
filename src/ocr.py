import logging
from .utils import setup_logging
from llama_parse import LlamaParse

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class OCR:
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
     
    def extract_text_from_pdf(self, api_key: str) -> str:
        """
        Extracts text from a PDF file

        Returns:
            str: Extracted and cleaned text from the PDF.
        """

        corpus = ""

        # set up parser
        try:
            docs = LlamaParse(result_type="markdown", api_key=api_key, 
                    num_workers=1).load_data(self.pdf_path)
            
            logger.info(f"Opened PDF file for text extraction: {self.pdf_path}")
            
            if len(docs) > 0:
                for i in range(len(docs)):
                    corpus += docs[i].text[:]
            
            else:
                logger.error(f"Error extracting text from PDF: {self.pdf_path}")
                raise ValueError("Error extracting text from PDF")
        
        except FileNotFoundError:
            logger.error(f"File not found: {self.pdf_path}")
            raise
        
        assert len(corpus) > 0, f"Blank PDF: {self.pdf_path}"
        return corpus