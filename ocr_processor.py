import logging
from typing import List, Tuple
import numpy as np
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    import cv2
    import fitz  # PyMuPDF for PDF handling
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing for scanned documents."""
    
    def __init__(self):
        if not OCR_AVAILABLE:
            raise ImportError("OCR dependencies not available. Install: pip install paddleocr PyMuPDF opencv-python")
        
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_scanned_pdf(self, pdf_path: str) -> List[str]:
        """
        Process scanned PDF using OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted text from each page
        """
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR processing not available")
        
        self.logger.info(f"Processing scanned PDF with OCR: {pdf_path}")
        
        extracted_texts = []
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # Scale factor for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Run OCR
                ocr_results = self.ocr.ocr(img, cls=True)
                
                # Extract text
                page_text = []
                if ocr_results and ocr_results[0]:
                    for result in ocr_results[0]:
                        text = result[1][0]  # Get text content
                        confidence = result[1][1]  # Get confidence score
                        
                        if confidence > 0.5:  # Filter low-confidence results
                            page_text.append(text)
                
                extracted_text = '\n'.join(page_text)
                extracted_texts.append(extracted_text)
                
                self.logger.info(f"Extracted {len(page_text)} text blocks from page {page_num + 1}")
            
            pdf_document.close()
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            raise
        
        return extracted_texts