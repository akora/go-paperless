#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import logging
from typing import Optional, List, Dict
from google.cloud import vision
from pdf2image import convert_from_path
from PIL import Image, ImageStat
import io
from PyPDF2 import PdfReader
from datetime import datetime
import numpy as np
import cv2

# Directory configuration
# Default paths that can be overridden with environment variables
DOCUMENTS_DIR = os.environ.get('PAPERLESS_DOCUMENTS_DIR', os.path.join(os.path.expanduser('~'), 'paperless', 'documents', 'scanned'))
CONTENTS_DIR = os.environ.get('PAPERLESS_CONTENTS_DIR', os.path.join(os.path.expanduser('~'), 'paperless', 'documents', 'scanned-content'))

# Get the script's directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CREDENTIALS_FILE = str(SCRIPT_DIR / "credentials" / "vision-api-credentials.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentOCR:
    def __init__(self, documents_dir: str, contents_dir: str):
        self.documents_dir = Path(documents_dir)
        self.contents_dir = Path(contents_dir)
        
        if not self.documents_dir.exists():
            raise ValueError(f"Documents directory does not exist: {documents_dir}")
            
        # Create contents directory if it doesn't exist
        self.contents_dir.mkdir(parents=True, exist_ok=True)
        
        # Set credentials environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(CREDENTIALS_FILE).resolve())
        
        # Initialize Google Cloud Vision client
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Successfully initialized Google Cloud Vision client")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {e}")
            raise

        # Initialize usage tracking
        self.monthly_units_used = 0

    def get_text_filename(self, original_file: Path) -> Path:
        """Generate the text file path for storing OCR results, maintaining the same directory structure."""
        # Get relative path from documents directory
        rel_path = original_file.relative_to(self.documents_dir)
        
        # Create corresponding path in contents directory
        contents_path = self.contents_dir / rel_path
        
        # Change extension to .md for Markdown
        return contents_path.with_suffix('.md')

    def extract_text_from_image(self, image_content: bytes) -> str:
        """Extract text from an image using Google Cloud Vision API."""
        try:
            image = vision.Image(content=image_content)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # The first annotation contains the entire text
                return texts[0].description
            
            if response.error.message:
                raise Exception(
                    f'{response.error.message}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'
                )
                
            return ""
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return ""

    def might_contain_text(self, image: Image.Image) -> bool:
        """
        Use OpenCV to detect if an image might contain text.
        This is a preliminary check before using the more expensive Vision API.
        """
        try:
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to preprocess the image
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Detect edges
            edges = cv2.Canny(threshold, 50, 150, apertureSize=3)
            
            # Apply morphological operations to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on characteristics that might indicate text
            possible_text_regions = 0
            min_area = img.shape[0] * img.shape[1] * 0.0001  # Minimum area threshold
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / float(h)
                
                # Text-like characteristics:
                # - Not too small (noise)
                # - Not too large (whole image)
                # - Reasonable aspect ratio for text
                if (area > min_area and 
                    area < img.shape[0] * img.shape[1] * 0.5 and
                    0.1 < aspect_ratio < 15):
                    possible_text_regions += 1
                
                if possible_text_regions >= 3:  # Found enough regions that might be text
                    return True
            
            if possible_text_regions == 0:
                logger.info("No potential text regions detected in image")
                return False
            
            return possible_text_regions >= 1
            
        except Exception as e:
            logger.warning(f"Error in text detection: {e}")
            # If analysis fails, assume there might be text
            return True

    def has_sufficient_content(self, image: Image.Image, threshold: float = 0.99) -> bool:
        """
        Check if an image has enough content to warrant OCR.
        Returns False for nearly blank pages or images without text.
        
        Args:
            image: PIL Image object
            threshold: Threshold for considering a pixel as white/blank (0-1)
        """
        try:
            # First check if the image is mostly blank
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Get image statistics
            stats = ImageStat.Stat(gray_image)
            
            # Calculate the percentage of light pixels
            total_pixels = gray_image.width * gray_image.height
            light_threshold = int(255 * threshold)
            light_pixels = sum(1 for pixel in gray_image.getdata() if pixel > light_threshold)
            light_ratio = light_pixels / total_pixels
            
            # If more than 98% of pixels are light, consider it too blank
            if light_ratio > 0.98:
                logger.info(f"Image appears to be mostly blank (light pixel ratio: {light_ratio:.2%})")
                return False
            
            # If the image isn't blank, check if it contains any text-like regions
            return self.might_contain_text(image)
            
        except Exception as e:
            logger.warning(f"Error checking image content: {e}")
            # If we can't analyze the image, assume it has content
            return True

    def process_image(self, image_path: Path) -> Optional[str]:
        """Extract text from an image file using Google Cloud Vision."""
        try:
            # Open and check the image first
            with Image.open(image_path) as img:
                if not self.has_sufficient_content(img):
                    logger.info(f"Skipping OCR for {image_path} - insufficient content")
                    return ""
            
            # If image has content, proceed with OCR
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            return self.extract_text_from_image(content)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file using Google Cloud Vision."""
        try:
            all_text = []
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images, 1):
                # Check if page has sufficient content
                if not self.has_sufficient_content(image):
                    logger.info(f"Skipping OCR for page {i} - insufficient content")
                    continue
                
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Extract text from the page
                text = self.extract_text_from_image(img_byte_arr)
                if text:
                    all_text.append(f"## Page {i}\n\n{text}\n")
            
            return "\n".join(all_text) if all_text else None
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None

    def should_process_file(self, file_path: Path) -> bool:
        """Check if the file should be processed."""
        # Check if it's a supported file type
        if file_path.suffix.lower() not in ['.pdf', '.jpg', '.jpeg']:
            logger.debug(f"Skipping {file_path.name} - unsupported file type")
            return False
            
        # Get the corresponding text file path
        text_file = self.get_text_filename(file_path)
        
        # Check if text file exists
        if text_file.exists():
            # Check if the text file is empty (might indicate a failed previous attempt)
            if text_file.stat().st_size == 0:
                logger.info(f"Found empty text file for {file_path.name} - will reprocess")
                return True
            logger.debug(f"Skipping {file_path.name} - already processed")
            return False
        
        return True

    def format_as_markdown(self, text: str, original_file: Path, metadata: Dict[str, str] = None) -> str:
        """Format the OCR text as a Markdown document with YAML frontmatter."""
        # Get file information
        creation_date = self.get_creation_date(metadata) if metadata else datetime.now()
        scanner_info = self.get_scanner_info(metadata) if metadata else "unknown"
        file_size = original_file.stat().st_size
        
        # Create YAML frontmatter
        frontmatter = [
            "---",
            f"title: {original_file.stem}",
            f"date: {creation_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"scanner: {scanner_info}",
            f"original_file: {original_file.name}",
            f"file_size: {file_size}",
            "type: document",
            "---",
            ""
        ]
        
        # Format the text content
        if original_file.suffix.lower() == '.pdf':
            # For PDFs, keep the page markers but format them as headers
            content_lines = []
            current_page = None
            
            for line in text.split('\n'):
                if line.startswith('--- Page '):
                    current_page = line.replace('--- Page ', '').replace(' ---', '')
                    content_lines.append(f"\n## Page {current_page}\n")
                else:
                    content_lines.append(line)
            
            content = '\n'.join(content_lines)
        else:
            # For images, just use the text as is
            content = text
        
        # Combine frontmatter and content
        return '\n'.join(frontmatter + [content])

    def save_text(self, text: str, text_file: Path, original_file: Path, metadata: Dict[str, str] = None) -> bool:
        """Save extracted text as a Markdown file."""
        try:
            # Ensure the parent directory exists
            text_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Format the content as Markdown
            markdown_content = self.format_as_markdown(text, original_file, metadata)
            
            # Save with UTF-8 encoding to preserve special characters
            text_file.write_text(markdown_content, encoding='utf-8')
            logger.info(f"Saved OCR results to {text_file.relative_to(self.contents_dir)}")
            return True
        except Exception as e:
            logger.error(f"Error saving text to {text_file}: {e}")
            return False

    def extract_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from the file."""
        # For now, just return an empty dictionary
        return {}

    def get_creation_date(self, metadata: Dict[str, str]) -> datetime:
        """Get the creation date from the metadata."""
        # For now, just return the current date
        return datetime.now()

    def get_scanner_info(self, metadata: Dict[str, str]) -> str:
        """Get the scanner information from the metadata."""
        # For now, just return "unknown"
        return "unknown"

    def process_file(self, file_path: Path) -> bool:
        """Process a single file and save OCR results."""
        if not self.should_process_file(file_path):
            return False

        # Get the corresponding text file path
        text_file = self.get_text_filename(file_path)

        # Extract metadata first for use in Markdown frontmatter
        metadata = self.extract_metadata(file_path)

        # Count pages for usage tracking
        if file_path.suffix.lower() == '.pdf':
            try:
                pdf = PdfReader(file_path)
                num_pages = len(pdf.pages)
            except Exception as e:
                logger.error(f"Error counting PDF pages: {e}")
                num_pages = 1
        else:
            num_pages = 1

        try:
            # Create an empty file to mark as "in progress"
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.touch()
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.process_pdf(file_path)
            else:  # jpg, jpeg
                text = self.process_image(file_path)
                
            if not text:
                logger.error(f"No text extracted from {file_path.name}")
                text_file.unlink()  # Remove empty file if processing failed
                return False
            
            # Update usage tracking
            self.monthly_units_used += num_pages
                
            # Save the extracted text as Markdown
            return self.save_text(text, text_file, file_path, metadata)
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            if text_file.exists():
                text_file.unlink()  # Remove file if there was an error
            return False

    def process_directory(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """Process all documents in the date range."""
        # First, collect all files to process
        files_to_process = []
        total_pages = 0
        
        # Walk through the year directories
        for year_dir in sorted(self.documents_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
                
            year = int(year_dir.name)
            if start_date and year < start_date.year:
                continue
            if end_date and year > end_date.year:
                continue
            
            # Walk through month directories
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                    
                # Extract month from directory name (YYYY-MM format)
                try:
                    month_str = month_dir.name.split('-')[1]
                    month = int(month_str)
                except (IndexError, ValueError):
                    continue
                    
                if start_date and year == start_date.year and month < start_date.month:
                    continue
                if end_date and year == end_date.year and month > end_date.month:
                    continue
                
                # Process files in this month's directory
                for file_path in month_dir.iterdir():
                    if not file_path.is_file():
                        continue
                        
                    if not self.should_process_file(file_path):
                        continue
                        
                    # Count pages for cost estimation
                    if file_path.suffix.lower() == '.pdf':
                        try:
                            pdf = PdfReader(file_path)
                            pages = len(pdf.pages)
                        except Exception:
                            pages = 1
                    else:
                        pages = 1
                        
                    total_pages += pages
                    files_to_process.append((file_path, pages))
        
        if not files_to_process:
            logger.info("No files to process")
            return
            
        # Show initial summary
        logger.info(f"\nProcessing Summary:")
        logger.info(f"Files to process: {len(files_to_process)}")
        logger.info(f"Total pages: {total_pages}")
        
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        # Process files
        for file_path, _ in files_to_process:
            logger.info(f"Processing: {file_path.relative_to(self.documents_dir)}")
            if self.process_file(file_path):
                processed_count += 1
            else:
                error_count += 1
                    
        # Show final summary
        logger.info(f"\nProcessing Complete:")
        logger.info(f"- Successfully processed: {processed_count}")
        logger.info(f"- Skipped (already processed): {skipped_count}")
        logger.info(f"- Errors: {error_count}")

def main():
    try:
        processor = DocumentOCR(DOCUMENTS_DIR, CONTENTS_DIR)
        
        if len(sys.argv) == 1:
            # No arguments - process all documents
            logger.info("Processing all documents")
            processor.process_directory()
        elif sys.argv[1] == '--files':
            # Process specific files
            if len(sys.argv) < 3:
                logger.error("Please provide at least one file to process")
                logger.info("Usage: python ocr_documents.py --files file1 [file2 ...]")
                sys.exit(1)
            
            files_to_process = sys.argv[2:]
            logger.info(f"Processing {len(files_to_process)} specific files")
            
            processed_count = 0
            error_count = 0
            
            for file_path in files_to_process:
                full_path = Path(DOCUMENTS_DIR) / file_path
                if not full_path.exists():
                    logger.error(f"File not found: {file_path}")
                    error_count += 1
                    continue
                    
                logger.info(f"Processing: {file_path}")
                if processor.process_file(full_path):
                    processed_count += 1
                else:
                    error_count += 1
            
            logger.info(f"\nProcessing Complete:")
            logger.info(f"- Successfully processed: {processed_count}")
            logger.info(f"- Errors: {error_count}")
        elif len(sys.argv) == 3:
            # Process date range
            try:
                start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
                end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
                logger.info(f"Processing documents from {start_date.date()} to {end_date.date()}")
                processor.process_directory(start_date, end_date)
            except ValueError:
                logger.error("Invalid date format. Please use YYYY-MM-DD")
                sys.exit(1)
        else:
            logger.error("Invalid arguments")
            logger.info("Usage:")
            logger.info("  Process all files: python ocr_documents.py")
            logger.info("  Process date range: python ocr_documents.py YYYY-MM-DD YYYY-MM-DD")
            logger.info("  Process specific files: python ocr_documents.py --files file1 [file2 ...]")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
