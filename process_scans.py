#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import logging
from typing import Optional, Tuple

# Directory configurations
# Default paths that can be overridden with environment variables
INPUT_DIR = os.environ.get('PAPERLESS_INPUT_DIR', os.path.join(os.path.expanduser('~'), 'paperless', 'inbox'))
ARCHIVE_DIR = os.environ.get('PAPERLESS_ARCHIVE_DIR', os.path.join(os.path.expanduser('~'), 'paperless', 'documents', 'scanned'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, input_dir: str, archive_dir: str):
        self.input_dir = Path(input_dir)
        self.archive_dir = Path(archive_dir)
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def extract_metadata(self, file_path: Path) -> dict:
        """Extract metadata from a file using exiftool."""
        try:
            # Run exiftool and capture its output
            result = subprocess.run(
                ['exiftool', '-json', str(file_path)],
                capture_output=True,
                text=True,
                check=True
            )
            import json
            metadata = json.loads(result.stdout)[0]
            return metadata
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error while extracting metadata: {e}")
            return {}

    def get_creation_date(self, metadata: dict) -> datetime:
        """Extract creation date from metadata or return current time if not available."""
        try:
            # Try different possible metadata fields for creation date
            for field in ['CreateDate', 'CreationDate', 'DateTimeOriginal', 'FileModifyDate']:
                date_str = metadata.get(field)
                if date_str:
                    # ExifTool typically returns dates in format: "YYYY:MM:DD HH:MM:SS"
                    # or "YYYY:MM:DD HH:MM:SS+XX:XX" (with timezone)
                    # or "YYYY:MM:DD HH:MM:SSZ" (UTC)
                    
                    # Remove timezone indicators and clean up the string
                    date_str = date_str.split('+')[0].strip()  # Remove any +XX:XX
                    date_str = date_str.replace('Z', '').strip()  # Remove Z (UTC indicator)
                    date_str = date_str.replace(':', '')  # Remove colons
                    
                    # Now parse the date
                    return datetime.strptime(date_str, '%Y%m%d %H%M%S')
            
            # If no creation date found, use current time
            logger.warning("No creation date found in metadata, using current time")
            return datetime.now()
        except Exception as e:
            logger.error(f"Error parsing creation date from {date_str}: {e}, using current time")
            return datetime.now()

    def get_scanner_info(self, metadata: dict) -> str:
        """Extract scanner information from metadata."""
        # Check for Doxie in Creator field first
        creator = metadata.get('Creator', '').strip()
        if 'Doxie' in creator:
            return 'doxie'
            
        # If not Doxie, try standard Make/Model fields
        scanner_make = metadata.get('Make', '').strip()
        scanner_model = metadata.get('Model', '').strip()
        
        if scanner_make and scanner_model:
            return f"{scanner_make}-{scanner_model}"
        elif scanner_make:
            return scanner_make
        elif scanner_model:
            return scanner_model
            
        return ""

    def get_date_based_directory(self, date: datetime) -> Path:
        """Generate date-based directory path (YYYY/YYYY-MM)."""
        return Path(str(date.year), f"{date.year}-{date.month:02d}")

    def process_file(self, file_path: Path) -> Optional[Path]:
        """Process a single file: extract metadata and move to archive while preserving filename."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Extract metadata
            metadata = self.extract_metadata(file_path)
            
            # Get creation date and generate date-based directory
            creation_date = self.get_creation_date(metadata)
            date_dir = self.get_date_based_directory(creation_date)
            
            # Create full target directory
            target_dir = self.archive_dir / date_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Keep original filename
            new_path = target_dir / file_path.name
            
            # Check for duplicates
            if new_path.exists():
                # If the file exists, check if it's a true duplicate by comparing size
                if new_path.stat().st_size == file_path.stat().st_size:
                    # Files are identical - consider it a duplicate
                    logger.info(f"Duplicate file detected {file_path.name}, deleting it")
                    file_path.unlink()  # Delete the duplicate from input directory
                    return None  # Nothing to process, file was deleted
                else:
                    # Files are different, keep both but log a warning
                    logger.warning(f"File with same name exists but different content: {file_path.name}")
                    return None
            
            # Move file to archive
            shutil.move(str(file_path), str(new_path))
            logger.info(f"Processed {file_path.name} -> {date_dir}/{new_path.name}")
            return new_path
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def process_directory(self):
        """Process all supported files in the input directory."""
        supported_extensions = {'.jpg', '.jpeg', '.pdf'}
        processed_count = 0
        error_count = 0
        
        for file_path in self.input_dir.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                if self.process_file(file_path):
                    processed_count += 1
                else:
                    error_count += 1
        
        if processed_count > 0 or error_count > 0:
            logger.info(f"Processing complete. Successfully processed: {processed_count}, Errors: {error_count}")

def main():
    processor = DocumentProcessor(INPUT_DIR, ARCHIVE_DIR)
    processor.process_directory()

if __name__ == "__main__":
    main()
