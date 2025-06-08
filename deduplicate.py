#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import hashlib
import logging
import re
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Use the same archive directory as defined in process_scans.py
# Default path that can be overridden with environment variable
ARCHIVE_DIR = os.environ.get('PAPERLESS_ARCHIVE_DIR', os.path.join(os.path.expanduser('~'), 'paperless', 'documents', 'scanned'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Deduplicator:
    def __init__(self, archive_dir: str):
        self.archive_dir = Path(archive_dir)
        if not self.archive_dir.exists():
            raise ValueError(f"Archive directory does not exist: {archive_dir}")
        
        # Track file hashes for comparison
        self.file_hashes: Dict[str, List[Tuple[Path, datetime]]] = {}

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_file_creation_time(self, file_path: Path) -> datetime:
        """Get file creation time (or modification time as fallback)."""
        try:
            # Try to get creation time (ctime)
            stat = file_path.stat()
            return datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # Fallback to modification time if creation time is not available
            return datetime.fromtimestamp(file_path.stat().st_mtime)

    def process_directory(self, directory: Path):
        """Process a directory recursively to find and handle duplicates."""
        try:
            # First pass: collect all file hashes
            for item in directory.rglob("*"):
                if item.is_file():
                    file_hash = self.compute_file_hash(item)
                    creation_time = self.get_file_creation_time(item)
                    
                    if file_hash not in self.file_hashes:
                        self.file_hashes[file_hash] = []
                    self.file_hashes[file_hash].append((item, creation_time))

            # Second pass: handle duplicates
            for file_hash, files in self.file_hashes.items():
                if len(files) > 1:
                    # First, try to find a file without "_ver" in the name
                    original_candidates = [f for f, _ in files if "_ver" not in f.stem.lower()]
                    
                    if original_candidates:
                        # If we have files without "_ver", keep the one with the shortest path
                        original_file = min(original_candidates, key=lambda x: len(str(x)))
                    else:
                        # If all files have "_ver", keep the oldest one
                        files.sort(key=lambda x: (x[1], len(str(x[0]))))
                        original_file = files[0][0]
                    
                    logger.info(f"Keeping original file: {original_file}")
                    
                    # Remove all other duplicates
                    for duplicate_file, duplicate_time in files:
                        if duplicate_file != original_file:
                            try:
                                logger.info(f"Found duplicate: {duplicate_file}")
                                logger.info(f"Removing duplicate of: {original_file}")
                                duplicate_file.unlink()
                                logger.info(f"Successfully removed duplicate: {duplicate_file}")
                            except Exception as e:
                                logger.error(f"Failed to remove duplicate {duplicate_file}: {e}")

        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")

def main():
    try:
        deduplicator = Deduplicator(ARCHIVE_DIR)
        logger.info(f"Starting deduplication in: {ARCHIVE_DIR}")
        deduplicator.process_directory(deduplicator.archive_dir)
        logger.info("Deduplication completed")
    except Exception as e:
        logger.error(f"Error during deduplication: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
