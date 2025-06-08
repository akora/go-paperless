#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
import logging
from typing import List, Set, Dict
from datetime import datetime

# Configuration constants
# Default paths that can be overridden with environment variables
DEFAULT_SOURCE_DIR = os.path.join(os.path.expanduser('~'), 'paperless', 'images')
DEFAULT_TARGET_DIR = os.path.join(os.path.expanduser('~'), 'paperless', 'scans')

# Get directories from environment variables or use defaults
SOURCE_DIRS = os.environ.get('PAPERLESS_SOURCE_DIRS', DEFAULT_SOURCE_DIR).split(':')
TARGET_DIR = os.environ.get('PAPERLESS_TARGET_DIR', DEFAULT_TARGET_DIR)
PATTERNS = [
    "scan", "scanned", "scanner",
    "Scan", "Scanned", "Scanner",
    "scanDoc", "ScanDoc", "scan_doc",
    "scanFile", "ScanFile", "scan_file", "Doxie", "doxie", "Swiftscan", "swiftscan", "swiftScan", "SwiftScan"]  # Patterns in various formats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScanCollector:
    def __init__(self, source_dirs: List[str], target_dir: str, patterns: List[str]):
        """
        Initialize the ScanCollector.
        
        Args:
            source_dirs: List of directories to search for scanned documents
            target_dir: Directory where to move the found documents
            patterns: List of patterns to match in filenames (case-insensitive)
        """
        self.source_dirs = [Path(d).resolve() for d in source_dirs]
        self.target_dir = Path(target_dir).resolve()
        self.patterns = [p.lower() for p in patterns]
        
        # Validate directories
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep track of empty directories for potential cleanup
        self.empty_dirs: Set[Path] = set()
        
        # Statistics
        self.stats = {
            'files_found': 0,
            'files_moved': 0,
            'dirs_removed': 0,
            'errors': 0
        }

    def matches_patterns(self, filename: str) -> bool:
        """
        Check if filename matches any of the patterns.
        Handles various case formats (camelCase, PascalCase, snake_case, etc.)
        """
        # Convert filename to lowercase for case-insensitive matching
        filename_lower = filename.lower()
        
        # Split filename into parts to handle different formats
        # This will split on both underscores and case changes
        parts = []
        
        # First split by underscores
        for part in filename_lower.split('_'):
            # Then split by camel case
            current_word = ''
            for char in part:
                if char.isalpha():
                    current_word += char
                else:
                    if current_word:
                        parts.append(current_word)
                    current_word = ''
            if current_word:
                parts.append(current_word)
        
        # Join parts back for full-word matching
        joined_parts = ''.join(parts)
        
        # Check if any pattern matches either:
        # 1. The full lowercase filename
        # 2. Any individual part
        # 3. The joined parts
        return any(
            pattern.lower() in filename_lower or  # Full filename match
            pattern.lower() in parts or           # Individual part match
            pattern.lower() in joined_parts       # Joined parts match
            for pattern in self.patterns
        )

    def is_safe_to_remove(self, directory: Path) -> bool:
        """
        Check if a directory is safe to remove.
        A directory is safe to remove if it's empty or only contains empty directories.
        """
        try:
            # List all items in the directory
            items = list(directory.iterdir())
            
            # If directory is empty, it's safe to remove
            if not items:
                return True
            
            # If directory contains only other empty directories, check them recursively
            return all(
                item.is_dir() and self.is_safe_to_remove(item)
                for item in items
            )
        except Exception as e:
            logger.error(f"Error checking directory {directory}: {e}")
            return False

    def remove_empty_directory(self, directory: Path) -> bool:
        """Safely remove an empty directory and its empty parent directories."""
        try:
            if not self.is_safe_to_remove(directory):
                return False
            
            # Remove the directory and all empty parent directories
            while directory.exists() and self.is_safe_to_remove(directory):
                # Don't remove if it's one of the source directories
                if directory in self.source_dirs:
                    break
                
                shutil.rmtree(directory)
                self.stats['dirs_removed'] += 1
                logger.info(f"Removed empty directory: {directory}")
                
                # Move to parent directory
                directory = directory.parent
            
            return True
        except Exception as e:
            logger.error(f"Error removing directory {directory}: {e}")
            return False

    def move_file(self, source_file: Path) -> bool:
        """
        Move a file to the target directory, maintaining the date-based structure.
        Returns True if successful, False otherwise.
        """
        try:
            # Generate target path with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = self.target_dir / f"{timestamp}_{source_file.name}"
            
            # Ensure target path doesn't exist
            counter = 1
            while target_path.exists():
                target_path = self.target_dir / f"{timestamp}_{counter}_{source_file.name}"
                counter += 1
            
            # Move the file
            shutil.move(str(source_file), str(target_path))
            self.stats['files_moved'] += 1
            logger.info(f"Moved file: {source_file} -> {target_path}")
            
            # Add the parent directory to potentially empty dirs
            self.empty_dirs.add(source_file.parent)
            
            return True
        except Exception as e:
            logger.error(f"Error moving file {source_file}: {e}")
            self.stats['errors'] += 1
            return False

    def process_directory(self, directory: Path):
        """Process a directory recursively."""
        try:
            for item in directory.rglob('*'):
                if item.is_file() and self.matches_patterns(item.name):
                    self.stats['files_found'] += 1
                    self.move_file(item)
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            self.stats['errors'] += 1

    def cleanup_empty_dirs(self):
        """Clean up empty directories after moving files."""
        logger.info("Cleaning up empty directories...")
        for directory in sorted(self.empty_dirs, key=lambda x: len(str(x)), reverse=True):
            self.remove_empty_directory(directory)

    def collect(self):
        """Main method to collect and move scanned documents."""
        logger.info("Starting scan collection...")
        logger.info(f"Source directories: {', '.join(str(d) for d in self.source_dirs)}")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"Searching for patterns: {', '.join(self.patterns)}")

        # Process each source directory
        for source_dir in self.source_dirs:
            self.process_directory(source_dir)

        # Clean up empty directories
        self.cleanup_empty_dirs()

        # Print statistics
        logger.info("\nCollection completed!")
        logger.info(f"Files found: {self.stats['files_found']}")
        logger.info(f"Files moved: {self.stats['files_moved']}")
        logger.info(f"Empty directories removed: {self.stats['dirs_removed']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")

def main():
    try:
        collector = ScanCollector(SOURCE_DIRS, TARGET_DIR, PATTERNS)
        collector.collect()
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
