# Paperless Document Processing System

A Python-based document processing system that automatically processes scanned documents, performs OCR, and organizes them in a structured way. The system creates Markdown files from the OCR results, making them perfect for use with Obsidian or other Markdown-based note-taking systems.

## Features

- Automatic document processing from scanner output
- Date-based organization (YYYY/YYYY-MM structure)
- Original filename preservation
- Intelligent OCR processing:
  - Detects and skips blank pages
  - Identifies and skips photos without text
  - Only processes pages with actual text content
- Markdown output with YAML frontmatter
- Support for PDF and image files (JPG, JPEG)
- Cost optimization for OCR processing
- Duplicate file detection and handling

## Directory Structure

```text
paperless/                           # Project root
├── credentials/                     # Google Cloud credentials
│   └── vision-api-credentials.json  # API credentials file
├── process_scans.py                # Script for processing new scans
├── ocr_documents.py                # Script for OCR processing
└── requirements.txt                # Python dependencies

External Directories (configurable via environment variables):
├── ~/paperless/inbox/                              # Drop folder for new scans (default)
└── ~/paperless/documents/
    ├── scanned/                                    # Processed documents
    │   └── YYYY/                                   # Year folders
        │       └── YYYY-MM/                            # Month folders with original files
        └── scanned-content/                            # OCR results as Markdown
            └── YYYY/                                   
                └── YYYY-MM/                            # Mirrored structure with .md files
```

## Prerequisites

1. Python 3.8 or higher
2. Google Cloud Vision API access
   - Create a project in Google Cloud Console
   - Enable the Vision API
   - Create a service account and download credentials
   - Save credentials as `credentials/vision-api-credentials.json`
3. Required system dependencies:
   - `exiftool` for metadata extraction
   - `poppler` for PDF processing
   - `opencv` for text detection

## Installation

1. Create and activate a Python virtual environment:

   ```bash
   # Create a new virtual environment
   python3 -m venv venv

   # Activate the virtual environment
   source venv/bin/activate
   ```

2. Install system dependencies:

   ```bash
   # macOS
   brew install exiftool poppler opencv
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up directories:

   ```bash
   # Create default directories
   mkdir -p ~/paperless/inbox
   mkdir -p ~/paperless/documents/scanned
   mkdir -p ~/paperless/documents/scanned-content
   mkdir -p credentials
   ```

5. Place your Google Cloud Vision API credentials in `credentials/vision-api-credentials.json`

## Configuration

The application uses environment variables for configuration. You can set these in your shell or in a `.env` file:

```bash
# Input directory for new scans
export PAPERLESS_INPUT_DIR=~/paperless/inbox

# Archive directory for processed documents
export PAPERLESS_ARCHIVE_DIR=~/paperless/documents/scanned

# Directory for OCR results
export PAPERLESS_CONTENTS_DIR=~/paperless/documents/scanned-content

# Source directories for collect_scans.py (colon-separated)
export PAPERLESS_SOURCE_DIRS=~/paperless/images:~/paperless/downloads

# Target directory for collect_scans.py
export PAPERLESS_TARGET_DIR=~/paperless/scans
```

If not set, the application will use sensible defaults in your home directory.

## Usage

Always activate the virtual environment before running the scripts:

```bash
source venv/bin/activate
```

### 1. Process New Scans

Before processing you may want to change the file's original creation date using exiftool. Here is an example:

```bash
exiftool -AllDates="1997:06:07 15:00:00" your_file.jpg
```

Place your scanned documents in the configured inbox directory (default: `~/paperless/inbox`), then run:

```bash
python3 process_scans.py
```

This will:

- Process all files in the inbox
- Extract metadata (creation date, scanner info)
- Preserve original filenames
- Move them to the appropriate date-based folder in `scanned/`
- Handle duplicates by checking file content (identical files are removed)

### 2. Perform OCR

After processing the scans, you have several options to run OCR:

```bash
# Process all documents
python3 ocr_documents.py

# Process documents from a specific date range
python3 ocr_documents.py 2024-01-01 2024-12-31

# Process specific files (paths relative to scanned/ directory)
python3 ocr_documents.py --files 2024/2024-01/document1.pdf 2024/2024-01/document2.pdf
```

The OCR process includes:

1. Pre-processing checks:
   - Detects and skips blank pages
   - Analyzes images for text content
   - Skips photos without text
2. OCR processing:
   - Only processes pages with actual text
   - Uses Google Cloud Vision API for high accuracy
   - Optimizes API usage to reduce costs
3. Output generation:
   - Creates Markdown files with extracted text
   - Includes metadata in YAML frontmatter
   - Organizes results in matching directory structure

## File Organization

Files are organized in a date-based directory structure while preserving their original names:

```text
YYYY/YYYY-MM/original_filename.ext
```

Example: `2024/2024-01/scan0001.pdf`

## Markdown Output Format

Each OCR result is saved as a Markdown file with YAML frontmatter:

```markdown
---
title: scanned-20240106-200111-123456-doxie
date: 2024-01-06 20:01:11
scanner: doxie
original_file: scanned-20240106-200111-123456-doxie.pdf
file_size: 123456
type: document
---

## Page 1
[Content of page 1]

## Page 2
[Content of page 2]
```

This format works well with Obsidian and other Markdown-based note systems.

## Troubleshooting

1. **Missing Metadata**: If creation date cannot be extracted, current time is used
2. **Failed OCR**: Empty text files are automatically removed and can be reprocessed
3. **Duplicate Files**: When a file with identical name and size already exists in the archive, the new file is considered a duplicate and is deleted. If files have the same name but different sizes, the new file is skipped and a warning is logged
4. **Environment Issues**: Make sure to activate the virtual environment before running scripts
5. **Permission Issues**: Ensure you have write access to all directories
6. **Blank Pages**: Pages that are mostly blank (>98% white) are automatically skipped
7. **Photos without Text**: Images that don't contain text-like regions are skipped to save processing costs
