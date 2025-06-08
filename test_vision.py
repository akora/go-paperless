#!/usr/bin/env python3

import os
from google.cloud import vision
from pathlib import Path

# Set up credentials
SCRIPT_DIR = Path(__file__).parent.resolve()
CREDENTIALS_FILE = str(SCRIPT_DIR / "credentials" / "vision-api-credentials.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_FILE

def test_vision_api():
    try:
        # Create a client
        client = vision.ImageAnnotatorClient()
        print("✅ Successfully created Vision API client")
        
        # Get supported languages
        print("\nSupported languages:")
        image_context = vision.ImageContext(
            language_hints=['en', 'hu', 'de']
        )
        print("✅ Successfully configured language hints")
        
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Google Cloud Vision API setup...")
    test_vision_api()
