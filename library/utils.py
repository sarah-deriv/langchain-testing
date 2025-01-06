"""Common utilities for LangChain applications."""

import os
from dotenv import load_dotenv, find_dotenv

def load_environment():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())

    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        # Add other environment variables as needed
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'pdf_files_path': os.getenv('PDF_FILES_PATH', 'PDF-docs')  # Default to 'PDF-docs' if not set
    }
