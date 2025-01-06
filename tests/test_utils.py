"""Test cases for utility functions."""

import os
import pytest
from unittest.mock import patch
from library.utils import load_environment

def test_load_environment(mock_env_vars):
    """Test loading environment variables."""
    env = load_environment()
    
    assert env['openai_api_key'] == 'test-key'
    assert env['langchain_api_key'] == 'test-langchain-key'
    assert env['pdf_files_path'] == 'test-pdf-path'

def test_load_environment_missing_vars(monkeypatch):
    """Test loading environment with missing variables."""
    # Clear all relevant environment variables
    for var in ['OPENAI_API_KEY', 'LANGCHAIN_API_KEY', 'PDF_FILES_PATH']:
        monkeypatch.delenv(var, raising=False)
    
    # Ensure no .env file is loaded by mocking find_dotenv to return an empty string
    with patch('library.utils.find_dotenv', return_value=''):
        env = load_environment()
        assert env['openai_api_key'] is None
        assert env['langchain_api_key'] is None
        # Should default to 'PDF-docs' when not set
        assert env['pdf_files_path'] == 'PDF-docs'
