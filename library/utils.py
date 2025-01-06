"""Common utilities for LangChain applications."""

import os
from dotenv import load_dotenv, find_dotenv

def load_environment():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "pr-artistic-push-97"
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        # Add other environment variables as needed
        'langchain_api_key' : os.getenv('LANGCHAIN_API_KEY')
        
    }
