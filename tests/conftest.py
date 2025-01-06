"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain.schema import Document, BaseRetriever

@pytest.fixture
def mock_retriever():
    """Create a mock retriever that returns predefined documents."""
    retriever = Mock(spec=BaseRetriever)
    docs = [
        Document(
            page_content="A swap-free account is a special type of trading account that doesn't charge overnight fees.",
            metadata={"source": "test_doc.pdf", "page": 1}
        )
    ]
    retriever.get_relevant_documents = MagicMock(return_value=docs)
    return retriever

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('LANGCHAIN_API_KEY', 'test-langchain-key')
