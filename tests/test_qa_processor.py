"""Test cases for QA processor functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import BaseRetriever
from langchain_core.runnables import Runnable
from library.qa_processor import (
    QAProcessor,
    QAConfig,
    QAProcessingError,
    DefaultPrompts
)

class MockChatOpenAI(Mock):
    """Mock ChatOpenAI that implements Runnable interface."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__ = type('MockChatOpenAI', (Mock, Runnable), {})

@pytest.fixture
def mock_chat_openai():
    """Create a mock ChatOpenAI instance that implements Runnable."""
    mock = MockChatOpenAI()
    mock.invoke.return_value = "Mock response"
    return mock

@pytest.fixture
def mock_qa_chain():
    """Create a mock QA chain."""
    chain = Mock()
    chain.invoke.return_value = {
        "result": "This is a test answer",
        "source_documents": []
    }
    return chain

@pytest.fixture
def qa_processor(mock_retriever, mock_chat_openai, mock_qa_chain):
    """Create a QA processor instance with mock components."""
    with patch('library.qa_processor.ChatOpenAI', return_value=mock_chat_openai), \
         patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        config = QAConfig(
            model_name="gpt-4",
            temperature=0,
            chain_type="stuff",
            return_source_documents=True
        )
        return QAProcessor(mock_retriever, config)

def test_init_qa_processor(mock_retriever, mock_chat_openai, mock_qa_chain):
    """Test QA processor initialization."""
    with patch('library.qa_processor.ChatOpenAI', return_value=mock_chat_openai), \
         patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        processor = QAProcessor(mock_retriever)
        assert processor.config.model_name == "gpt-4"
        assert processor.config.temperature == 0
        assert processor.config.chain_type == "stuff"
        assert processor.config.return_source_documents is True
        assert processor.retriever == mock_retriever

def test_init_qa_processor_custom_config(mock_retriever, mock_chat_openai, mock_qa_chain):
    """Test QA processor initialization with custom config."""
    config = QAConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        chain_type="map_reduce",
        return_source_documents=False
    )
    with patch('library.qa_processor.ChatOpenAI', return_value=mock_chat_openai), \
         patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        processor = QAProcessor(mock_retriever, config)
        assert processor.config.model_name == "gpt-3.5-turbo"
        assert processor.config.temperature == 0.7
        assert processor.config.chain_type == "map_reduce"
        assert processor.config.return_source_documents is False

def test_update_config(qa_processor, mock_chat_openai, mock_qa_chain):
    """Test updating QA processor configuration."""
    new_config = QAConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        chain_type="refine"
    )
    with patch('library.qa_processor.ChatOpenAI', return_value=mock_chat_openai), \
         patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        qa_processor.update_config(new_config)
        assert qa_processor.config.model_name == "gpt-3.5-turbo"
        assert qa_processor.config.temperature == 0.5
        assert qa_processor.config.chain_type == "refine"

def test_ask_question(qa_processor):
    """Test asking a question with default settings."""
    result = qa_processor.ask("What is a test question?")
    assert result["result"] == "This is a test answer"
    qa_processor.qa_chain.invoke.assert_called_once()

def test_ask_question_with_custom_prompt(qa_processor, mock_chat_openai, mock_qa_chain):
    """Test asking a question with a custom prompt template."""
    with patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        result = qa_processor.ask(
            "What is a test question?",
            prompt_template=DefaultPrompts.CONCISE
        )
        assert result["result"] == "This is a test answer"
        mock_qa_chain.invoke.assert_called_once()

def test_qa_processing_error(qa_processor):
    """Test error handling in QA processor."""
    qa_processor.qa_chain.invoke.side_effect = Exception("Test error")
    
    with pytest.raises(QAProcessingError) as exc_info:
        qa_processor.ask("What is a test question?")
    
    assert "Error processing question" in str(exc_info.value)

def test_different_chain_types(mock_retriever, mock_chat_openai):
    """Test QA processor with different chain types."""
    chain_types = ["stuff", "map_reduce", "refine"]
    
    for chain_type in chain_types:
        mock_chain = Mock()
        mock_chain.invoke.return_value = {
            "result": f"Answer using {chain_type} chain",
            "source_documents": []
        }
        
        with patch('library.qa_processor.ChatOpenAI', return_value=mock_chat_openai), \
             patch('library.qa_processor.RetrievalQA') as mock_retrieval_qa:
            mock_retrieval_qa.from_chain_type.return_value = mock_chain
            config = QAConfig(chain_type=chain_type)
            processor = QAProcessor(mock_retriever, config)
            
            result = processor.ask("Test question")
            assert result["result"] == f"Answer using {chain_type} chain"
