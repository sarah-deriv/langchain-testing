# LangChain Testing Repository

This repository contains a collection of Python scripts demonstrating various functionalities of LangChain for document processing, question answering, and information retrieval. The project showcases different approaches to handling documents from multiple sources and implementing intelligent question-answering systems.

## Features

- **Document Processing**
  - PDF document reading and processing with configurable text splitting
  - YouTube video transcript extraction and processing
  - Vector store implementation for efficient document storage
  
- **Question Answering**
  - Implementation of question-answering systems using GPT-4
  - Retrieval-based question answering with customizable prompts
  - Support for different chain types (map_reduce, refine)
  - Similarity search functionality

- **Vector Store**
  - Document embedding using OpenAI embeddings
  - Efficient similarity search with ChromaDB
  - Persistent storage of embeddings

## Repository Structure

```
langchain-testing/
├── chat.py                # Main chat interface
├── retrieval.py          # Document retrieval implementation
├── selfQueryRetriever.py # Self-query retrieval system
├── examples/            # Example implementations
│   ├── pdf_example.py    # PDF processing example
│   ├── qa_example.py     # Question answering example
│   ├── vector_store_example.py  # Vector store usage
│   └── youtube_example.py # YouTube processing example
├── library/             # Core functionality modules
│   ├── pdf_processor.py   # PDF processing utilities
│   ├── qa_processor.py    # QA system implementation
│   ├── utils.py          # Common utilities
│   ├── vector_store_processor.py # Vector store operations
│   └── youtube_processor.py # YouTube processing utilities
├── tests/              # Test suite
│   ├── __init__.py      # Test package initialization
│   ├── conftest.py      # Test fixtures and configuration
│   ├── test_utils.py    # Utility function tests
│   └── test_qa_processor.py # QA processor tests
├── PDF-docs/            # Directory for PDF documents
└── docs/               # Directory for ChromaDB persistence
    └── chroma/        # ChromaDB storage
```

## Setup

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install langchain langchain_openai langchain_community langchain_chroma openai python-dotenv
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   LANGCHAIN_API_KEY=your_api_key_here
   ```

## Usage

### Main Chat Interface
```python
# Use the main chat interface
python chat.py
```

### PDF Processing
```python
# Example of PDF document processing
from library.pdf_processor import process_pdf
docs = process_pdf("PDF-docs/your_document.pdf")
```

### YouTube Transcript Processing
```python
# Example of YouTube transcript processing
from library.youtube_processor import process_youtube
docs = process_youtube("your_youtube_url")
```

### Question Answering
```python
# Example of using the QA system
from library.qa_processor import QAProcessor

qa_processor = QAProcessor()
result = qa_processor.ask_question("your question here")
```

## Key Components

- **Modular Architecture**: Core functionality is separated into reusable modules in the library/ directory
- **Example Implementations**: Complete examples demonstrating different use cases in the examples/ directory
- **Text Splitting**: Implements both RecursiveCharacterTextSplitter and CharacterTextSplitter with configurable chunk sizes and overlaps
- **Vector Storage**: Uses ChromaDB for efficient storage and retrieval of document embeddings
- **Question Answering**: Supports multiple chain types and custom prompt templates
- **Document Processing**: Handles both PDF documents and YouTube video transcripts

## Environment Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages (see Setup section)

## Testing

The project includes a comprehensive test suite to ensure reliability and maintainability. Tests are written using pytest and cover core functionalities including environment setup, QA processing, and utility functions.

### Test Structure

- **Fixtures (`conftest.py`)**: Provides reusable test components like mock retrievers and environment variables
- **Utility Tests**: Tests for environment loading and configuration
- **QA Processor Tests**: Tests for QA functionality including:
  - Initialization and configuration
  - Different chain types
  - Custom prompts
  - Error handling

### Running Tests

1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

2. Run the test suite:
   ```bash
   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=library

   # Run specific test file
   pytest tests/test_qa_processor.py
   ```

### Writing New Tests

When adding new functionality:
1. Create a new test file in the `tests/` directory
2. Add relevant fixtures to `conftest.py` if needed
3. Follow existing test patterns for consistency
4. Ensure proper mocking of external dependencies
5. Include both success and error cases

This repository serves as a practical implementation of LangChain's capabilities for document processing and question answering, suitable for both learning and production use cases.
