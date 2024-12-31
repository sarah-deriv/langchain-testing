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
├── QuestionAnswering.py    # Main question answering implementation
├── ReadFromPdf.py         # PDF document processing with text splitting
├── ReadFromYoutube.py     # YouTube transcript extraction
├── retrieval.py           # Document retrieval implementation
├── selfQueryRetriever.py  # Self-query retrieval system
├── vectore_store.py       # Vector store implementation
├── PDF-docs/              # Directory for PDF documents
└── docs/                  # Directory for ChromaDB persistence
    └── chroma/           # ChromaDB storage
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
   ```

## Usage

### PDF Processing
```python
# Load and process PDF documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("PDF-docs/your_document.pdf")
pages = loader.load()
```

### YouTube Transcript Processing
```python
# Extract transcripts from YouTube videos
from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("your_youtube_url")
docs = loader.load()
```

### Question Answering
```python
# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)

# Ask questions
result = qa_chain.invoke({"query": "your question here"})
```

## Key Components

- **Text Splitting**: Implements both RecursiveCharacterTextSplitter and CharacterTextSplitter with configurable chunk sizes and overlaps
- **Vector Storage**: Uses ChromaDB for efficient storage and retrieval of document embeddings
- **Question Answering**: Supports multiple chain types and custom prompt templates
- **Document Processing**: Handles both PDF documents and YouTube video transcripts

## Environment Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages (see Setup section)

This repository serves as a practical implementation of LangChain's capabilities for document processing and question answering, suitable for both learning and production use cases.
