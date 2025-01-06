# Document Chat System

> 👉 **New Developer?** Check out [QUICKSTART.md](QUICKSTART.md) for a quick setup guide and [HANDOVER.md](HANDOVER.md) for detailed technical documentation.

This project implements a document-based chat system using LangChain and OpenAI's language models. It allows users to interact with their PDF documents through natural language queries, leveraging vector embeddings for efficient document retrieval and GPT models for generating contextual responses.

## Overview

The system consists of two main components:
1. Document Preparation (`prepareDocumentStore.py`): Processes PDF documents and creates a vector store
2. Interactive Chat (`chatWithDocuments.py`): Enables natural language interaction with the processed documents

This implementation is based on the LangChain course: [Chat with Your Data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data)

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements-test.txt
```
3. Set up your environment variables (see Environment Variables section)
4. Place your PDF documents in the PDF-docs directory
5. Run the document preparation script
6. Start chatting with your documents

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
pdf_files_path=PDF-docs
```

- `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings and chat)
- `pdf_files_path`: Directory path where PDF files are stored (default: PDF-docs)

## File Structure

```
langchain-testing/
├── PDF-docs/           # Store your PDF files here
├── docs/
│   └── chroma/        # Vector store data
├── prepareDocumentStore.py
├── chatWithDocuments.py
└── library/           # Helper modules
```

## Main Components

### 1. Document Preparation (prepareDocumentStore.py)

This script processes PDF documents and creates a searchable vector store:

- Loads PDF files from the specified directory
- Splits documents into chunks (default: 1000 tokens with 100 token overlap)
- Creates embeddings using OpenAI's text-embedding-ada-002 model
- Stores the vectors in a Chroma database

Configuration:
```python
chunk_size=1000        # Size of text chunks
chunk_overlap=100      # Overlap between chunks
embedding_model="text-embedding-ada-002"
```

### 2. Chat Interface (chatWithDocuments.py)

Provides an interactive chat interface to query your documents:

- Uses GPT-3.5-turbo for generating responses
- Maintains conversation history for context
- Retrieves relevant document chunks based on similarity search
- Generates contextual responses based on retrieved information

Features:
- Similarity search with top-3 chunk retrieval
- Conversation memory for follow-up questions
- Temperature setting of 0.7 for balanced creativity
- Built-in error handling and graceful exits

## Usage

1. **Prepare Your Documents**:
   - Place your PDF files in the `PDF-docs` directory
   - Run the document preparation script:
     ```bash
     python prepareDocumentStore.py
     ```
   - The script will process all PDFs and create a vector store

2. **Chat with Your Documents**:
   - Start the chat interface:
     ```bash
     python chatWithDocuments.py
     ```
   - Type your questions naturally
   - Type 'exit' to end the conversation

Example Interaction:
```
=== Document Chat System ===
Chat initialized! Type 'exit' to end the conversation.
You can now ask questions about the documents.

You: What are the main topics covered in the documents?
Assistant: [AI-generated response based on your documents]

You: Can you elaborate on [specific topic]?
Assistant: [Context-aware response considering chat history]
```

## Best Practices

1. **Document Preparation**:
   - Ensure PDFs are text-searchable (not scanned images)
   - Use descriptive filenames
   - Keep PDFs in the designated PDF-docs directory

2. **Querying**:
   - Ask specific questions for better results
   - Provide context in your questions
   - Use follow-up questions to drill down into topics

3. **System Management**:
   - Regularly update your vector store when adding new documents
   - Monitor your OpenAI API usage
   - Back up your vector store (docs/chroma directory) periodically

## Troubleshooting

Common issues and solutions:

1. **OpenAI API Key Error**:
   - Ensure your `.env` file exists and contains the correct API key
   - Check if the API key has sufficient credits

2. **PDF Loading Issues**:
   - Verify PDF files are in the correct directory
   - Ensure PDFs are not corrupted or password-protected
   - Check file permissions

3. **Vector Store Errors**:
   - Delete the docs/chroma directory and rebuild if corruption occurs
   - Ensure sufficient disk space for vector storage

For additional support or feature requests, please open an issue in the repository.
