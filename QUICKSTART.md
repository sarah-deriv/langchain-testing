# Quick Start Guide - Document Chat System

Hey there! 👋 Welcome to the Document Chat System project. Here's what you need to know to get started:

## In 30 Seconds
- This is a PDF document chat system using LangChain + OpenAI
- Two main files: `prepareDocumentStore.py` (processes PDFs) and `chatWithDocuments.py` (handles chat)
- Uses vector embeddings (Chroma DB) for document search and GPT-3.5 for chat

## Get Running in 5 Minutes

1. **Setup**
   ```bash
   pip install -r requirements-test.txt
   cp .env.example .env  # Add your OpenAI API key
   ```

2. **Add Documents**
   - Put PDFs in `PDF-docs/` directory
   - Run: `python prepareDocumentStore.py`

3. **Start Chatting**
   - Run: `python chatWithDocuments.py`
   - Type questions about your documents
   - Type 'exit' to quit

## Key Files
- `prepareDocumentStore.py`: Processes PDFs into searchable chunks
- `chatWithDocuments.py`: Handles chat interface and document queries
- `README.md`: User guide and setup instructions
- `HANDOVER.md`: Detailed technical documentation

## Need More?
- Check `HANDOVER.md` for detailed technical documentation
- Key configurations are in the .env file
- Vector store is saved in docs/chroma/

## Common Issues
- OpenAI API key not set → Check .env file
- PDFs not found → Ensure they're in PDF-docs/
- Vector store errors → Delete docs/chroma/ and rerun preparation

Happy coding! 🚀
