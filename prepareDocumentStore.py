"""Script to prepare vector store from PDF documents."""

import sys
import os

# Get the absolute path to the library directory
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(current_dir, 'library')
sys.path.append(library_dir)

from vector_store_processor import VectorStoreProcessor, VectorStoreConfig, VectorStoreProcessingError
from utils import load_environment
from langchain_community.document_loaders import PyPDFLoader

def main():
    # Load environment variables (for OpenAI API key)
    load_environment()
    
    # Configure vector store
    config = VectorStoreConfig(
        chunk_size=1000,  # Size of text chunks
        chunk_overlap=100,  # Overlap between chunks for context preservation
        persist_directory=os.path.join(current_dir, 'docs/chroma/'),  # Where to save the vector store
        embedding_model="text-embedding-ada-002"  # OpenAI's embedding model
    )
    
    # Initialize processor
    processor = VectorStoreProcessor(config)
    
    try:
        # Get all PDF files from PDF-docs directory
        pdf_dir = os.path.join(current_dir, 'PDF-docs')
        pdf_files = [
            os.path.join(pdf_dir, f) 
            for f in os.listdir(pdf_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            print("No PDF files found in the PDF-docs directory.")
            return
            
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"- {os.path.basename(pdf_file)}")
        
        # Load all documents
        print("\nLoading documents...")
        docs = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                loader = PyPDFLoader(pdf_file)
                docs.extend(loader.load())
                print(f"Loaded: {os.path.basename(pdf_file)}")
            else:
                print(f"Warning: File not found - {pdf_file}")
        
        if not docs:
            print("No documents were loaded. Please check the PDF files exist.")
            return
        
        # Create and persist vector store
        print("\nCreating vector store...")
        processor.create_vector_store(docs)
        
        # Print statistics
        collection_count = processor.get_collection_count()
        print(f"\nSuccessfully created vector store with {collection_count} chunks")
        print(f"Vector store is persisted at: {config.persist_directory}")
        
    except VectorStoreProcessingError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
