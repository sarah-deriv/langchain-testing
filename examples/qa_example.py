"""Example usage of the QA processor library."""

import sys
import os
from typing import List

# Get the absolute path to the library directory
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(os.path.dirname(current_dir), 'library')
sys.path.append(library_dir)

from qa_processor import QAProcessor, QAConfig, DefaultPrompts, QAProcessingError
from vector_store_processor import VectorStoreProcessor, VectorStoreConfig
from utils import load_environment
from langchain_community.document_loaders import PyPDFLoader

def demonstrate_basic_qa(processor: QAProcessor):
    """Demonstrate basic QA operations."""
    # Basic question answering
    question = "What is a swap-free account?"
    print(f"\nBasic question: '{question}'")
    result = processor.ask(question)
    print(f"Answer: {result['result']}")
    
    if result.get('source_documents'):
        print("\nSource document:")
        print(f"Source: {result['source_documents'][0].metadata.get('source', 'N/A')}")
        print(f"Content: {result['source_documents'][0].page_content[:200]}...")

def demonstrate_advanced_qa(processor: QAProcessor):
    """Demonstrate advanced QA features."""
    question = "what is a zero spread account?"
    
    # Using concise prompt
    print(f"\n1. Using concise prompt for: '{question}'")
    result = processor.ask(question, prompt_template=DefaultPrompts.CONCISE)
    print(f"Concise answer: {result['result']}")
    
    # Using detailed prompt
    print(f"\n2. Using detailed prompt for: '{question}'")
    result = processor.ask(question, prompt_template=DefaultPrompts.DETAILED)
    print(f"Detailed answer: {result['result']}")
    
    # Using different chain types
    print("\n3. Using different chain types")
    chain_types = ["stuff", "map_reduce", "refine"]
    
    for chain_type in chain_types:
        print(f"\nTrying {chain_type} chain:")
        new_config = QAConfig(chain_type=chain_type)
        processor.update_config(new_config)
        result = processor.ask(question)
        print(f"Answer: {result['result']}")

def main():
    # Load environment
    load_environment()
    
    # First set up the vector store
    vector_config = VectorStoreConfig(
        chunk_size=1000,
        chunk_overlap=100,
        persist_directory=os.path.join(os.path.dirname(current_dir), 'docs/chroma/')
    )
    vector_processor = VectorStoreProcessor(vector_config)
    
    try:
        # Load documents
        pdf_dir = os.path.join(os.path.dirname(current_dir), 'PDF-docs')
        pdf_files = [
            os.path.join(pdf_dir, f"Engineering{i}.pdf") for i in range(1, 7)
        ]
        
        docs = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                loader = PyPDFLoader(pdf_file)
                docs.extend(loader.load())
            else:
                print(f"Warning: File not found - {pdf_file}")
        
        if not docs:
            print("No documents were loaded. Please check the PDF files exist.")
            return
            
        # Create vector store
        vector_processor.create_vector_store(docs)
        print(f"Created vector store with {vector_processor.get_collection_count()} chunks")
        
        # Create QA processor
        qa_config = QAConfig(
            model_name="gpt-4",
            temperature=0,
            chain_type="stuff",
            return_source_documents=True
        )
        qa_processor = QAProcessor(
            vector_processor.vectordb.as_retriever(),
            qa_config
        )
        
        # Demonstrate features
        print("\n=== Basic QA Features ===")
        demonstrate_basic_qa(qa_processor)
        
        print("\n=== Advanced QA Features ===")
        demonstrate_advanced_qa(qa_processor)
        
    except (QAProcessingError, Exception) as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
