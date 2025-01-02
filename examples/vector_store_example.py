"""Example usage of the Vector Store processor library."""

import sys
import os
from typing import List

# Get the absolute path to the library directory
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(os.path.dirname(current_dir), 'library')
sys.path.append(library_dir)

from vector_store_processor import VectorStoreProcessor, VectorStoreConfig, VectorStoreProcessingError
from utils import load_environment
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

def demonstrate_basic_usage(processor: VectorStoreProcessor, docs: List[Document]):
    """Demonstrate basic vector store operations."""
    # Create vector store
    processor.create_vector_store(docs)
    collection_count = processor.get_collection_count()
    print(f"\nCreated vector store with {collection_count} chunks")
    
    # Perform similarity search
    query = "what is a zero spread account?"
    print(f"\nBasic similarity search for: '{query}'")
    results = processor.similarity_search(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content[:200] + "...")
    
    # Try with different configuration
    new_config = VectorStoreConfig(
        chunk_size=500,
        chunk_overlap=50,
        persist_directory=os.path.join(os.path.dirname(current_dir), 'docs/chroma_small/')
    )
    processor.update_config(new_config)
    
    # Create new vector store with updated config
    processor.create_vector_store(docs)
    new_count = processor.get_collection_count()
    print(f"\nCreated new vector store with {new_count} chunks using smaller chunk size")

def demonstrate_advanced_features(processor: VectorStoreProcessor):
    """Demonstrate advanced vector store features like MMR search and metadata filtering."""
    # Example texts for direct text storage
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    
    print("\nDemonstrating advanced features:")
    print("1. Creating vector store from raw texts")
    processor.create_from_texts(texts)
    print(f"Created vector store with {processor.get_collection_count()} chunks")
    
    # Demonstrate MMR search
    question = "Tell me about all-white mushrooms with large fruiting bodies"
    print(f"\n2. MMR Search for: '{question}'")
    results = processor.mmr_search(question, k=2, fetch_k=3)
    for i, doc in enumerate(results, 1):
        print(f"\nMMR Result {i}:")
        print(doc.page_content[:200])
    
    # Demonstrate metadata filtering
    question = "who are the stakeholders of project zero spread?"
    print(f"\n3. Filtered Search for: '{question}'")
    results = processor.similarity_search(
        question,
        k=3,
        filter={"source": "PDF-docs/Engineering4.pdf"}
    )
    for i, doc in enumerate(results, 1):
        print(f"\nFiltered Result {i}:")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(doc.page_content[:200])

def main():
    # Load environment
    load_environment()
    
    # Create processor with custom configuration
    config = VectorStoreConfig(
        chunk_size=800,
        chunk_overlap=100,
        persist_directory=os.path.join(os.path.dirname(current_dir), 'docs/chroma/'),
        embedding_model="text-embedding-ada-002"
    )
    processor = VectorStoreProcessor(config)
    
    try:
        # Load multiple PDF documents
        pdf_dir = os.path.join(os.path.dirname(current_dir), 'PDF-docs')
        pdf_files = [
            os.path.join(pdf_dir, f"Engineering{i}.pdf") for i in range(1, 7)
        ]
        
        # Load all documents
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
        
        # Demonstrate basic usage
        print("\n=== Basic Usage ===")
        demonstrate_basic_usage(processor, docs)
        
        # Demonstrate advanced features
        print("\n=== Advanced Features ===")
        demonstrate_advanced_features(processor)
        
    except VectorStoreProcessingError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
