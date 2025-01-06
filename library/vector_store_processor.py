"""
Vector Store Processing Library for LangChain applications.
This module provides a reusable interface for processing documents into vector stores
with various options for embedding, storage, and retrieval.
"""

import os
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils import load_environment

@dataclass
class VectorStoreConfig:
    """Configuration for vector store parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    persist_directory: str = 'docs/chroma/'
    embedding_model: str = "text-embedding-ada-002"

class VectorStoreProcessingError(Exception):
    """Custom exception for vector store processing errors."""
    pass

class VectorStoreProcessor:
    """
    A class to handle document processing and vector store operations.
    
    This class provides methods to:
    - Process documents into chunks
    - Generate embeddings
    - Create and manage vector stores
    - Perform similarity searches
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the vector store processor with optional configuration.
        
        Args:
            config (VectorStoreConfig, optional): Configuration for processing.
                If not provided, default values will be used.
        """
        # Load environment variables
        env = load_environment()
        self.config = config or VectorStoreConfig()
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize text splitter and embedding components."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.embedding = OpenAIEmbeddings(
            model=self.config.embedding_model
        )
        
        # Ensure persist directory exists
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        self.vectordb = None
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents into chunks suitable for embedding.
        
        Args:
            documents (List[Document]): List of documents to process.
            
        Returns:
            List[Document]: List of processed document chunks.
            
        Raises:
            VectorStoreProcessingError: If documents cannot be processed.
        """
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise VectorStoreProcessingError(f"Error processing documents: {str(e)}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from processed documents.
        
        Args:
            documents (List[Document]): List of documents to store.
            
        Raises:
            VectorStoreProcessingError: If vector store cannot be created.
        """
        try:
            splits = self.process_documents(documents)
            self.vectordb = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding,
                persist_directory=self.config.persist_directory
            )
        except Exception as e:
            raise VectorStoreProcessingError(f"Error creating vector store: {str(e)}")
    
    def similarity_search(self, 
                        query: str, 
                        k: int = 3,
                        filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query (str): The query string to search for.
            k (int): Number of results to return.
            filter (Dict[str, Any], optional): Metadata filter for the search.
            
        Returns:
            List[Document]: List of similar documents.
            
        Raises:
            VectorStoreProcessingError: If search cannot be performed.
        """
        if not self.vectordb:
            raise VectorStoreProcessingError("No vector store available. Create one first.")
        
        try:
            return self.vectordb.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            raise VectorStoreProcessingError(f"Error performing similarity search: {str(e)}")
    
    def mmr_search(self,
                  query: str,
                  k: int = 3,
                  fetch_k: int = 10,
                  filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search on the vector store.
        MMR optimizes for diversity in results while maintaining relevance.
        
        Args:
            query (str): The query string to search for.
            k (int): Number of results to return.
            fetch_k (int): Number of results to fetch before reranking.
            filter (Dict[str, Any], optional): Metadata filter for the search.
            
        Returns:
            List[Document]: List of documents selected by MMR.
            
        Raises:
            VectorStoreProcessingError: If search cannot be performed.
        """
        if not self.vectordb:
            raise VectorStoreProcessingError("No vector store available. Create one first.")
        
        try:
            return self.vectordb.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, filter=filter
            )
        except Exception as e:
            raise VectorStoreProcessingError(f"Error performing MMR search: {str(e)}")
    
    def create_from_texts(self, texts: List[str]) -> None:
        """
        Create a vector store directly from a list of texts.
        
        Args:
            texts (List[str]): List of text strings to store.
            
        Raises:
            VectorStoreProcessingError: If vector store cannot be created.
        """
        try:
            self.vectordb = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding,
                persist_directory=self.config.persist_directory
            )
        except Exception as e:
            raise VectorStoreProcessingError(f"Error creating vector store from texts: {str(e)}")
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            int: Number of documents in the collection.
            
        Raises:
            VectorStoreProcessingError: If count cannot be retrieved.
        """
        if not self.vectordb:
            raise VectorStoreProcessingError("No vector store available. Create one first.")
        
        try:
            return self.vectordb._collection.count()
        except Exception as e:
            raise VectorStoreProcessingError(f"Error getting collection count: {str(e)}")
    
    def update_config(self, new_config: VectorStoreConfig) -> None:
        """
        Update the processor configuration.
        
        Args:
            new_config (VectorStoreConfig): New configuration to use.
        """
        self.config = new_config
        self._init_components()

# Example usage:
if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFLoader
    
    processor = VectorStoreProcessor()
    
    try:
        # Load sample documents
        loaders = [
            PyPDFLoader("PDF-docs/CFDs - General.pdf"),
            PyPDFLoader("PDF-docs/CFDs - Zero Spread.pdf")
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        # Create vector store
        processor.create_vector_store(docs)
        print(f"Created vector store with {processor.get_collection_count()} documents")
        
        # Perform similarity search
        results = processor.similarity_search("what is a zero spread account?")
        print("\nSearch results:")
        for doc in results:
            print(f"\n{doc.page_content[:200]}...")
        
    except VectorStoreProcessingError as e:
        print(f"Error: {str(e)}")
