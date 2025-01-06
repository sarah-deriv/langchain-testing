"""
PDF Processing Library for LangChain applications.
This module provides a reusable interface for processing PDF documents with various options
for text extraction, chunking, and vector store integration.
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass
import os.path

from langchain_community.document_loaders import PyPDFLoader
from .utils import load_environment
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document

@dataclass
class TextSplitterConfig:
    """Configuration for text splitting parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separator: str = ' '

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

class PDFProcessor:
    """
    A class to handle PDF document processing with various configuration options.
    
    This class provides methods to:
    - Load PDF documents
    - Split text into chunks
    - Process multiple PDFs
    - Configure text splitting parameters
    """
    
    def __init__(self, splitter_config: Optional[TextSplitterConfig] = None):
        self.pdf_base_path = load_environment()['pdf_files_path']
        """
        Initialize the PDF processor with optional configuration.
        
        Args:
            splitter_config (TextSplitterConfig, optional): Configuration for text splitting.
                If not provided, default values will be used.
        """
        self.config = splitter_config or TextSplitterConfig()
        self._init_splitters()
    
    def _init_splitters(self) -> None:
        """Initialize text splitters with current configuration."""
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.char_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator=self.config.separator
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return its pages as documents.
        
        Args:
            pdf_path (str): Path to the PDF file relative to PDF_FILES_PATH.
            
        Returns:
            List[Document]: List of documents, one per page.
            
        Raises:
            PDFProcessingError: If the PDF cannot be loaded or processed.
        """
        try:
            full_path = os.path.join(self.pdf_base_path, pdf_path)
            if not os.path.exists(full_path):
                raise PDFProcessingError(f"PDF file not found: {full_path}")
            
            loader = PyPDFLoader(full_path)
            return loader.load()
        except Exception as e:
            raise PDFProcessingError(f"Error loading PDF {pdf_path}: {str(e)}")
    
    def split_documents(self, 
                       documents: List[Document], 
                       use_recursive: bool = True) -> List[Document]:
        """
        Split documents into chunks using specified splitter.
        
        Args:
            documents (List[Document]): List of documents to split.
            use_recursive (bool): Whether to use recursive splitter (True) or
                                character splitter (False).
                                
        Returns:
            List[Document]: List of split documents.
            
        Raises:
            PDFProcessingError: If documents cannot be split.
        """
        try:
            splitter = self.recursive_splitter if use_recursive else self.char_splitter
            return splitter.split_documents(documents)
        except Exception as e:
            raise PDFProcessingError(f"Error splitting documents: {str(e)}")
    
    def process_pdf(self, 
                   pdf_path: str, 
                   use_recursive: bool = True) -> List[Document]:
        """
        Load and process a PDF file in one step.
        
        Args:
            pdf_path (str): Path to the PDF file.
            use_recursive (bool): Whether to use recursive splitter.
            
        Returns:
            List[Document]: List of processed document chunks.
            
        Raises:
            PDFProcessingError: If the PDF cannot be processed.
        """
        documents = self.load_pdf(pdf_path)
        return self.split_documents(documents, use_recursive)
    
    def update_config(self, new_config: TextSplitterConfig) -> None:
        """
        Update the text splitter configuration.
        
        Args:
            new_config (TextSplitterConfig): New configuration to use.
        """
        self.config = new_config
        self._init_splitters()

# Example usage:
if __name__ == "__main__":
    # Create processor with default configuration
    processor = PDFProcessor()
    
    try:
        # Process a single PDF
        chunks = processor.process_pdf("KYC_Data.pdf")  # Now relative to PDF_FILES_PATH
        print(f"Successfully processed PDF into {len(chunks)} chunks")
        
        # Update configuration
        new_config = TextSplitterConfig(chunk_size=500, chunk_overlap=50)
        processor.update_config(new_config)
        
        # Process with new configuration
        chunks = processor.process_pdf("KYC_Data.pdf", use_recursive=False)  # Now relative to PDF_FILES_PATH
        print(f"Successfully processed PDF into {len(chunks)} chunks with new config")
        
    except PDFProcessingError as e:
        print(f"Error processing PDF: {str(e)}")
