"""
YouTube Processing Library for LangChain applications.
This module provides a reusable interface for processing YouTube video transcripts with various options
for text extraction, chunking, and vector store integration.
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document

@dataclass
class TextSplitterConfig:
    """Configuration for text splitting parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separator: str = ' '

class YouTubeProcessingError(Exception):
    """Custom exception for YouTube processing errors."""
    pass

class YouTubeProcessor:
    """
    A class to handle YouTube video transcript processing with various configuration options.
    
    This class provides methods to:
    - Load YouTube video transcripts
    - Split text into chunks
    - Process multiple videos
    - Configure text splitting parameters
    """
    
    def __init__(self, splitter_config: Optional[TextSplitterConfig] = None):
        """
        Initialize the YouTube processor with optional configuration.
        
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
    
    def load_video(self, video_url: str) -> List[Document]:
        """
        Load a YouTube video transcript and return it as documents.
        
        Args:
            video_url (str): URL of the YouTube video.
            
        Returns:
            List[Document]: List of documents containing the transcript.
            
        Raises:
            YouTubeProcessingError: If the video cannot be loaded or processed.
        """
        try:
            loader = YoutubeLoader.from_youtube_url(video_url)
            return loader.load()
        except Exception as e:
            raise YouTubeProcessingError(f"Error loading YouTube video {video_url}: {str(e)}")
    
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
            YouTubeProcessingError: If documents cannot be split.
        """
        try:
            splitter = self.recursive_splitter if use_recursive else self.char_splitter
            return splitter.split_documents(documents)
        except Exception as e:
            raise YouTubeProcessingError(f"Error splitting documents: {str(e)}")
    
    def process_video(self, 
                     video_url: str, 
                     use_recursive: bool = True) -> List[Document]:
        """
        Load and process a YouTube video transcript in one step.
        
        Args:
            video_url (str): URL of the YouTube video.
            use_recursive (bool): Whether to use recursive splitter.
            
        Returns:
            List[Document]: List of processed document chunks.
            
        Raises:
            YouTubeProcessingError: If the video cannot be processed.
        """
        documents = self.load_video(video_url)
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
    processor = YouTubeProcessor()
    
    try:
        # Process a YouTube video
        video_url = "https://www.youtube.com/watch?v=pJY0mBWHPw4"
        chunks = processor.process_video(video_url)
        print(f"Successfully processed video into {len(chunks)} chunks")
        
        # Update configuration
        new_config = TextSplitterConfig(chunk_size=500, chunk_overlap=50)
        processor.update_config(new_config)
        
        # Process with new configuration
        chunks = processor.process_video(video_url, use_recursive=False)
        print(f"Successfully processed video into {len(chunks)} chunks with new config")
        
    except YouTubeProcessingError as e:
        print(f"Error processing video: {str(e)}")
