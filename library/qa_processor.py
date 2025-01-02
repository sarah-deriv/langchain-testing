"""
Question Answering Processing Library for LangChain applications.
This module provides a reusable interface for question answering with various
chain types and customizable prompts.
"""

import os
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

@dataclass
class QAConfig:
    """Configuration for QA processing parameters."""
    model_name: str = "gpt-4"
    temperature: float = 0
    chain_type: str = "stuff"  # One of: stuff, map_reduce, refine
    return_source_documents: bool = True

class QAProcessingError(Exception):
    """Custom exception for QA processing errors."""
    pass

class DefaultPrompts:
    """Default prompt templates for QA."""
    CONCISE = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    Question: {question}
    Helpful Answer:"""

    DETAILED = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide a detailed explanation with examples if possible.
    Include relevant quotes from the context if available.
    
    {context}
    Question: {question}
    Detailed Answer:"""

class QAProcessor:
    """
    A class to handle question answering operations with various configurations.
    
    This class provides methods to:
    - Configure QA parameters
    - Use different chain types
    - Customize prompts
    - Process questions and return answers
    """
    
    def __init__(self, 
                 retriever: BaseRetriever,
                 config: Optional[QAConfig] = None):
        """
        Initialize the QA processor with a retriever and optional configuration.
        
        Args:
            retriever (BaseRetriever): The retriever to use for finding relevant documents.
            config (QAConfig, optional): Configuration for QA processing.
                If not provided, default values will be used.
        """
        self.config = config or QAConfig()
        self.retriever = retriever
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize LLM and default chain."""
        try:
            self.llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature
            )
            
            # Initialize with default chain
            self._create_chain()
            
        except Exception as e:
            raise QAProcessingError(f"Error initializing QA components: {str(e)}")
    
    def _create_chain(self, prompt_template: Optional[str] = None) -> None:
        """
        Create a QA chain with optional custom prompt.
        
        Args:
            prompt_template (str, optional): Custom prompt template to use.
        """
        try:
            chain_type_kwargs = {}
            if prompt_template:
                chain_type_kwargs["prompt"] = PromptTemplate.from_template(prompt_template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.retriever,
                chain_type=self.config.chain_type,
                return_source_documents=self.config.return_source_documents,
                chain_type_kwargs=chain_type_kwargs
            )
            
        except Exception as e:
            raise QAProcessingError(f"Error creating QA chain: {str(e)}")
    
    def ask(self, 
            question: str,
            prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a question and return the answer.
        
        Args:
            question (str): The question to answer.
            prompt_template (str, optional): Custom prompt template for this question.
                If provided, a new chain will be created with this prompt.
            
        Returns:
            Dict[str, Any]: Result containing answer and optionally source documents.
            
        Raises:
            QAProcessingError: If question cannot be processed.
        """
        try:
            if prompt_template:
                self._create_chain(prompt_template)
            
            return self.qa_chain.invoke({"query": question})
            
        except Exception as e:
            raise QAProcessingError(f"Error processing question: {str(e)}")
    
    def update_config(self, new_config: QAConfig) -> None:
        """
        Update the QA configuration.
        
        Args:
            new_config (QAConfig): New configuration to use.
        """
        self.config = new_config
        self._init_components()

# Example usage:
if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Initialize vector store
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    
    # Create QA processor
    processor = QAProcessor(vectordb.as_retriever())
    
    try:
        # Ask a question with default settings
        question = "What is a swap-free account?"
        result = processor.ask(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['result']}")
        
        # Ask with custom prompt
        custom_prompt = """Answer the question based on the context provided.
        Be very brief and use simple language.
        {context}
        Question: {question}
        Simple Answer:"""
        
        result = processor.ask(question, prompt_template=custom_prompt)
        print(f"\nWith custom prompt:")
        print(f"Answer: {result['result']}")
        
        # Try different chain type
        new_config = QAConfig(chain_type="refine")
        processor.update_config(new_config)
        
        result = processor.ask(question)
        print(f"\nWith refine chain:")
        print(f"Answer: {result['result']}")
        
    except QAProcessingError as e:
        print(f"Error: {str(e)}")
