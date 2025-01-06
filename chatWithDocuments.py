import os
import sys
from typing import List

# Add the library directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
library_dir = os.path.join(current_dir, 'library')
sys.path.append(library_dir)

from utils import load_environment
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory

def load_vector_store(persist_directory: str) -> Chroma:
    """
    Load the vector store from the specified directory
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb

def create_chat_chain(vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Create a conversational chain with the vector store
    """
    # Initialize the language model
    llm = ChatOpenAI(
        temperature=0.7,
        model_name='gpt-3.5-turbo'
    )
    
    # Create the prompt template
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        ("human", "{chat_history}\nFollow up: {question}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Use the following pieces of context to answer the user's question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}"""),
        ("human", "{question}")
    ])

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        ),
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=True
    )
    
    return chain

def chat_with_documents():
    """
    Main function to chat with the documents
    """
    # Load environment variables
    load_environment()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Load the vector store
    persist_directory = os.path.join(current_dir, 'docs/chroma')
    vector_store = load_vector_store(persist_directory)
    
    # Create the chat chain
    chat_chain = create_chat_chain(vector_store)
    
    # Initialize chat history as a list of message objects
    chat_history = []
    
    print("\n=== Document Chat System ===")
    print("Chat initialized! Type 'exit' to end the conversation.")
    print("You can now ask questions about the documents.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit command
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_input:
            try:
                # Get the response from the chain
                response = chat_chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })
                
                # Extract and print the answer
                answer = response['answer']
                print("\nAssistant:", answer)
                
                # Update chat history with message objects
                chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])
                
                # Print source documents if available
                if response.get('source_documents'):
                    print("\nSources:")
                    for i, doc in enumerate(response['source_documents'], 1):
                        print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")
                print("\n" + "-"*50 + "\n")
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                continue
if __name__ == "__main__":
    chat_with_documents()
