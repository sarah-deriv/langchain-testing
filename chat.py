import os
import openai
import sys
sys.path.append('../..')

# import panel as pn  # GUI
# pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = "gpt-4"

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-artistic-push-97"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Initialize LLM
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.

Previous conversation:
{chat_history}

Context:
{context}

Query: {query}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "query", "chat_history"],
    template=template,
)

# Initialize memory  (Updated to avoid deprecation warning)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Create chain with memory
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={
#         "prompt": QA_CHAIN_PROMPT,
#         "memory": memory
#     }
# )

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)
# Interactive chat loop
print("Chat initialized! Type 'exit' to end the conversation.")
while True:
    question = input("\nYour question: ")
    if question.lower() == 'exit':
        break
        
    result = qa_chain.invoke({"query": question})  # Match the prompt's input variable name
    print("\nAnswer:", result["result"])
