import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

import datetime
llm_name = "gpt-4"
print(llm_name)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

persist_directory = os.path.join(os.path.dirname(__file__), 'docs/chroma/')
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain.invoke({"query": question})
print("\nAnswer:", result["result"])

# ================================

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "what are the kyc types"
result = qa_chain.invoke({"query": question})
print(result["result"])
print(result["source_documents"][0])