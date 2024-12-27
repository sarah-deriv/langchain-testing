import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("PDF-docs/Engineering1.pdf"),
    PyPDFLoader("PDF-docs/Engineering2.pdf"),
    PyPDFLoader("PDF-docs/Engineering3.pdf"),
    PyPDFLoader("PDF-docs/Engineering4.pdf"),
    PyPDFLoader("PDF-docs/Engineering5.pdf"),
    PyPDFLoader("PDF-docs/Engineering6.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
splits = text_splitter.split_documents(docs)
print(len(splits))

# Embedding
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# sentence1 = "i like dogs"
# sentence2 = "i like canines"
# sentence3 = "the weather is ugly outside"

# embedding1 = embedding.embed_query(sentence1)
# embedding2 = embedding.embed_query(sentence2)
# embedding3 = embedding.embed_query(sentence3)

# import numpy as np
# print(np.dot(embedding1, embedding2))
# print(np.dot(embedding1, embedding3))
# print(np.dot(embedding3, embedding2))

from langchain_community.vectorstores import Chroma
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
question = "what is a zero spread account?"
docs = vectordb.similarity_search(question,k=3)
len(docs)
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)
