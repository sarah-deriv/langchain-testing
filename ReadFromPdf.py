import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')


from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("PDF-docs/KYC_Data.pdf")
pages = loader.load()

len(pages)
page = pages[0]
# print(page.page_content[0:500])

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size =1000
chunk_overlap = 150
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator= ' '
)


print(r_splitter.split_documents(pages))
print("""===============""")
print(c_splitter.split_documents(pages))


