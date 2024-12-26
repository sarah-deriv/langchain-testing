import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("PDF-docs/KYC_Data.pdf")
pages = loader.load()

len(pages)
page = pages[0]
print(page.page_content[0:500])
page.metadata
