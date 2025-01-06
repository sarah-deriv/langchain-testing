import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The Engineering handbook the chunk is from, should be one of `docs/Engineering1.pdf`, `docs/Engineering2.pdf`, or `docs/Engineering3.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the Engineering handbook",
        type="integer",
    ),
]

document_content_description = "Engineering handbook"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
question = "list down Key Point of Contacts in engineering"
docs = retriever.invoke(question)
for d in docs:
    print(d.metadata)

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
compressed_docs = compression_retriever.invoke(question)
# pretty_print_docs(compressed_docs)

## using mmr
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)
compressed_docs = compression_retriever.invoke(question)
pretty_print_docs(compressed_docs)