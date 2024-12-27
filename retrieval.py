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

print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"
smalldb.similarity_search(question, k=2)
print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))

question = "what is a zero spread account?"
docs_ss = vectordb.similarity_search(question,k=3)
print(docs_ss[0].page_content[:100])
print(docs_ss[1].page_content[:100])

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
print(docs_mmr[0].page_content[:100])
print(docs_mmr[1].page_content[:100])