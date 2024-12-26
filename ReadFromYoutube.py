import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import YoutubeLoader

url = "https://www.youtube.com/watch?v=pJY0mBWHPw4"

# Load transcript directly from YouTube
loader = YoutubeLoader.from_youtube_url(url)
docs = loader.load()

# Print the first 500 characters of the transcript
print(docs[0].page_content[0:500])
