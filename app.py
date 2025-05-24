import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file")

# initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small")

