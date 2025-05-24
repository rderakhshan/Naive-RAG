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
openai_ef      = embedding_functions.OpenAIEmbeddingFunction(
    api_key    = openai_api_key,
    model_name = "text-embedding-3-small")

# Initialize ChromaDB client with persistance
chroma_client = chromadb.PersistentClient(
    path               = "./srsc/database/chromadb_persistence",)

# Create or get a collection
collection             = chroma_client.get_or_create_collection(
    name               = "documents_table",
    embedding_function = openai_ef,
)

# OpenAI client initialization
client = OpenAI(api_key = openai_api_key)

# response = client.chat.completions.create(
#     model = "gpt-3.5-turbo",
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France?"}
#     ]
# )

# print(response.choices[0].message.content)

def load_documents_from_directory(directory_path: str) -> list:
    """
    Load all text files from a directory and return their contents as a list.
    """
    documents = []
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())
    return documents

existing_documents = load_documents_from_directory("./srsc/documents")
print(f"Loaded {len(existing_documents)} documents from the directory.")