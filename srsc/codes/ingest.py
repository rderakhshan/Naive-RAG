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

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

# Initialize ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(
    path="./srsc/database/"
)

# Create or get a collection
collection = chroma_client.get_or_create_collection(
    name="documents_table",
    embedding_function=openai_ef
)

# OpenAI client initialization
client = OpenAI(api_key=openai_api_key)

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

def split_text(text, chunk_size=1000, chunk_overlap=20):
    """
    Split a text string into chunks of specified size with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_openai_embedding(text):
    """
    Generate embeddings using OpenAI API.
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

def process_documents(directory_path: str):
    """
    Load, split, embed, and upsert documents into ChromaDB.
    """
    documents = load_documents_from_directory(directory_path)
    print(f"Loaded {len(documents)} documents from the directory.")

    chunked_documents = []
    for doc_idx, doc in enumerate(documents):
        chunks = split_text(doc)
        print(f"Splitting document {doc_idx+1} into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"doc{doc_idx+1}_chunk{i+1}", "text": chunk})

    embedding_id = 0
    for doc in chunked_documents:
        print(f"==== Generating embeddings of chunk {embedding_id} ====")
        doc["embedding"] = get_openai_embedding(doc["text"])
        embedding_id += 1

    for doc in chunked_documents:
        print("==== Inserting chunks into db ====")
        collection.upsert(
            ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
        )
        # print(ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]])
    print("==== All chunks inserted into the database ====")
    
    # print(chunked_documents)
    return chunked_documents
process_documents("./srsc/docpool")
print("==== Ingest module finished successfully ====")