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
    embedding_function = openai_ef,)

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


documents = load_documents_from_directory("./srsc/documents")
print(f"Loaded {len(documents)} documents from the directory.")

# Function to split a text string into chunks of specified size with overlap
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    # Continue splitting until the start index reaches the text length
    while start < len(text):
        # Calculate the end index for the current chunk
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Initialize an empty list to store chunked documents as dictionaries
chunked_documents = []
# Iterate over the documents list with an index for unique document IDs
for doc_idx, doc in enumerate(documents):  # Use index to create document ID
    # Split the current document (string) into chunks using split_text
    chunks = split_text(doc)
    # Print the number of chunks for the current document
    print(f"Splitting document {doc_idx+1} into {len(chunks)} chunks")
    # Add the number of chunks for this document to the total
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"doc{doc_idx+1}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = client.embeddings.create(input = text, model = "text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

embedding_id = 0
# Generate embeddings for the document chunks
for doc in chunked_documents:
    print(f"==== Generating embeddings of chunk of {embedding_id}====")
    doc["embedding"] = get_openai_embedding(doc["text"])
    embedding_id = embedding_id + 1

# print(doc["embedding"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids = [doc["id"]], documents = [doc["text"]], embeddings = [doc["embedding"]])

print("==== All chunks inserted into the database ====")

# Function to query documents
def query_documents(question, n_results = 2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts = question, n_results = n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer.content)