# Naive RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for question-answering. It loads text documents from a directory, splits them into chunks, generates embeddings using OpenAI's `text-embedding-3-small` model, stores them in a ChromaDB vector database, and answers user queries by retrieving relevant chunks and generating responses with OpenAI's `gpt-3.5-turbo` model.

## Features
- Loads text files from a specified directory.
- Splits documents into chunks with configurable size and overlap.
- Generates embeddings using OpenAI's embedding API.
- Stores chunks and embeddings in a persistent ChromaDB database.
- Retrieves relevant document chunks for a query and generates concise answers.

## Prerequisites
- Python 3.8+
- An OpenAI API key (set in a `.env` file)
- A directory containing `.txt` files to process (default: `./srsc/documents`)

## Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install openai chromadb python-dotenv
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Prepare documents**:
   Place your text files (`.txt`) in the `./srsc/documents` directory. Ensure the directory exists, or update the `load_documents_from_directory` path in the code.

## Usage
1. **Run the script**:
   ```bash
   python app.py
   ```
   The script will:
   - Load text files from `./srsc/documents`.
   - Split documents into chunks (default: 1000 characters, 20-character overlap).
   - Generate embeddings using OpenAI's `text-embedding-3-small` model.
   - Store chunks and embeddings in a ChromaDB database at `./srsc/database/chromadb_persistence`.
   - Execute an example query ("tell me about databricks") and print the response.

2. **Query the system**:
   Modify the example query in the script:
   ```python
   question = "your question here"
   relevant_chunks = query_documents(question)
   answer = generate_response(question, relevant_chunks)
   print(answer.content)
   ```
   Run the script again to process your query.

## File Structure
```
naive-rag/
├── .env                    # Environment variables (OPENAI_API_KEY)
├── .gitignore              # Git ignore file (add standard Python ignores)
├── app.py                  # Main script for RAG pipeline
├── srsc/
│   ├── documents/          # Directory for input text files (.txt)
│   ├── database/
│   │   ├── chromadb_persistence/  # ChromaDB persistent storage
└── README.md               # Project documentation
```

## Code Overview
- **Document Loading**: `load_documents_from_directory` reads `.txt` files from a specified directory.
- **Text Chunking**: `split_text` splits documents into chunks with configurable size and overlap.
- **Embedding Generation**: `get_openai_embedding` uses OpenAI's API to generate embeddings for text chunks.
- **Vector Storage**: ChromaDB stores chunks and embeddings persistently.
- **Querying**: `query_documents` retrieves relevant chunks based on a query.
- **Response Generation**: `generate_response` uses OpenAI's `gpt-3.5-turbo` to generate answers from retrieved chunks.

## Example
For a directory with a single text file containing "Databricks is a unified data analytics platform...", the script:
1. Loads the file.
2. Splits it into chunks (e.g., 1 chunk if <1000 characters).
3. Generates embeddings and stores them in ChromaDB.
4. Answers a query like "tell me about databricks" using the retrieved chunk and OpenAI's model.

Example output:
```
Loaded 1 documents from the directory.
Splitting document 1 into 1 chunks
==== Generating embeddings of chunk of 0====
==== Generating embeddings... ====
==== Inserting chunks into db;;; ====
==== All chunks inserted into the database ====
==== Returning relevant chunks ====
Databricks is a unified data analytics platform...
```

## Notes
- Ensure your OpenAI API key is valid and has sufficient quota.
- The ChromaDB database persists data in `./srsc/database/chromadb_persistence`.
- Adjust `chunk_size` and `chunk_overlap` in `split_text` for your use case (e.g., use `tiktoken` for token-based chunking if needed).
- The example query uses `gpt-3.5-turbo`, which can be updated to other models supported by OpenAI.

## Troubleshooting
- **Missing API Key**: Ensure `.env` contains `OPENAI_API_KEY`.
- **Directory Not Found**: Verify `./srsc/documents` exists and contains `.txt` files.
- **ChromaDB Errors**: Check that the persistence path is writable.
- **OpenAI Errors**: Confirm your API key has access to `text-embedding-3-small` and `gpt-3.5-turbo`.

## Future Improvements
- Add token-based chunking for LLM compatibility using `tiktoken`.
- Implement batch embedding generation for efficiency.
- Add error handling for API rate limits and network issues.
- Support additional file formats (e.g., PDF, Markdown).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, contact [Your Name] at [your.email@example.com].