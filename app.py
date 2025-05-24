from srsc.codes.naiverag import process_documents, query_documents, generate_response

# Process documents from the directory
process_documents("./srsc/documents")

# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(f"Answer: {answer}")