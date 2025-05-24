import streamlit as st
from srsc.codes.naiverag import query_documents, generate_response # process_documents

# Set page title
st.title("RAG Chat")

# Initialize session state for document processing and chat history
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Directory path for documents (adjust as needed)
directory_path = "./srsc/docpool"  # Replace with your actual directory path

# Process documents once when the app starts
# if not st.session_state.documents_processed:
#     with st.spinner("Processing documents..."):
#         try:
#             process_documents(directory_path)
#             st.session_state.documents_processed = True
#             st.success("Documents processed successfully!")
#         except Exception as e:
#             st.error(f"Error processing documents: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input for user queries
if question := st.chat_input("Ask a question (e.g., Tell me about Databricks):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Process query and generate response
    with st.spinner("Generating answer..."):
        try:
            # Query documents and generate response
            relevant_chunks = query_documents(question, n_results=2)
            answer = generate_response(question, relevant_chunks)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
        except Exception as e:
            error_message = f"Error generating answer: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)