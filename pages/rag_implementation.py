import streamlit as st
import openai
import json
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from helpers.handle_pdf_partition import process_pdf, get_texts_from_chunks, get_images_from_chunks
from helpers.generate_summaries import summarize_texts, summarize_tables, summarize_images
from helpers.loading_data_to_db import store_summaries1
from helpers.rag_pipeline import parse_docs, build_prompt1  # Assuming these are defined elsewhere
# import chromadb
import tempfile
# from helpers.show_context import render_page

# Clear system cache
# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize session state variables
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False

# Function to process uploaded PDF
def process_document(uploaded_file, llm):
    """Processes the uploaded PDF, extracts and summarizes content, and loads embeddings."""
    st.write("🔄 Processing PDF...")

    # Extract content from PDF
    chunks = process_pdf(uploaded_file)
    texts = get_texts_from_chunks(chunks)
    images, tables = get_images_from_chunks(chunks)

    # Summarization
    st.write("📝 Summarizing text, tables, and images...")
    summarized_texts = summarize_texts(texts, llm)
    summarized_tables = summarize_tables(tables, llm)
    summarized_images = summarize_images(images, llm)

    # Store in Vector Store
    st.write("📥 Loading into VectorStore...")
    retriever = store_summaries1(texts, summarized_texts, tables, summarized_tables, images, summarized_images)

    # Store retriever in session state
    st.session_state["retriever"] = retriever
    st.session_state["parsed_docs"] = retriever.invoke("")  # Invoke retriever once and store results
    st.session_state["document_processed"] = True  # Mark document as processed
    
    st.success("✅ Document processed successfully!")
    return retriever

# Function to handle user queries
def query_llm(user_question, llm):
    """Runs the RAG pipeline to answer queries using stored embeddings."""
    if st.session_state["retriever"] is None:
        st.error("❌ No document uploaded! Please upload a PDF first.")
        return "Please upload a document first."

    try:
        # Retrieve relevant documents from the retriever
        retrieved_docs = st.session_state["retriever"].invoke(user_question)

        # Create a properly formatted context dictionary
        context = {
            "context": "\n\n".join(
                [str(doc) if hasattr(doc, '__str__') else str(doc.text) for doc in retrieved_docs]
            ),
            "question": user_question
        }

        # Define the query chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(build_prompt1)
            | llm
            | StrOutputParser()
        )

        # Run the query with the context dictionary
        response = chain.invoke(context)

        # Parse response
        response_data = json.loads(response)

        # Display response
        # st.write("### 🤖 AI Response:")
        # st.markdown(response_data["response"])

        return response_data["response"]

    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        return "Error processing your query."


# Sidebar for API key and file upload
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")

file = st.sidebar.file_uploader("📂 Upload a file", type=["pdf"])

# App Title
st.title("🧠 Multimodal RAG: Your AI-Powered Knowledge Assistant")

# Subheader with a welcoming touch
st.subheader("📄 Upload, Chat, and Explore!")

# Stylish description with spacing for better readability
st.markdown(
    """
    Unlock the power of **Multimodal Retrieval-Augmented Generation (RAG)**!  
 
    🔍 **Ask questions and receive intelligent, context-aware responses** powered by advanced AI.  
    """
)
# Check for API key and file
if not api_key or not file:
    st.warning("⚠️ Please enter an API key and upload a PDF file to start.")
else:
    # Store API Key in session
    st.session_state["api_key"] = api_key

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    # Process document once
    if file and not st.session_state["document_processed"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            st.session_state['file_path'] = tmp_file.name
        
        st.session_state['retriever'] = process_document(file, llm)
        st.session_state["document_processed"] = True
        # st.session_state['retriever'] = process_document(file, llm)

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input handling
    if user_input := st.chat_input("💬 Ask something..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response using query_llm function
        bot_response = query_llm(user_input, llm)

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
