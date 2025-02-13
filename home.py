import streamlit as st
from PIL import Image

# Set Page Configuration
st.set_page_config(page_title="Multimodal RAG", layout="wide")

# Title and Introduction
st.title("📄🔍 Multimodal RAG: Enhancing Retrieval with Text, Tables & Images")
st.write("""
Many documents contain a combination of **text, tables, and images**, making retrieval difficult for traditional RAG (Retrieval-Augmented Generation) systems.  
Our **Multimodal RAG** overcomes this by leveraging **multimodal embeddings** and **LLMs** to process and retrieve data effectively.  
""")

# Architecture Explanation
st.header("🛠️ System Architecture")
st.write("""
This system processes **PDF documents** containing **text, tables, and images** to enable efficient information retrieval.  
It follows these key steps:
- **Document Parsing**: Extracts text, tables, and images from PDFs using the Unstructured library.
- **Multimodal Processing**: Uses an **LLM (e.g., GPT-4o-mini, Gemini-1.5-pro)** to generate summaries for extracted content.
- **Storage & Indexing**: Saves **original content** and **summaries** in a document store, while text embeddings are stored in **VectorDB**.
- **Retrieval & Querying**: Retrieves relevant documents based on **semantic search** and presents the **original text, tables, and images**.
""")

# Display Architecture Diagram
st.image(Image.open("Architecture Diagram.png"), caption="Multimodal RAG Architecture", use_column_width=True)

# Features Section
st.header("🚀 Key Features")
st.write("""
- **Supports Multimodal Data**: Handles **text, tables, and images** for better document understanding.
- **Advanced Retrieval**: Uses **semantic search** with text embeddings in **VectorDB**.
- **Summarization & Indexing**: Generates **summaries** for extracted content to enhance retrieval.
- **Efficient Query Processing**: Ensures **faster and more accurate** search results.
- **Future Enhancements**: We plan to explore **multimodal embeddings (CLIP) and hybrid retrieval techniques**.
""")

st.write("🔎 Ready to experience the magic of Multimodal RAG?")
if st.button("🚀 Get Started"):
    st.switch_page("./pages/rag_implementation.py")  # Ensure your app follows multi-page Streamlit structure


# st.write(f"SQLite Version: {sqlite3.sqlite_version}")
st.markdown(
    "<div style='position: fixed; bottom: 10px; width: 100%; color: gray;'>"
    "⚠️ Note: Currently, the app does not show retrieved documents but only the LLM response. This feature will be added in future updates."
    "</div>",
    unsafe_allow_html=True
)
# st.footer("⚠️ Note: Currently, the app does not show retrieved documents but only the LLM response. This feature will be added in future updates.")