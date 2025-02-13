# Multimodal RAG: Enhancing Retrieval with Text, Tables & Images

## 📄🔍 Overview
Many documents contain a combination of **text, tables, and images**, making retrieval difficult for traditional RAG (Retrieval-Augmented Generation) systems. Our **Multimodal RAG** overcomes this by leveraging **multimodal embeddings** and **LLMs** to process and retrieve data effectively.

## 🛠️ System Architecture
This system processes **PDF documents** containing **text, tables, and images** to enable efficient information retrieval. It follows these key steps:

- **Document Parsing**: Extracts text, tables, and images from PDFs using the Unstructured library.
- **Multimodal Processing**: Uses an **LLM (e.g., GPT-4o-mini, Gemini-1.5-pro)** to generate summaries for extracted content.
- **Storage & Indexing**: Saves **original content** and **summaries** in a document store, while text embeddings are stored in **VectorDB**.
- **Retrieval & Querying**: Retrieves relevant documents based on **semantic search** and presents the **original text, tables, and images**.

## 🚀 Key Features
- **Supports Multimodal Data**: Handles **text, tables, and images** for better document understanding.
- **Advanced Retrieval**: Uses **semantic search** with text embeddings in **VectorDB**.
- **Summarization & Indexing**: Generates **summaries** for extracted content to enhance retrieval.
- **Efficient Query Processing**: Ensures **faster and more accurate** search results.

![Architecture Diagram](<Architecture Diagram.png>)

## 🖥️ Local Setup Instructions
### 1️⃣ Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- pip
- [Streamlit](https://streamlit.io/)
- [Unstructured] for document parsing
- [ChromaDB] for vector storage (or alternative VectorDB)
- Poppler, Tesseract, and libmagic (for additional PDF/image processing)
  ```bash
  brew install poppler tesseract libmagic
  ```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/saumyagupta2025/Multimodal-RAG.git
cd Multimodal-RAG
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run home.py
```

### 5️⃣ Upload a PDF and Query
- Upload a document containing **text, tables, and images**.
- Ask questions related to the document.
- Get relevant **retrieved content** with LLM-generated responses.

## 📌 Notes
- The app now displays **retrieved documents along with LLM responses**.
- Ensure that **Architecture Diagram.png** is placed in the root directory for proper display.

## 💡 Future Enhancements
- Display the **retrieved documents along with LLM-generated responses**.
- Improve **retrieval quality** using hybrid techniques.
- Support **more file formats** beyond PDFs.

🔎 **Ready to experience the magic of Multimodal RAG? Start now!**
