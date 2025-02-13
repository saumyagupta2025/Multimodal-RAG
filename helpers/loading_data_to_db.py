import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os

def store_summaries(texts, text_summaries, tables, table_summaries, images, image_summaries, llm):
    """
    Stores summarized texts, tables, and images in a vector store and document store.
    
    Parameters:
    texts (list): Original text chunks.
    text_summaries (list): Summarized text chunks.
    tables (list): Original tables.
    table_summaries (list): Summarized tables.
    images_b64 (list): Base64-encoded images.
    image_summaries (list): Summarized image descriptions.
    embedding_model (str): The model used for embedding.
    
    Returns:
    MultiVectorRetriever: A retriever for searching across stored summaries.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_model, persist_directory=None)


    # vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=llm)
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    
    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
    
    return retriever


def store_summaries1(texts, text_summaries, tables, table_summaries, images, image_summaries):
    """
    Stores summarized texts, tables, and images in a vector store and document store.
    
    Parameters:
    texts (list): Original text chunks.
    text_summaries (list): Summarized text chunks.
    tables (list): Original tables.
    table_summaries (list): Summarized tables.
    images (list): Images (Base64-encoded).
    image_summaries (list): Summarized image descriptions.
    llm: LLM model for embedding (not used here).
    
    Returns:
    MultiVectorRetriever: A retriever for searching across stored summaries.
    """
    
    # Ensure API Key is available
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OpenAI API Key. Set the OPENAI_API_KEY environment variable.")
    
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Initialize vector store
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_model, persist_directory=None)
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )

    # Function to check and add documents safely
    def add_documents_safe(doc_list, original_data, label):
        if not doc_list:
            print(f"⚠️ Warning: No {label} summaries provided, skipping storage.")
            return

        # Generate unique IDs
        doc_ids = [str(uuid.uuid4()) for _ in original_data]

        # Convert to Document format
        docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(doc_list)
        ]

        # Check if embeddings are generated
        embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
        if not embeddings or not all(embeddings):
            raise ValueError(f"❌ Error: Embeddings could not be generated for {label}.")

        print(f"✅ Successfully generated embeddings for {label} ({len(embeddings)} items).")

        # Add to vector store & document store
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, original_data)))

    # Add summaries to the store
    add_documents_safe(text_summaries, texts, "Text Summaries")
    add_documents_safe(table_summaries, tables, "Table Summaries")
    add_documents_safe(image_summaries, images, "Image Summaries")

    return retriever
