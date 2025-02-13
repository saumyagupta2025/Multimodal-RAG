from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def summarize_texts(texts, llm):
    """
    Summarizes a list of text chunks.
    
    Parameters:
    texts (list): List of text chunks to summarize.
    
    Returns:
    list: List of summarized text chunks.
    """
    prompt_text = '''
        You are an assistant tasked with summarizing tables and texts.
        Give a concise summary of the table or text.
        
        Respond only with the summary, no additional comments.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        
        Table or text chunk:{element}
    '''
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model='llama-3.1-8b-instant', groq_api_key=GROQ_API_KEY)
    summarize_chain = prompt | llm | StrOutputParser()
    
    return summarize_chain.batch(texts)


def summarize_tables(tables, llm):
    """
    Summarizes a list of tables.
    
    Parameters:
    tables (list): List of tables with metadata containing HTML representations.
    
    Returns:
    list: List of summarized table descriptions.
    """
    tables_html = [table.metadata.text_as_html for table in tables]
    return summarize_texts(tables_html, llm)



# def summarize_images(images_b64):
#     """
#     Summarizes a list of image descriptions.
    
#     Parameters:
#     images_b64 (list): List of base64-encoded image data.
    
#     Returns:
#     list: List of summarized image descriptions.
#     """
#     prompt_template = """ Describe the image in detail. For context,
#     the image is part of the research paper explaining the transformers architecture. 
#     Be specific about graphs, such as bar plots."""
    
#     messages = [(
#         "user",
#         [
#             {"type": "text", "text": prompt_template},
#             {
#                 "type": "image_url",
#                 "image_url": {"url": "data:image/jpeg;base64,{image}"}
#             },
#         ]
#     )]
    
#     prompt = ChatPromptTemplate.from_messages(messages)
#     model = ChatGroq(temperature=0.5, model='llama-3.1-8b-instant', groq_api_key=GROQ_API_KEY)
#     chain = prompt | model | StrOutputParser()
    
#     return chain.batch(images_b64)


def summarize_images(images_b64, llm):
    """
    Summarizes a list of image descriptions.
    
    Parameters:
    images_b64 (list): List of base64-encoded image data.
    
    Returns:
    list: List of summarized image descriptions.
    """
    prompt_template = """Describe the image in detail. For context,
    the image is part of the research paper explaining the transformers architecture. 
    Be specific about graphs, such as bar plots."""
    
    # model = ChatGroq(temperature=0.5, model='llama-3.1-8b-instant', groq_api_key=GROQ_API_KEY)
    summaries = []

    for image in images_b64:
        messages = [(
    "user",
    [
        {"type": "text", "text": prompt_template},
        {
            "type":"image_url",
            "image_url":{"url":"data:image/jpeg;base64,{image}"}
        },
    ]
)]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        
        # summaries.append(chain.invoke({}))  # Invoke the chain properly
        summaries.append(chain.invoke({"image": image}))  #
        # summaries = chain.batch(images_b64)
        print(summaries)
    
    return summaries
