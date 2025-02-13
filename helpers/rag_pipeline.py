from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from base64 import b64decode


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def build_prompt1(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if "texts" in docs_by_type and docs_by_type["texts"]:
        for text_element in docs_by_type["texts"]:
            # Ensure it's a string
            context_text += text_element if isinstance(text_element, str) else str(text_element)

    # Construct the prompt
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
    Context: {context_text}
    Question: {user_question}
    
    Please provide a detailed answer and include the context used in your response in the following format:
    {{
        "response": "<your answer>",
        "context": {{
            "texts": [
                {{"text": "<context text>", "metadata": {{"page_number": <page number>}}}},
                ...
            ]
        }}
    }}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if "images" in docs_by_type and docs_by_type["images"]:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])
