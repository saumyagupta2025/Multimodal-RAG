# Set Gemini Key, OpenAI Key
# Upload pdfs
from unstructured.partition.pdf import partition_pdf

def process_pdf(uploaded_file):
    """
    Processes an uploaded PDF file and extracts its content using the `partition_pdf` function.
    
    Parameters:
    uploaded_file (UploadedFile): The uploaded PDF file from Streamlit.
    output_path (str): The directory where the file is located. Default is './content/'.
    
    Returns:
    list: A list of extracted chunks from the PDF.
    """
    file_path = uploaded_file.name
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        chunking_strategy='by_title',
        max_characters=8000,
        combine_text_under_n_characters=2000,
        new_after_n_characters=6000
    )
    
    return chunks

def get_texts_from_chunks(chunks):
    texts = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    return texts


def get_images_from_chunks(chunks):
    images = []
    tables = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_ele = chunk.metadata.orig_elements
            for el in chunk_ele:
                if 'Image' in str(type(el)):
                    images.append(el.metadata.image_base64)
                
                elif 'Table' in str(type(el)):
                    tables.append(el)
    return images, tables
