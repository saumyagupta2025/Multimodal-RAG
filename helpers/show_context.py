import fitz
from langchain_core.documents import Document
import matplotlib.patches as pataches
import matplotlib.pyplot as plt
from PIL import Image


def plot_pdf_with_boxes(pdf_page, segments):
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categorites = set()
    category_to_color = {
    'Title': 'orchid',
    'Image':'forestgreen',
    'Table':'tomato',
    }
    for segment in segments:
        points = segment['coordinates']['points']
        layout_width = segment["coordinates"]['layout_width']
        layout_height = segment['coordinates']['layout_height']
        scaled_points = [
        (x * pix.width / layout_width, y * pix.height / layout_height)
        for x, y in points
        ]
        box_color = category_to_color.get(segment['category'], 'deepskyblue')
        categorites.add(segment['category'])
        rect = pataches.Polygon(
        scaled_points, linewidth=1, edgecolor=box_color, facecolor='none'
        )
        ax.add_patch(rect)

    #Legend
    legend_handles = [pataches.Patch(color='deepskyblue', label='Text')]
    for category in ['Title', 'Image', 'Table']:
        if category in categorites:
            legend_handles.append(
            pataches.Patch(color=category_to_color[category],label=category))
    ax.axis('off')
    ax.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    plt.show()

def render_page(file_path: str, doc_list: list, page_number: int, print_text=True) -> None:
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get('page_number') == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page=pdf_page, segments=segments)
    if print_text:
        for doc in page_docs:
            print(f'{doc.page_content}\n')


def extract_page_numbers_from_chunk(chunk):
    elements = chunk.metadata.orig_elements
    page_numbers = set()
    for element in elements:
        page_numbers. add (element.metadata.page_number)
    return page_numbers


def display_chunk_pages (chunk):
    page_numbers = extract_page_numbers_from_chunk(chunk)
    docs = []
    for element in chunk.metadata.orig_elements:
        metadata = element.metadata.to_dict()
        if "Table" in str(type (element)):
            metadata ["category"] = "Table"
        elif "Image" in str(type(element) ):
            metadata ["category"] = "Image"
        else:
            metadata ["category"] = "Text"
        metadata ["page_number"] = int (element.metadata.page_number)
        
        docs. append (Document( page_content=element.text, metadata=metadata))
    
    for page_number in page_numbers:
        render_page(docs, page_number, False)
