from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, CharacterTextSplitter

def split_text_by_page(input_file_path):
    """
    Splits text from a file into pages.

    Args:
        input_file_path (str): The path to the text file to be split.

    Returns:
        List[str]: The text split into pages.
    """
    text_loader = TextLoader(input_file_path)
    splits = text_loader.load_and_split()
    print("Length of splits: " + str(len(splits)))
    return splits

def split_pdf_by_page(input_file_path):
    """
    Splits text from a PDF file into pages.

    Args:
        input_file_path (str): The path to the PDF file to be split.

    Returns:
        List[str]: The text split into pages.
    """
    text_loader = PyPDFLoader(input_file_path)
    splits = text_loader.load_and_split()
    print("Length of splits: " + str(len(splits)))
    return splits

def split_text_simple(text, chunk_size=2000, chunk_overlap=200):
    """
    Splits text into chunks of a specified size with optional overlap.

    Args:
        text (str): The text to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to 2000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 200.

    Returns:
        List[str]: The text split into chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    splits = text_splitter.create_documents([text])
    print("Length of splits: " + str(len(splits)))
    return splits

def split_markdown_headings(markdown_text):
    """
    Splits markdown text into sections based on headings.

    Args:
        markdown_text (str): The markdown text to be split.

    Returns:
        List[str]: The sections of the markdown text.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    splits = text_splitter.split_text(markdown_text)

    print("Length of splits: " + str(len(splits)))
    return splits
