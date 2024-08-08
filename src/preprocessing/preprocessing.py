import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
import os
import re
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat
from openai import AzureOpenAI
import json


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_word(docx_path):
    """
    Extracts text from a Word document.

    Args:
        docx_path (str): The path to the Word document (.docx).

    Returns:
        str: The extracted text from the Word document.
    """
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def extract_text_from_html(html_path):
    """
    Extracts text from an HTML file.

    Args:
        html_path (str): The path to the HTML file.

    Returns:
        str: The extracted text from the HTML file.
    """
    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
    return text


def convert_markdown_headings(markdown_text):
    """
    Converts markdown headings to a unified format.

    Args:
        markdown_text (str): The markdown text.

    Returns:
        str: The markdown text with unified heading formats.
    """
    # Convert "===" headers to "#"
    markdown_text = re.sub(r'^(.*?)\n={3,}$', r'# \1', markdown_text, flags=re.MULTILINE)

    # Convert "---" headers to "##"
    markdown_text = re.sub(r'^(.*?)\n-{3,}$', r'## \1', markdown_text, flags=re.MULTILINE)

    return markdown_text


def analyze_layout_text_only(input_file_path, output_folder, doc_intelligence_endpoint, doc_intelligence_key):
    """
    Analyzes the layout of a document and extracts figures along with their descriptions.

    Args:
        input_file_path (str): The path to the input document file.
        output_folder (str): The path to the output folder where the updated markdown content will be saved.
        doc_intelligence_endpoint (str): The endpoint for the Azure Document Intelligence API.
        doc_intelligence_key (str): The key for the Azure Document Intelligence API.

    Returns:
        str: The updated markdown content with figure descriptions.
    """
    client = DocumentIntelligenceClient(
        endpoint=doc_intelligence_endpoint, 
        credential=AzureKeyCredential(doc_intelligence_key),
        headers={"x-ms-useragent": "sample-code-figure-understanding/1.0.0"}
    )

    with open(input_file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", analyze_request=f, content_type="application/octet-stream", output_content_format=ContentFormat.MARKDOWN 
        )

    result = poller.result()
    md_content = convert_markdown_headings(result.content)
    
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file_path))[0]}.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return md_content


def correct_text_gpt(text, azure_endpoint, api_key, api_version, model):
    """
    Uses GPT to correct text for grammar, OCR errors, and inconsistencies.

    Args:
        text (str): The text to be corrected.
        azure_endpoint (str): The endpoint for the Azure OpenAI service.
        api_key (str): The API key for the Azure OpenAI service.
        api_version (str): The API version for the Azure OpenAI service.
        model (str): The model name for the Azure OpenAI service.

    Returns:
        str: The corrected text.
    """
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint, 
        api_key=api_key,  
        api_version=api_version
    )

    system_message = """
    # Your Role
    You are an excellent AI assistant for proofreading text data. Your task is to ensure the provided text data is of high quality. You are only allowed to proofread. Adding or removing context from the original document is not allowed. Additionally, you cannot change the structure of the document.

    # Examples of Corrections
    - Grammar errors and typos
    - OCR misrecognitions
    - Inconsistencies in terminology and expressions

    # Your input
    text: 
    """
    
    message_text = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text}
    ]
    
    completion = client.chat.completions.create(
        model=model, 
        messages=message_text,
        temperature=0,
    )
    
    return completion.choices[0].message.content


def clean_text(text):
    """
    Cleans text by removing URLs, HTML tags, and duplicate lines.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove duplicate lines
    lines = text.split("\n")
    unique_lines = list(dict.fromkeys(lines))
    return "\n".join(unique_lines)
