import sys, pickle, os, cohere
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain import VectorDBQA
from langchain.llms import Cohere, OpenAI
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch
from qdrant_client.http import models
import requests, re
import json,os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

HOST_URL_QDRANT = os.getenv("CLUSTER_URL_QDRANT")
API_KEY_QDRANT = os.getenv("API_KEY_QDRANT")

print(HOST_URL_QDRANT)
print(API_KEY_QDRANT)

def init_qdrant_client():
    qdrant_client = QdrantClient(
        url=HOST_URL_QDRANT, api_key=API_KEY_QDRANT, prefer_grpc=False
    )
    return qdrant_client


# read a file
def process_text_data(text, title, authors, subject):
    """This function reads the utf8 file, chunks it into paragraphs, and then
        sends the processed text back for being embedded.

    Args:
        text (str): the raw text file that was downloaded
        title (str): Title of the text
        authors (str): Authors of the text
        subject (str): Subject
    Returns:
        documents: List of documents that can be embedded
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    processed_texts = text_splitter.create_documents([text])
    paragraphs = text.split("\n")  # Split on double newline (paragraph separator)
    first_paragraph = paragraphs[0]
    metadata_dict = {
        "text_description": first_paragraph,
        "title": title,
        "authors": authors,
        "subject": subject,
    }
    for text in processed_texts:
        text.metadata = metadata_dict

    return processed_texts


def fetch_book_text(url):
    """
    Download text file and return the response content

    Args:
        url (str) :  download link
    """
    response = requests.get(url)
    if response.status_code == 200:
        content_str = response.content.decode("utf-8")
        return content_str, 1
    else:
        print(
            f"Failed to download file from {url}. Status code: {response.status_code}"
        )
        return 0, 0

def init_collection_qdrant(collection_name:str, client_q: QdrantClient, dimensions = 768):
    """Initialize collection qdrant

    Args:
        collection_name (str): name
        client_q (QdrantClient): client
    """

    client_q.recreate_collection(
        collection_name=f"{collection_name}",
        vectors_config=models.VectorParams(size=dimensions,
                                           distance=models.Distance.COSINE),
    )

def create_and_store_embedding(
    qdrant_client, collection_name: str, docs: list, host=HOST_URL_QDRANT
):
    """Adds new documents to an existing vector store.

    Args:
        qdrant_client : qdrant client instance
        collection_name : collection name
        docs : list of documents
        host: host_url. Defaults to HOST_URL_QDRANT.

    Returns:
        _type_: _description_
    """
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    store = Qdrant(
        client=qdrant_client,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    store.add_texts(texts=[i.page_content for i in docs],metadatas=[i.metadata for i in docs])
    return 1,1


if __name__ == "__main__":
    print("Loading functions ...")
    qdrant_client = init_qdrant_client()
    utf_pattern = r"\.txt\.utf-8$"
    with open("./gutenberg-metadata.json", "r") as json_file:
        metadata_dict = json.load(json_file)
    # try:
    c = 0
    try:
        init_collection_qdrant("gutenberg_v0",qdrant_client)
    except:
        pass

    for key,value in metadata_dict.items():
        c+=1
        start = datetime.now()
        links = value["formaturi"]
        author = value["author"]
        subject = value["subject"]
        title = value["title"]

        links = [l for l in links if re.search(utf_pattern, l)][-1]

        print("\n --- 1. Downloading the required files ...")
        raw_text,flag = fetch_book_text(links)
        if flag == 0:
            raise Exception

        print("\n --- 2. Process Texts")
        processed_texts = process_text_data(raw_text,title=title,authors=author,subject=subject)

        print(" --- 3. Create & Store embeddings ")
        _, _ = create_and_store_embedding(
            qdrant_client, "gutenberg_v0", processed_texts
        )
        end = datetime.now()
        print(f"{end-start} seconds elapsed for 1 file!")
        if c == 5:
            break
    # except Exception as e:
    #     print(f"Error, exception happened {e}")
