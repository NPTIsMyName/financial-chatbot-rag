# build_data.py
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import hashlib
import os

def load_json_news(json_path="vnexpress_kinhdoanh.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_documents(items):
    docs = []
    for item in items:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        title = item.get("title", "")
        meta = {
            "title": title,
            "date": item.get("date", ""),
            "url": item.get("url", ""),
            "author": item.get("author", ""),
        }
        # Prepend title to content to weight it in retrieval
        page_content = f"{title}\n\n{content}".strip() if title else content
        docs.append(Document(page_content=page_content, metadata=meta))
    return docs

def text_2_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=900,
        chunk_overlap=180,
        length_function=len,
    )
    return splitter.split_documents(docs)

def doc_ids_for(doc: Document, idx: int) -> str:
    base = (doc.metadata.get("url") or doc.metadata.get("title") or str(idx)) + f"#{idx}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def build_chroma():
    load_dotenv()
    print("Loading data...")
    items = load_json_news("vnexpress_kinhdoanh.json")
    docs = make_documents(items)
    chunks = text_2_chunks(docs)

    print(f"Chunks: {len(chunks)}")
    print("Generating embeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(model='BAAI/bge-m3', 
        task="feature-extraction", 
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    print("Saving to local ChromaDB...")
    collection_name = "vnexpress_kinhdoanh"
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="chroma_store",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Create stable IDs to avoid duplicates on rebuild
    ids = [doc_ids_for(chunks[i], i) for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)
    vectorstore.persist()
    print("Done! Data saved to 'chroma_store/'")

if __name__ == "__main__":
    build_chroma()
