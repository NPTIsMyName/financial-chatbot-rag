# build_data.py
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
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

        # Lọc metadata để tránh NoneType
        meta = {
            "title": str(item.get("title") or ""),
            "date": str(item.get("date") or ""),
            "url": str(item.get("url") or ""),
            "author": str(item.get("author") or ""),
        }

        title = meta["title"]
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
    
    #Use HF API embeddings
    # embeddings = HuggingFaceEndpointEmbeddings(model='BAAI/bge-m3', 
    #     task="feature-extraction", 
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    
    #Use local HF embeddings
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', 
        model_kwargs = {"device" : "cpu"},
        encode_kwargs= {"normalize_embeddings": False}
        )

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
