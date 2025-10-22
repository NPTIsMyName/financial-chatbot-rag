import os
import re
from typing import Dict

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_groq import ChatGroq

def _require_groq_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    return api_key




def create_llm():
    api_key = _require_groq_key()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        api_key=api_key,
    )
# def _require_google_key() -> str:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise RuntimeError("⚠️ Missing GOOGLE_API_KEY in environment.")
#     return api_key


# def create_llm():
#     _require_google_key()
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.1,
#         max_output_tokens=1024,
#     )

def create_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    llm = create_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # Chat-style prompt compatible with ChatHuggingFace FinSight
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing and summarizing financial news, market reports, and economic trends.
            Your goals:
            1. Summarize key information from the provided context clearly and accurately.
            2. Analyze financial or market trends (e.g., stocks, currencies, companies, macroeconomy) based on available data.
            3. If possible, provide short-term **predictions or insights** (e.g., likely to rise/fall/stay stable) — but make it clear that this is an **estimated analysis**, not investment advice.
            4. If the question asks for forecasts, respond cautiously, using data-driven reasoning.
            5. If no relevant information is found in the provided context, respond exactly with:"Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."
            6. Always respond in **natural, fluent Vietnamese**, using a **professional, objective, and concise** tone.
            7. The answer must be **short, clear, and information-rich** — concise but still complete.
            """
            ),
        HumanMessagePromptTemplate.from_template(
            """Question: {question}
            Context: {context}
            Please provide:
            - A concise and accurate summary.
            - A brief analysis or prediction if supported by the data.
            - The answer must be written entirely in Vietnamese."""
            ),
        ])


    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )


def load_vectorstore():
    """Load the existing persisted Chroma vectorstore built by build_data.py.
    Note: We still instantiate the same embedding function for query-time embeddings.
    """
    # Use local embeddings (same as build_data.py)
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU requirements
        encode_kwargs={'normalize_embeddings': True}
    )
    # embeddings = HuggingFaceEndpointEmbeddings(
    #     model='Qwen/Qwen3-Embedding-0.6B', 
    #     task="feature-extraction", 
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # )
    return Chroma(
        collection_name="vnexpress_kinhdoanh",
        persist_directory="chroma_store",
        embedding_function=embeddings,
    )
    

def format_answer(text: str) -> str:
    """Format the answer text by normalizing bullets, removing markdown, and adding line breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace markdown asterisk bullets at line starts with dashes
    text = re.sub(r"(^|\n)\s*\*\s+", r"\1- ", text)
    # Remove bold markers **...**
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # Insert newline after sentence-ending punctuation followed by space
    text = re.sub(r"([.!?])\s+", r"\1\n", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class RAGHandler:
    """Handles RAG operations with session management."""
    
    def __init__(self):
        self.vectorstore = None
        self.sessions: Dict[str, ConversationalRetrievalChain] = {}
        self.llm = None
    
    def initialize(self):
        """Initialize the vectorstore."""
        self.vectorstore = load_vectorstore()
    
    def get_llm(self):
        """Get or create the LLM instance."""
        if self.llm is None:
            self.llm = create_llm()
        return self.llm
    
    def get_chain(self, session_id: str):
        """Get or create a conversation chain for the session."""
        if session_id not in self.sessions:
            if self.vectorstore is None:
                raise RuntimeError("Vectorstore not initialized. Call initialize() first.")
            self.sessions[session_id] = create_chain(self.vectorstore)
        return self.sessions[session_id]
    
    def process_rag_query(self, session_id: str, message: str) -> str:
        chain = self.get_chain(session_id)
        result = chain.invoke({"question": message})
        raw_answer = result.get("answer") or result.get("result") or ""
        return format_answer(raw_answer)
