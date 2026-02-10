# =========================
# Environment & Imports
# =========================
import os
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================
# Load Environment Variables
# =========================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env file")


# =========================
# Load & Process Documents
# =========================
PDF_FILES = [
    "https://www.iima.ac.in/sites/default/files/2023-01/HR%20Policy%20Manual%202023%20%288%29.pdf"
    # "https://www.aeee.in/wp-content/uploads/2020/08/Sample-pdf.pdf"
]

all_docs = []
for pdf in PDF_FILES:
    loader = PyPDFLoader(pdf)
    all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(all_docs)


# =========================
# Embeddings & Vector Store
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vector_store.as_retriever()


# =========================
# LLM (Groq)
# =========================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.7
)


# =========================
# Prompt Templates
# =========================
SYSTEM_PROMPT = """
You are Innovex Genie, an onboarding assistant for new employees at Innovex.

Use ONLY the retrieved context to answer the question.
If the answer is not in the context, say you don’t know.

Be concise, friendly, and helpful.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# =========================
# RAG Chain (Runnable-based)
# =========================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)


# =========================
# FastAPI App
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API Schemas
# =========================
class ChatRequest(BaseModel):
    message: str


# =========================
# In-Memory Chat History
# =========================
chat_histories: Dict[str, List] = {}


# =========================
# Chat Endpoint
# =========================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = "default"

        if session_id not in chat_histories:
            chat_histories[session_id] = []

        chat_history = chat_histories[session_id]

        answer = rag_chain.invoke(request.message)

        chat_history.append(HumanMessage(content=request.message))
        chat_history.append(AIMessage(content=answer))

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
