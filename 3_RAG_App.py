# pip install -U langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv langsmith huggingface-hub sentence-transformers

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_PROJECT"] = "RAG Application"
load_dotenv()

PDF_PATH = "islr.pdf"  # keep in same folder
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- helpers -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    # Free + strong embedding
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_documents(splits, emb)

# ----------------- cache / fingerprint -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "format": "v2",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- index builder -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path):
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap)
    index_dir = INDEX_ROOT / key
    faiss_file = index_dir / "index.faiss"

    # 🔹 Check if FAISS file actually exists
    if not force_rebuild and faiss_file.exists():
        return load_index_run(index_dir)
    else:
        print("⚡ Building FAISS index (first time or forced)...")
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap)


# ----------------- pipeline -----------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step 1: retrieval prompt
combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer the question. If not found, say 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str, chunk_size: int = 1000, chunk_overlap: int = 150, force_rebuild: bool = False):
    vectorstore = load_or_build_index(pdf_path, chunk_size, chunk_overlap, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # ✅ more chunks

    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = combine_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("📖 PDF RAG ready (multi-chunk summary mode). Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
