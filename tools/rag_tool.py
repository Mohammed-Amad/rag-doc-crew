"""
custom:rag_search
------------------
A CrewAI custom tool implementing the RAG pipeline ourselves (chunk -> embed
-> store -> retrieve) instead of using CrewAI's built-in PDFSearchTool/RagTool.
This makes the retrieval step fully inspectable.

Retrieval happens ENTIRELY inside this tool's _run() method. Agents never see
the raw PDFs — they only see whatever chunks this tool decides to return.
"""

import os
import glob
import hashlib
import time
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".chroma_store")

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def extract_text_by_file(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


import re


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback: blind character sliding window, used only for pieces that
    are still too long after splitting on section boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


SECTION_HEADER_RE = re.compile(r'(?=\b\d{1,2}\.\s+[A-Z])')


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Section-aware chunker: splits on the document's own numbered section
    headers first, so each chunk stays topically pure (e.g. "gross margin"
    doesn't get diluted by being crammed together with the neighboring
    "revenue by region" and "operating expenses" sections). Only falls back
    to a blind sliding window for any individual section that's still too
    long on its own.

    This matters for retrieval precision: a narrow query like "gross margin"
    ranks much better against a small chunk that's ONLY about gross margin
    than against a chunk where that's one of three unrelated topics mixed
    together.
    """
    text = " ".join(text.split())

    split_points = [m.start() for m in SECTION_HEADER_RE.finditer(text)]
    if not split_points:
        return [c for c in _sliding_window(text, chunk_size, overlap) if c.strip()]

    boundaries = [0] + split_points + [len(text)]
    sections = [text[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]

    chunks = []
    for section in sections:
        if len(section) <= chunk_size * 1.4:
            chunks.append(section)
        else:
            chunks.extend(_sliding_window(section, chunk_size, overlap))

    return [c.strip() for c in chunks if c.strip()]


def _build_embedding_function():
    if EMBEDDING_PROVIDER == "local":
        return embedding_functions.DefaultEmbeddingFunction()
    elif EMBEDDING_PROVIDER == "gemini":
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="models/text-embedding-004",
        )
    elif EMBEDDING_PROVIDER == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def _build_or_load_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name="northwind_knowledge",
        embedding_function=_build_embedding_function(),
    )

    if collection.count() == 0:
        ids, docs, metas = [], [], []
        pdf_paths = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf")))
        for pdf_path in pdf_paths:
            source_name = os.path.basename(pdf_path)
            full_text = extract_text_by_file(pdf_path)
            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{source_name}::chunk_{i}"
                uid = hashlib.md5(chunk_id.encode()).hexdigest()
                ids.append(uid)
                docs.append(chunk)
                metas.append({"source": source_name, "chunk_index": i})
        if docs:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            print(f"[rag_tool] Ingested {len(docs)} chunks from {len(pdf_paths)} PDFs.")
    return collection


class RagSearchInput(BaseModel):
    query: str = Field(..., description="The question or topic to search for in the local knowledge base.")


class DocumentRAGTool(BaseTool):
    name: str = "custom:rag_search"
    description: str = (
        "Searches the company's internal knowledge base (employee handbook, "
        "product spec, Q3 financial summary) and returns the most relevant "
        "text chunks, each tagged with its source document and chunk index. "
        "Use this instead of guessing or using outside knowledge — it is the "
        "ONLY source of truth for this task. If nothing relevant is returned, "
        "say so explicitly rather than making something up."
    )
    args_schema: type[BaseModel] = RagSearchInput

    def _run(self, query: str) -> str:
        last_error = None
        for attempt in range(3):
            try:
                collection = _build_or_load_collection()
                results = collection.query(query_texts=[query], n_results=TOP_K)

                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]

                if not docs:
                    return "NO_RELEVANT_CHUNKS_FOUND"

                formatted = []
                for doc, meta, dist in zip(docs, metas, dists):
                    formatted.append(
                        f"[SOURCE: {meta['source']} | chunk {meta['chunk_index']} | "
                        f"similarity_distance={dist:.3f}]\n{doc}"
                    )
                return "\n\n---\n\n".join(formatted)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue

        return (
            f"RAG_SEARCH_TEMPORARILY_FAILED: could not query the knowledge base "
            f"after 3 attempts ({last_error}). This is an infrastructure error, "
            f"not evidence that the information doesn't exist -- retry this "
            f"query once more before concluding it isn't in the documents."
        )


try:
    _build_or_load_collection()
except Exception as _warmup_error:
    print(f"[rag_tool] Warm-up query failed (will retry on first real call): {_warmup_error}")
