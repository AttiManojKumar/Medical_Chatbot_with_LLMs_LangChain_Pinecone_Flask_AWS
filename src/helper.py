import os
import time
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from google import genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.prompt import build_gemini_prompt


INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_DIMENSION = 384
GEMINI_MODEL = "gemini-2.5-flash-lite"


def get_project_root() -> Path:
    """Return the main project folder, even when code runs from research/."""
    current_dir = Path.cwd()
    if current_dir.name == "research":
        return current_dir.parent
    return current_dir


def get_data_path(folder_name: str = "data") -> Path:
    """Return the absolute path to the folder that contains the PDF files."""
    return get_project_root() / folder_name


def load_pdf_file(data_path: str | Path) -> List[Document]:
    """Load every PDF file from the given folder."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def filter_to_minimal_docs(docs: Iterable[Document]) -> List[Document]:
    """Keep only useful text and the source file path from each document."""
    minimal_docs = []

    for doc in docs:
        if not doc.page_content or not doc.page_content.strip():
            continue

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source")},
            )
        )

    return minimal_docs


def text_split(docs: List[Document]) -> List[Document]:
    """Split large PDF pages into smaller chunks for vector search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    return text_splitter.split_documents(docs)


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used for both PDF chunks and questions."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_api_keys() -> dict:
    """Load API keys from .env and return them in one place."""
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("Gemini_API_Key")
        or os.getenv("Gemini API Key")
    )

    if pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

    return {
        "pinecone": pinecone_api_key,
        "openai": openai_api_key,
        "gemini": gemini_api_key,
    }


def require_key(keys: dict, key_name: str) -> str:
    """Read one key from the loaded key dictionary or raise a clear error."""
    value = keys.get(key_name)
    if not value:
        raise ValueError(f"{key_name} API key is missing. Check your .env file.")
    return value


def get_pinecone_client(api_key: str) -> Pinecone:
    """Create a Pinecone client."""
    return Pinecone(api_key=api_key)


def create_or_get_pinecone_index(
    pc: Pinecone,
    index_name: str = INDEX_NAME,
):
    """Create the Pinecone index if needed, then return it."""
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

    wait_for_index_ready(pc, index_name)
    return pc.Index(index_name)


def wait_for_index_ready(pc: Pinecone, index_name: str) -> None:
    """Wait until a newly created Pinecone index is ready to use."""
    while True:
        description = pc.describe_index(index_name)
        status = getattr(description, "status", {})

        if isinstance(status, dict) and status.get("ready"):
            return
        if getattr(status, "ready", False):
            return

        time.sleep(1)


def get_vector_count(index) -> int:
    """Return how many vectors are already stored in the Pinecone index."""
    stats = index.describe_index_stats()

    if hasattr(stats, "total_vector_count"):
        return int(stats.total_vector_count)

    return int(stats.get("total_vector_count", 0))


def build_vector_store(index_name: str, embeddings) -> PineconeVectorStore:
    """Connect LangChain to an existing Pinecone index."""
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )


def upload_chunks_to_pinecone(
    text_chunks: List[Document],
    embeddings,
    index_name: str = INDEX_NAME,
) -> PineconeVectorStore:
    """Upload document chunks and their embeddings to Pinecone."""
    return PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )


def search_pinecone(index, embeddings, question: str, top_k: int = 3):
    """Search Pinecone directly with a user question and return matches."""
    query_vector = embeddings.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return results.get("matches", [])


def get_relevant_docs(docsearch: PineconeVectorStore, question: str, top_k: int = 3):
    """Use LangChain retriever to fetch relevant chunks from Pinecone."""
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever.invoke(question)


def get_gemini_client(api_key: str):
    """Create a Gemini client using the Google GenAI SDK."""
    return genai.Client(api_key=api_key)


def answer_with_gemini(
    client,
    question: str,
    docs: List[Document],
    model: str | None = None,
) -> str:
    """Ask Gemini to write a final answer using retrieved Pinecone chunks."""
    selected_model = model or os.getenv("GEMINI_MODEL", GEMINI_MODEL)
    prompt = build_gemini_prompt(question, docs)

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
    )
    return response.text
