import sys
from pathlib import Path
import logging
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.routes import auth, chat, health
from utils.config import settings
from core.embedding import ONNXEmbedding
from core.reranker import ONNXReranker
from core.retriever import HybridRetriever
from core.generator import MedicalRAGGenerator
from db.vector_store import VectorStoreFactory
from db.postgres import PostgresDB
from db.redis_cache import RedisCache

app = FastAPI(
    title="Medical RAG Chatbot API",
    version="1.0.0",
    description="AI-powered medical chatbot using RAG",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    logger.info("Loading models...")
    app.state.embedding_model = ONNXEmbedding(settings.EMBEDDING_MODEL_PATH)
    app.state.reranker = ONNXReranker(settings.RERANKER_MODEL_PATH)

    from core.retriever import BM25Retriever

    app.state.bm25_retriever = BM25Retriever.load(settings.BM25_INDEX_PATH)

    vector_kwargs = {}
    if settings.VECTOR_STORE_TYPE == "qdrant":
        vector_kwargs["url"] = settings.QDRANT_URL
    elif settings.VECTOR_STORE_TYPE == "pinecone":
        vector_kwargs["api_key"] = settings.PINECONE_API_KEY
        vector_kwargs["index_name"] = settings.PINECONE_INDEX_NAME
        vector_kwargs["environment"] = settings.PINECONE_ENV or None

    app.state.vector_store = VectorStoreFactory.create(
        settings.VECTOR_STORE_TYPE,
        **vector_kwargs,
    )
    if settings.VECTOR_STORE_TYPE == "pinecone":
        app.state.vector_store.create_collection(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIM,
        )

    app.state.hybrid_retriever = HybridRetriever(
        vector_store=app.state.vector_store,
        bm25_retriever=app.state.bm25_retriever,
        embedding_model=app.state.embedding_model,
        reranker=app.state.reranker,
        candidates=settings.RETRIEVAL_CANDIDATES,
    )

    app.state.generator = MedicalRAGGenerator(
        openai_api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
    )

    app.state.db = PostgresDB(
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    app.state.cache = RedisCache(host=settings.REDIS_HOST, port=settings.REDIS_PORT, default_ttl=settings.SESSION_TTL)

    logger.info("Models loaded successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")


app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(health.router)
