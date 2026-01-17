from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "medical_rag"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Vector Store
    VECTOR_STORE_TYPE: str = "pinecone"
    QDRANT_URL: str = "http://localhost:6333"
    PINECONE_API_KEY: str = ""
    PINECONE_ENV: str = ""
    PINECONE_INDEX_NAME: str = "youmed-articles"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # Models
    EMBEDDING_MODEL_PATH: str = "models/bge-m3-onnx"
    RERANKER_MODEL_PATH: str = "models/bge-reranker-v2-m3-onnx"
    BM25_INDEX_PATH: str = "models/bm25_index.pkl"
    EMBEDDING_DIM: int = 1024

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATES: int = 20
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # Session
    SESSION_TTL: int = 3600
    MAX_CONVERSATION_TURNS: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
