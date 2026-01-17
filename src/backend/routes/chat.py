import time
from fastapi import APIRouter, HTTPException, Depends

from backend.dependencies import get_retriever, get_generator, get_db, get_redis
from backend.models.chat import ChatRequest, ChatResponse
from core.retriever import HybridRetriever
from core.generator import MedicalRAGGenerator
from db.postgres import PostgresDB
from db.redis_cache import RedisCache
from utils.config import settings
from utils.logger import setup_logger

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = setup_logger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    generator: MedicalRAGGenerator = Depends(get_generator),
    db: PostgresDB = Depends(get_db),
    cache: RedisCache = Depends(get_redis),
):
    session = cache.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    conversation_id = request.conversation_id or db.create_conversation(session["user_id"])

    context = cache.get_conversation_context(conversation_id)
    if not context:
        context = db.get_conversation_history(conversation_id, last_n=settings.MAX_CONVERSATION_TURNS)

    t0 = time.time()
    retrieved_chunks = retriever.retrieve(
        request.query,
        top_k=settings.RETRIEVAL_TOP_K,
        vector_weight=settings.VECTOR_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
    )
    retrieval_latency = time.time() - t0

    t0 = time.time()
    if len(context) > settings.MAX_CONVERSATION_TURNS * 2:
        summary = generator.summarize_conversation(context)
        db.save_conversation_summary(conversation_id, summary)
        context = [{"role": "system", "content": f"Summary: {summary}"}] + context[-4:]

    result = generator.generate(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        conversation_history=context,
    )
    llm_latency = time.time() - t0

    db.add_message(conversation_id, "user", request.query)
    db.add_message(conversation_id, "assistant", result["answer"], result["tokens_used"])

    cache.append_to_context(conversation_id, "user", request.query, max_turns=settings.MAX_CONVERSATION_TURNS)
    cache.append_to_context(
        conversation_id,
        "assistant",
        result["answer"],
        max_turns=settings.MAX_CONVERSATION_TURNS,
    )

    logger.info(
        "Retrieval: %.2fs, LLM: %.2fs, Tokens: %s, Cost: $%.4f",
        retrieval_latency,
        llm_latency,
        result["tokens_used"],
        result["cost"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conversation_id,
        tokens_used=result["tokens_used"],
        cost=result["cost"],
    )


@router.get("/history/{conversation_id}")
async def get_history(
    conversation_id: int,
    session_id: str,
    db: PostgresDB = Depends(get_db),
    cache: RedisCache = Depends(get_redis),
):
    session = cache.get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    messages = db.get_full_conversation(conversation_id)
    return {"messages": messages}
