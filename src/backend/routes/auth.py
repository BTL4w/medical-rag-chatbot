from fastapi import APIRouter, HTTPException, Depends

from backend.dependencies import get_db, get_redis
from backend.models.user import UserCreate, UserLogin
from db.postgres import PostgresDB
from db.redis_cache import RedisCache

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register")
async def register(user: UserCreate, db: PostgresDB = Depends(get_db)):
    try:
        user_id = db.create_user(user.username, user.password)
        return {"message": "User created successfully", "user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/login")
async def login(
    user: UserLogin,
    db: PostgresDB = Depends(get_db),
    cache: RedisCache = Depends(get_redis),
):
    user_id = db.authenticate_user(user.username, user.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    conversation_id = db.create_conversation(user_id)
    session_id = cache.create_session(user_id, conversation_id)

    return {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "message": "Login successful",
    }


@router.post("/logout")
async def logout(session_id: str, cache: RedisCache = Depends(get_redis)):
    cache.delete_session(session_id)
    return {"message": "Logout successful"}


@router.get("/conversations")
async def get_conversations(
    session_id: str,
    db: PostgresDB = Depends(get_db),
    cache: RedisCache = Depends(get_redis),
):
    session = cache.get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    conversations = db.get_user_conversations(session["user_id"])
    return conversations
