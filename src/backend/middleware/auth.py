from fastapi import HTTPException

from db.redis_cache import RedisCache


def require_session(session_id: str, cache: RedisCache):
    session = cache.get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    return session
