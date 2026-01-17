from __future__ import annotations

import json
import uuid
from typing import List, Optional

import redis


class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379, default_ttl: int = 3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.default_ttl = default_ttl

    def create_session(self, user_id: int, conversation_id: int) -> str:
        session_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        self.client.setex(f"session:{session_id}", self.default_ttl, json.dumps(data))
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        data = self.client.get(f"session:{session_id}")
        if not data:
            return None
        self.client.expire(f"session:{session_id}", self.default_ttl)
        return json.loads(data)

    def delete_session(self, session_id: str):
        self.client.delete(f"session:{session_id}")

    def cache_conversation_context(self, conversation_id: int, messages: List[dict], max_turns: int = 4):
        payload = json.dumps(messages[-max_turns * 2 :], ensure_ascii=False)
        self.client.setex(f"context:{conversation_id}", self.default_ttl, payload)

    def get_conversation_context(self, conversation_id: int) -> Optional[List[dict]]:
        cached = self.client.get(f"context:{conversation_id}")
        if cached:
            return json.loads(cached)
        return None

    def append_to_context(self, conversation_id: int, role: str, content: str, max_turns: int = 4):
        existing = self.get_conversation_context(conversation_id) or []
        existing.append({"role": role, "content": content})
        payload = json.dumps(existing[-max_turns * 2 :], ensure_ascii=False)
        self.client.setex(f"context:{conversation_id}", self.default_ttl, payload)

    def check_rate_limit(self, user_id: int, max_requests: int = 10, window: int = 60) -> bool:
        key = f"rate:{user_id}"
        count = self.client.incr(key)
        if count == 1:
            self.client.expire(key, window)
        return count <= max_requests
