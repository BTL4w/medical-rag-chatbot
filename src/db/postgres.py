from __future__ import annotations

from typing import List, Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt

try:  # optional pgvector adapter
    from pgvector.psycopg2 import register_vector
except Exception:  # pragma: no cover
    register_vector = None

from core.embedding import ONNXEmbedding


class PostgresDB:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self.conn.autocommit = True
        if register_vector:
            register_vector(self.conn)
        self.embedding_model = ONNXEmbedding()

    def create_user(self, username: str, password: str) -> int:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING user_id",
                (username, password_hash),
            )
            return cur.fetchone()[0]

    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT user_id, password_hash FROM users WHERE username=%s", (username,))
            row = cur.fetchone()
            if not row:
                return None
            user_id, password_hash = row
            if bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
                return user_id
            return None

    def create_conversation(self, user_id: int, title: str = "New Chat") -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING conversation_id",
                (user_id, title),
            )
            return cur.fetchone()[0]

    def get_user_conversations(self, user_id: int) -> List[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT conversation_id, title, created_at, updated_at FROM conversations WHERE user_id=%s ORDER BY updated_at DESC",
                (user_id,),
            )
            return cur.fetchall()

    def add_message(self, conversation_id: int, role: str, content: str, tokens_used: int = 0):
        embedding = self.embedding_model.embed([content])[0]
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, embedding, tokens_used) VALUES (%s, %s, %s, %s, %s)",
                (conversation_id, role, content, embedding.tolist(), tokens_used),
            )
            cur.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id=%s",
                (conversation_id,),
            )

    def get_conversation_history(self, conversation_id: int, last_n: int = 4) -> List[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id=%s ORDER BY created_at DESC LIMIT %s",
                (conversation_id, last_n * 2),
            )
            rows = cur.fetchall()
        return list(reversed(rows))

    def search_similar_messages(self, query_embedding, user_id: int, top_k: int = 5) -> List[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT m.message_id, m.content, m.role, m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.conversation_id
                WHERE c.user_id = %s
                ORDER BY m.embedding <=> %s
                LIMIT %s
                """,
                (user_id, query_embedding.tolist(), top_k),
            )
            return cur.fetchall()

    def get_full_conversation(self, conversation_id: int) -> List[dict]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id=%s ORDER BY created_at ASC",
                (conversation_id,),
            )
            return cur.fetchall()

    def save_conversation_summary(self, conversation_id: int, summary: str):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET summary=%s, updated_at=CURRENT_TIMESTAMP WHERE conversation_id=%s",
                (summary, conversation_id),
            )
