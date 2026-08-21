from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import redis

from app.core.config import Settings


@dataclass
class Message:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._local: dict[str, list[Message]] = {}
        self._redis = None
        if settings.use_redis:
            self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    def load(self, session_id: str, limit: int = 8) -> list[Message]:
        if self._redis:
            raw = self._redis.lrange(self._key(session_id), -limit, -1)
            return [Message(**json.loads(item)) for item in raw]
        return self._local.get(session_id, [])[-limit:]

    def append(self, session_id: str, role: str, content: str) -> None:
        message = Message(role=role, content=content)
        if self._redis:
            key = self._key(session_id)
            self._redis.rpush(key, json.dumps(asdict(message), ensure_ascii=False))
            self._redis.ltrim(key, -20, -1)
            self._redis.expire(key, 60 * 60 * 24)
            return
        self._local.setdefault(session_id, []).append(message)
        self._local[session_id] = self._local[session_id][-20:]

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:session:{session_id}"
