from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import redis

from app.core.config import Settings


@dataclass
class Message:
    role: str
    content: str


@dataclass
class StructuredCustomerMemory:
    customer_profile: dict[str, Any]
    order_context: dict[str, Any]
    current_case: dict[str, Any]
    previous_actions: list[dict[str, Any]]
    important_business_facts: list[str]

    @classmethod
    def empty(cls) -> StructuredCustomerMemory:
        return cls(
            customer_profile={},
            order_context={},
            current_case={},
            previous_actions=[],
            important_business_facts=[],
        )


class ConversationMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._local: dict[str, list[Message]] = {}
        self._structured_local: dict[str, StructuredCustomerMemory] = {}
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

    def load_structured(self, session_id: str) -> StructuredCustomerMemory:
        if self._redis:
            raw = self._redis.get(self._structured_key(session_id))
            if raw:
                return StructuredCustomerMemory(**json.loads(raw))
            return StructuredCustomerMemory.empty()
        return self._structured_local.get(session_id, StructuredCustomerMemory.empty())

    def save_structured(self, session_id: str, memory: StructuredCustomerMemory) -> None:
        if self._redis:
            key = self._structured_key(session_id)
            self._redis.set(key, json.dumps(asdict(memory), ensure_ascii=False), ex=60 * 60 * 24 * 7)
            return
        self._structured_local[session_id] = memory

    def update_structured(
        self,
        session_id: str,
        *,
        customer_profile: dict[str, Any] | None = None,
        order_context: dict[str, Any] | None = None,
        current_case: dict[str, Any] | None = None,
        previous_actions: list[dict[str, Any]] | None = None,
        important_business_facts: list[str] | None = None,
    ) -> StructuredCustomerMemory:
        memory = self.load_structured(session_id)
        if customer_profile:
            memory.customer_profile.update(customer_profile)
        if order_context:
            memory.order_context.update(order_context)
        if current_case:
            memory.current_case = current_case
        if previous_actions:
            memory.previous_actions.extend(previous_actions)
            memory.previous_actions = memory.previous_actions[-30:]
        if important_business_facts:
            known = set(memory.important_business_facts)
            for fact in important_business_facts:
                if fact not in known:
                    memory.important_business_facts.append(fact)
            memory.important_business_facts = memory.important_business_facts[-30:]
        self.save_structured(session_id, memory)
        return memory

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:session:{session_id}"

    @staticmethod
    def _structured_key(session_id: str) -> str:
        return f"agent:structured:{session_id}"
