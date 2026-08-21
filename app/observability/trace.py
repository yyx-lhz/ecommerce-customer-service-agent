from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class TraceEvent:
    stage: str
    payload: dict[str, Any]


@dataclass
class AgentTrace:
    trace_id: str = field(default_factory=lambda: f"TR-{uuid4().hex[:12].upper()}")
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, stage: str, **payload: Any) -> None:
        self.events.append(TraceEvent(stage=stage, payload=payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": [
                {
                    "stage": event.stage,
                    "payload": event.payload,
                }
                for event in self.events
            ],
        }
