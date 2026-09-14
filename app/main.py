from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.agent.workflow import CustomerServiceAgent
from app.api.routes import router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app):
    app.state.agent = CustomerServiceAgent(get_settings(), Path("data/knowledge"))
    try:
        yield
    finally:
        close = getattr(app.state.agent.retriever, "close", None)
        if close:
            close()


app = FastAPI(title="Ecommerce Customer Service Agent", version="0.2.0", lifespan=lifespan)
app.include_router(router)
