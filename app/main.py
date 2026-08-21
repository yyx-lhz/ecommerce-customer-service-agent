from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Cross-border Ecommerce Customer Service Agent",
    version="0.1.0",
    description="Agentic customer service API with intent routing, RAG, tool calling, memory, and reflection.",
)

app.include_router(router)
