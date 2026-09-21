from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Cross-border Ecommerce Customer Service Agent",
    version="0.1.0",
    description=(
        "Agentic customer service API with intent routing, RAG, tool calling, "
        "memory, and reflection."
    ),
)

app.include_router(router)

frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")


@app.get("/", include_in_schema=False)
def frontend_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")
