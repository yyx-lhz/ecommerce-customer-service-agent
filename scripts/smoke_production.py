"""Real models + real services + FastAPI; fails if production is unavailable."""

import json
import os

from fastapi.testclient import TestClient

from app.core.config import get_settings


def main():
    os.environ["RAG_BACKEND"] = "production"
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/chat", json={"message": "Is AX900 water damage covered?"})
        response.raise_for_status()
        data = response.json()
        assert any(
            c["source"] == "sample_policies.pdf" and c["page"] == 1 for c in data["citations"]
        ), data
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
