"""
Chroma-based RAG knowledge base for ecommerce products.
Builds embeddings from product data and supports bilingual (EN/ZH) retrieval.
"""

import json
import os
from pathlib import Path

from chromadb import Client, Settings
from chromadb.utils import embedding_functions

DATA_PATH = Path(__file__).parent / "data" / "products.json"


class ProductKnowledgeBase:
    """Vector-store-backed knowledge base for ecommerce product information."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or str(Path(__file__).parent / "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = Client(Settings(
            is_persistent=True,
            persist_directory=self.persist_dir,
            anonymized_telemetry=False,
        ))

        # Use OpenAI embedding function
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

        self.collection_name = "ecommerce_products"
        self._load_or_build()

    def _load_or_build(self):
        """Load existing collection or build from scratch."""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.ef,
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.ef,
            )
            self._build_index()

    def _build_index(self):
        """Index all product data into the vector store."""
        with open(DATA_PATH) as f:
            data = json.load(f)

        products = data["products"]
        return_policy = data["return_policy"]
        shipping_info = data["shipping_info"]

        docs = []
        metadatas = []
        ids = []

        for i, p in enumerate(products):
            # Pre-compute shipping strings to avoid nested f-string issues
            ship = p["shipping"]
            if ship["free_shipping"]:
                ship_en = "Free"
                ship_cn = "包邮"
            else:
                cost = ship.get("shipping_cost", 0)
                ship_en = f"${cost}"
                ship_cn = f"运费${cost}"

            doc_en = (
                f"Product: {p['name']}\n"
                f"Category: {p['category']}\n"
                f"Price: ${p['price']} {p['currency']}\n"
                f"Stock: {p['stock']} units\n"
                f"Shipping: {ship_en}, estimated {ship['estimated_days']} days\n"
                f"Description: {p['description']}\n"
            )
            doc_cn = (
                f"产品: {p['name_cn']} ({p['name']})\n"
                f"类别: {p['category']}\n"
                f"价格: ${p['price']} {p['currency']}\n"
                f"库存: {p['stock']} 件\n"
                f"配送: {ship_cn}, 预计{ship['estimated_days']}天\n"
                f"描述: {p['description_cn']}\n"
            )
            combined = doc_en + "\n---\n" + doc_cn

            docs.append(combined)
            metadatas.append({
                "product_id": p["id"],
                "name": p["name"],
                "name_cn": p["name_cn"],
                "category": p["category"],
                "price": p["price"],
                "stock": p["stock"],
            })
            ids.append(f"product_{p['id']}")

            # Index FAQ entries individually
            for j, faq in enumerate(p.get("faq", [])):
                docs.append(f"[FAQ for {p['name']}] Q: {faq['q']}\nA: {faq['a']}")
                metadatas.append({
                    "product_id": p["id"],
                    "name": p["name"],
                    "type": "faq",
                })
                ids.append(f"faq_{p['id']}_{j}")

        # Index return policy
        docs.append(
            f"Return Policy: {return_policy['period_days']}-day return window. "
            f"Conditions: {return_policy['conditions']} "
            f"Refund timeline: {return_policy['refund_timeline']}"
        )
        metadatas.append({"type": "policy", "policy_type": "return"})
        ids.append("policy_return")

        # Index shipping info
        carriers = ", ".join(shipping_info["carriers"])
        docs.append(
            f"Shipping Info: Free shipping on orders over ${shipping_info['free_shipping_threshold']}. "
            f"Carriers: {carriers}. Processing time: {shipping_info['processing_time_days']} business days."
        )
        metadatas.append({"type": "policy", "policy_type": "shipping"})
        ids.append("policy_shipping")

        self.collection.add(documents=docs, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search the knowledge base.
        Returns a list of documents with metadata and relevance scores.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return output

    def get_product_faqs(self, product_id: str) -> list[dict]:
        """Get all FAQ entries for a specific product."""
        results = self.collection.get(
            where={"$and": [
                {"product_id": product_id},
                {"type": "faq"},
            ]}
        )
        return [
            {"id": rid, "content": doc, "metadata": meta}
            for rid, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def rebuild(self):
        """Delete and rebuild the entire index."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
        )
        self._build_index()
