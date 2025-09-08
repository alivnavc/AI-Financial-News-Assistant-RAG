import os
import json
import asyncio
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models as qmodels
from openai import OpenAI
import logging
import time
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_REST_PORT = int(os.getenv("QDRANT_REST_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "news_articles")
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_JSON = os.getenv("INPUT_JSON")

# Pipeline settings
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "64"))
EMBED_DIM_OVERRIDE = os.getenv("EMBED_DIM", None)

LOG_FILE = os.getenv("LOG_FILE", "ingestion.log")


# ------------------------------
# Logging setup
# ------------------------------
class JsonFormatter(logging.Formatter):
    """Custom logger that outputs JSON for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["error"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            for k, v in getattr(record, "extra", {}).items():
                base[k] = v
        return json.dumps(base, ensure_ascii=False)


logger = logging.getLogger("ingest")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(JsonFormatter())

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ------------------------------
# Data Models
# ------------------------------
@dataclass
class Article:
    """Represents a news article with basic metadata and text."""
    title: str
    link: str
    ticker: str
    full_text: str

    @classmethod
    def from_raw(cls, raw: Dict[str, Any], ticker: str) -> Optional["Article"]:
        """
        Convert raw JSON object into an Article object.
        Returns None if required fields are missing.
        """
        try:
            title = raw.get("title", "")
            link = raw.get("link", "")
            full_text = raw.get("full_text", raw.get("full_text ", ""))
            if not title and not full_text:
                return None
            return cls(title=title, link=link, ticker=ticker, full_text=full_text)
        except Exception:
            return None

    def embedding_text(self) -> str:
        """Combine title, ticker, and content for embedding."""
        return f"Title: {self.title}\nTicker: {self.ticker}\nContent: {self.full_text}"


# ------------------------------
# OpenAI Embeddings
# ------------------------------
class OpenAIEmbedder:
    """Wrapper for OpenAI embeddings API."""
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def embed(self, text: str, *, dimensions: Optional[int] = None) -> List[float]:
        """
        Generate embedding for a text string asynchronously.
        Uses threads to avoid blocking asyncio loop.
        """
        def _sync_call():
            kwargs = {"model": self.model, "input": text}
            if dimensions:
                kwargs["dimensions"] = int(dimensions)
            resp = self.client.embeddings.create(**kwargs)
            return resp.data[0].embedding

        return await asyncio.to_thread(_sync_call)


# ------------------------------
# Qdrant Store
# ------------------------------
class QdrantStore:
    """Handles Qdrant collection creation and upserts."""
    def __init__(self, host: str, rest_port: int, grpc_port: int, prefer_grpc: bool = True):
        self.client = AsyncQdrantClient(
            host=host,
            port=rest_port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )

    async def ensure_collection(self, name: str, vector_size: int,
                                distance: qmodels.Distance = qmodels.Distance.COSINE):
        """
        Create collection if it does not exist.
        Handles HNSW index parameters for fast similarity search.
        """
        try:
            exists = await self.client.collection_exists(name)
            if not exists:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=distance,
                        hnsw_config=qmodels.HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                        ),
                    ),
                )
                logger.info(f"Created collection {name}", extra={"extra": {"collection": name, "size": vector_size}})
        except Exception:
            logger.error("ensure_collection_failed", extra={"extra": {"collection": name}})
            logger.error(traceback.format_exc())
            raise

    async def upsert_points(self, collection: str, points: List[qmodels.PointStruct]):
        """Insert or update points in a Qdrant collection."""
        try:
            await self.client.upsert(collection_name=collection, points=points)
        except Exception:
            logger.error("upsert_failed", extra={"extra": {"collection": collection, "num_points": len(points)}})
            logger.error(traceback.format_exc())
            raise


# ------------------------------
# Helpers
# ------------------------------
def load_articles(path: str) -> List[Article]:
    """
    Load articles from a JSON file.
    Supports both dict-of-lists (ticker -> articles) or plain list.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: List[Article] = []
    if isinstance(raw, dict):
        for ticker, items in raw.items():
            if not isinstance(items, list):
                continue
            for obj in items:
                a = Article.from_raw(obj, ticker=ticker)
                if a:
                    out.append(a)
    elif isinstance(raw, list):
        for obj in raw:
            ticker = (obj.get("ticker") or "").strip()
            a = Article.from_raw(obj, ticker=ticker)
            if a:
                out.append(a)
    return out


async def embed_articles(
    embedder: OpenAIEmbedder,
    articles: List[Article],
    *,
    concurrency: int,
    dim: Optional[int],
) -> List[Tuple[Article, List[float]]]:
    """
    Embed multiple articles asynchronously.
    Writes failed embeddings to a JSONL file for review.
    """
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[Article, List[float]]] = []
    failures: List[Dict[str, Any]] = []

    async def worker(a: Article):
        async with sem:
            try:
                vec = await embedder.embed(a.embedding_text(), dimensions=dim)
                results.append((a, vec))
            except Exception as e:
                failures.append({
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "article": a.__dict__,
                })

    await asyncio.gather(*(worker(a) for a in articles))

    if failures:
        with open("failures.jsonl", "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.warning("embedding_failures_written", extra={"extra": {"count": len(failures), "file": "failures.jsonl"}})

    return results


def batch(iterable, n):
    """Simple generator to yield batches of size n."""
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ------------------------------
# Main Ingestion Pipeline
# ------------------------------
async def main():
    """Main pipeline: load JSON, embed articles, upsert to Qdrant."""
    logger.info("start_ingestion", extra={"extra": {"file": INPUT_JSON, "collection": QDRANT_COLLECTION}})
    articles = load_articles(INPUT_JSON)
    if not articles:
        logger.warning("no_articles_found", extra={"extra": {"file": INPUT_JSON}})
        return

    embedder = OpenAIEmbedder(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

    # Probe first article to determine vector size
    probe_text = articles[0].embedding_text()
    probe_vec = await embedder.embed(probe_text, dimensions=int(EMBED_DIM_OVERRIDE) if EMBED_DIM_OVERRIDE else None)
    vector_size = len(probe_vec)

    # Ensure collection exists in Qdrant
    store = QdrantStore(host=QDRANT_HOST, rest_port=QDRANT_REST_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
    await store.ensure_collection(QDRANT_COLLECTION, vector_size)

    # Embed all articles concurrently
    embedded = await embed_articles(embedder, articles, concurrency=EMBED_CONCURRENCY,
                                    dim=int(EMBED_DIM_OVERRIDE) if EMBED_DIM_OVERRIDE else None)
    logger.info("embedding_done", extra={"extra": {"count": len(embedded), "vector_size": vector_size}})

    # Upsert articles in batches
    total = 0
    for chunk in batch(embedded, UPSERT_BATCH_SIZE):
        points = []
        for art, vec in chunk:
            payload = {
                "title": art.title,
                "link": art.link,
                "ticker": art.ticker,
                "full_text": art.full_text
            }
            points.append(
                qmodels.PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload=payload,
                )
            )
        await store.upsert_points(QDRANT_COLLECTION, points)
        total += len(points)
        logger.info("batch_upsert_ok", extra={"extra": {"batch": len(points), "total": total}})

    logger.info("ingestion_complete", extra={"extra": {"inserted": total, "collection": QDRANT_COLLECTION}})


if __name__ == "__main__":
    asyncio.run(main())
