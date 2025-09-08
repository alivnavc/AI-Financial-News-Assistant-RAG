import asyncio
from clients.openai_client import openai_client
from clients.qdrant_client import qdrant_client
from clients.yahoo_client import fetch_yahoo_news, fetch_yahoo_rss_latest
from pipeline.prompts import SUMMARIZER_PROMPT
from utils.logging import setup_logger, log_exception_sync
from typing_extensions import TypedDict
from openai import OpenAI
import os
from pydantic import BaseModel, Field

logger = setup_logger("nodes")


class TickerStructure(BaseModel):
    """Pydantic model for GPT-extracted ticker"""
    ticker: str = Field(..., description="Company ticker extracted from query")

class PipelineState(TypedDict):
    """State dictionary passed between nodes in the pipeline"""
    query: str
    chat_history: str
    articles: list
    summary: str
    ticker: str


# -------------------------
# Nodes
# -------------------------
def semantic_search_node(state: PipelineState):
    """
    Perform semantic search in Qdrant for the user query.
    Updates state['articles'] with relevant hits above threshold.
    """
    try:
        
        embedding = openai_client.get_embedding(state["query"])
        hits = qdrant_client.search(
            collection_name="news_articles",
            query_vector=embedding,
            limit=10
        )
        filtered = [h for h in hits if getattr(h, "score", 0) >= 0.6]
        state["articles"] = filtered
        logger.info(f"Semantic search returned {len(filtered)} articles")
        return state
    except Exception as e:
        log_exception_sync(logger, "semantic_search_node failed", e)
        state["articles"] = []
        return state


def summarize_rag_node(state: PipelineState):
    """
    Summarize articles retrieved from DB using OpenAI LLM.
    If no articles found, returns a default message.
    """
    try:
        if not state["articles"]:
            state["summary"] = "No articles found"
            return state
        articles_text = "\n".join(
            [f"- {a.payload['title']}: {a.payload.get('full_text','')} - Source: {a.payload.get('link','')} - Ticker: {a.payload.get('ticker','')}" for a in state["articles"]]
        )
        prompt = SUMMARIZER_PROMPT.format(
            chat_history=state["chat_history"],
            query=state["query"],
            articles=articles_text
        )
        summary = openai_client.chat(prompt)
        state["summary"] = summary
        logger.info("RAG summarization completed")
        return state
    except Exception as e:
        log_exception_sync(logger, "summarize_rag_node failed", e)
        state["summary"] = "Summarization failed"
        return state


def extract_ticker_node(state: PipelineState):
    """
    Extract company ticker from user query using GPT.
    If not inferable, returns main company name.
    Updates state['ticker'].
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", "")

    async def extract_ticker_async(query: str, chat_history: str):
        prompt = f"""
Based on the conversation history: {chat_history}
Extract the company ticker from the following user query.
If not present or inferable from history, provide the main company name.
Return strictly in JSON format with a single key 'ticker' as below
{{"ticker": "TICKER_SYMBOL"}}

Query: "{query}"
"""
        try:
            openai_clients = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai_clients.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                response_format=TickerStructure
            )
            ticker_obj = response.choices[0].message.parsed
            return ticker_obj.ticker
        except Exception as e:
            logger.error(f"Ticker extraction failed: {e}")
            return None

    try:
        ticker = asyncio.run(extract_ticker_async(query, chat_history))
        state["ticker"] = ticker
        logger.info(f"Extracted ticker: {state['ticker']}")
        return state
    except Exception as e:
        log_exception_sync(logger, "extract_ticker_node failed", e)
        state["ticker"] = None
        return state


def yahoo_fetch_node(state: PipelineState):
    """
    Fetch articles from Yahoo Finance using extracted ticker.
    Updates state['articles'].
    """ 
    try:
        if not state.get("ticker"):
            logger.warning("No ticker found for Yahoo fetch")
            state["articles"] = []
            return state
        articles = fetch_yahoo_news(state["ticker"])
        state["articles"] = articles
        logger.info(f"Yahoo fetch returned {len(articles)} articles")
        return state
    except Exception as e:
        log_exception_sync(logger, "yahoo_fetch_node failed", e)
        state["articles"] = []
        return state


def summarize_yahoo_node(state: PipelineState):
    """
    Summarize Yahoo articles using the same RAG summarizer node.
    """
    return summarize_rag_node(state)


def rss_fallback_node(state: PipelineState):
    """
    Fallback: fetch latest article from Yahoo RSS feed if no articles found.
    Updates state['summary'].
    """
    try:
        if not state.get("ticker"):
            logger.warning("No ticker for RSS fallback")
            state["summary"] = "No news found."
            return state
        rss = fetch_yahoo_rss_latest(state["ticker"])
        if rss:
            state["summary"] = f"No DB/Yahoo articles. Latest RSS article: {rss['title']} ({rss['link']})"
        else:
            state["summary"] = "No news found."
        logger.info("RSS fallback executed")
        return state
    except Exception as e:
        log_exception_sync(logger, "rss_fallback_node failed", e)
        state["summary"] = "No news found."
        return state
