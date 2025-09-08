from langgraph.graph import StateGraph, START, END
from pipeline.nodes import (
    semantic_search_node, summarize_rag_node, extract_ticker_node,
    yahoo_fetch_node, summarize_yahoo_node, rss_fallback_node
)
from pipeline.nodes import PipelineState
from utils.logging import setup_logger

logger = setup_logger("nodes")


# --------------------------
# Routing functions
# --------------------------
def route_after_semantic(state: PipelineState):
    """
    Decide next node after semantic search.
    If articles are found, go to summarize_rag.
    Otherwise, go to extract_ticker for Yahoo fetching.
    """

    return "summarize_rag" if state["articles"] else "extract_ticker"

def route_after_yahoo(state: PipelineState):
    """
    Decide next node after Yahoo fetch.
    If articles are found, go to summarize_yahoo.
    Otherwise, go to rss_fallback as a last resort.
    """

    return "summarize_yahoo" if state["articles"] else "rss_fallback"


# --------------------------
# Build RAG graph
# --------------------------
def build_rag_graph():
    """
    Build a retrieval-augmented generation (RAG) workflow as a state graph.
    
    Workflow steps:
        1. Start → semantic search
        2. semantic_search → summarize_rag OR extract_ticker
        3. summarize_rag → END
        4. extract_ticker → yahoo_fetch
        5. yahoo_fetch → summarize_yahoo OR rss_fallback
        6. summarize_yahoo → END
        7. rss_fallback → END
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("summarize_rag", summarize_rag_node)
    workflow.add_node("extract_ticker", extract_ticker_node)
    workflow.add_node("yahoo_fetch", yahoo_fetch_node)
    workflow.add_node("summarize_yahoo", summarize_yahoo_node)
    workflow.add_node("rss_fallback", rss_fallback_node)

    # Start → semantic search
    workflow.add_edge(START, "semantic_search")

    # Conditional: semantic_search → summarize_rag OR extract_ticker
    workflow.add_conditional_edges("semantic_search", route_after_semantic, {
        "summarize_rag": "summarize_rag",
        "extract_ticker": "extract_ticker"
    })

    # summarize_rag → END
    workflow.add_edge("summarize_rag", END)

    # extract_ticker → yahoo_fetch
    workflow.add_edge("extract_ticker", "yahoo_fetch")

    # Conditional: yahoo_fetch → summarize_yahoo OR rss_fallback
    workflow.add_conditional_edges("yahoo_fetch", route_after_yahoo, {
        "summarize_yahoo": "summarize_yahoo",
        "rss_fallback": "rss_fallback"
    })

    # Summarize nodes → END
    workflow.add_edge("summarize_yahoo", END)
    workflow.add_edge("rss_fallback", END)

    return workflow.compile()
