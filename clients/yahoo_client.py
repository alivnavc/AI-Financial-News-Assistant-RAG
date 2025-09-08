from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from utils.logging import setup_logger

logger = setup_logger("yahoo_client")
# Initialize the Yahoo Finance tool from LangChain Community
# This tool fetches financial news articles for a given ticker
yahoo_tool = YahooFinanceNewsTool()

def fetch_yahoo_news(ticker: str):
    """
    Fetch financial news articles for a given stock ticker from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT').

    Returns:
        List[DummyArticle]: A list of article-like objects containing title, link, and full text snippet.
    """
    try:
        results = yahoo_tool.run(ticker)
        if results and isinstance(results, list):
            class DummyArticle:
                def __init__(self, title, link, full_text):
                    self.payload = {"title": title, "link": link, "full_text": full_text}
            articles = []
            for a in results:
                if isinstance(a, dict):
                    title = a.get("title", "")
                    link = a.get("link", "")
                    snippet = a.get("snippet", title)
                else:
                    title = str(a)
                    link = ""
                    snippet = title
                articles.append(DummyArticle(title, link, snippet))
            return articles
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed: {e}")
    return []

def fetch_yahoo_rss_latest(ticker: str):
    """
    Fetch the latest news headline for a given stock ticker using Yahoo Finance RSS feed.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict or None: Returns a dictionary with 'title', 'link', 'published' of the latest article,
                      or None if the feed is empty or parsing fails.
    """
    import feedparser
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return None
    try:
        latest = feed.entries[0]
        return {
            "title": latest.get("title", ""),
            "link": latest.get("link", ""),
            "published": latest.get("published", "")
        }
    except Exception as e:
        logger.warning(f"RSS parsing failed: {e}")
        return None
