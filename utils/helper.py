# You can add common utilities here, e.g., safe async wrapper
import asyncio
import logging

logger = logging.getLogger("helper")

async def safe_async(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return None
