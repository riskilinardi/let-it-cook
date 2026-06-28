from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.agent.calculator import calculator
from app.agent.retriever import search_documents
from app.config import SEARCH_MAX_RESULTS

_tavily = TavilySearch(max_results=SEARCH_MAX_RESULTS, topic="general")


@tool
def web_search(query: str) -> str:
    """Search the web for current events or general world knowledge.
    Use this for facts about the wider world or recent information, not for things
    contained in the user's own documents.
    """
    return str(_tavily.invoke({"query": query}))


TOOLS = [web_search, search_documents, calculator]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}