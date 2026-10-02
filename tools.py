from langchain_core.tools import tool
from settings import settings
from ollama import web_search


@tool
def web_search_tool(query: str, max_results=3):
    """
    Performs a web search.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return (default=3).
    
    Returns:
        WebSearchResponse with the search results
    Raises:
        ValueError: If OLLAMA_API_KEY environment variable is not set
    """
    return web_search(query=query, max_results=max_results)


@tool
def save_learnings_tool(note: str):
    """Saves an important learning note to the project log."""
    with open(settings.LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"- {note} \n\n")
    return f"Note saved: {note}"

@tool
def load_learnings_tool():
    """Loads the history and context required if missing from the current query."""
    if not settings.LOG_FILE.exists():
        return "Learning Log is empty."
    with open(settings.LOG_FILE, 'r', encoding='utf-8') as log:
        return log.read()
