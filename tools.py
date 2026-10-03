from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from ollama import WebFetchResponse, web_fetch, web_search


def fetch_webpage_content(url: str) -> WebFetchResponse:
    """
    Fetch webpage and return results as WebFetchResponse
        class WebFetchResponse(SubscriptableBaseModel):
            title: Optional[str] = None
            content: Optional[str] = None
            links: Optional[Sequence[str]] = None

    Raises:
        RuntimeError: If the page cannot be fetched.
    """
    try:
        response = web_fetch(url)
    except Exception as error:
        raise RuntimeError(f"Error fetching {url}: {error}") from error
    if not response:
        raise RuntimeError(f"Error fetching {url}: empty response")
    return response

@tool
def search_web(
        query: str,
        max_results: Annotated[int, InjectedToolArg] = 1,
    ) -> str:

    """
    Search the web for information on a given query.

    Uses Ollama web search to discover relevant URLs, then fetches and returns
    full webpage content as Markdown.

    Args:
    query: Search query to execute
    max_results: Maximum number of results to return (default: 1)

    Returns:
        Formatted search results with full webpage content
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    search_results = web_search(query=query, max_results=max_results)
    result_texts = []
    for result in search_results.results:
        url = result.url
        webpage = fetch_webpage_content(url)

        title = webpage.title or url
        content = webpage.content or ""
        links = webpage.links or []

        result_texts.append(f"##{title}\n**URL:** {url}\n\n{content}\n\nOther Links: {links}\n---")
    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(
        result_texts
    )
