import json
from contextlib import closing
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from ollama import WebFetchResponse, web_fetch, web_search
from setup_learning import (
    QuizAttempt,
    find_learning_records,
    init_db,
    save_learning_record,
    save_quiz_attempt,
)

MAX_PAGE_CHARACTERS = 12_000
MAX_RELATED_LINKS = 10


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
        max_results: Annotated[int, InjectedToolArg] = 2,
    ) -> str:

    """
    Search the web for information on a given query.

    Uses Ollama web search to discover relevant URLs, then fetches and returns
    bounded webpage content as Markdown.

    Args:
    query: Search query to execute
    max_results: Maximum number of results to return (default: 2)

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
        content = (webpage.content or "")[:MAX_PAGE_CHARACTERS]
        links = list(webpage.links or [])[:MAX_RELATED_LINKS]

        result_texts.append(
            f"## {title}\n**URL:** {url}\n\n{content}\n\n"
            f"Other Links: {links}\n---"
        )
    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(
        result_texts
    )


@tool
def save_learning(
    topic: str,
    summary: str,
    concepts: list[str],
    source_urls: list[str],
    confidence: float,
) -> str:
    """Save a learning record to the database."""
    with closing(init_db()) as connection:
        record = save_learning_record(
            connection=connection,
            record={
                "topic": topic,
                "summary": summary,
                "concepts": concepts,
                "source_urls": source_urls,
                "confidence": confidence,
            },
        )
    return f"Saved learning record with ID {record.id}"


@tool
def find_saved_learning(topic: str, limit: int = 10) -> str:
    """Find saved learning records matching a topic.

    Results are returned as JSON so the model receives a stable, structured
    representation rather than a Python-specific object.
    """
    with closing(init_db()) as connection:
        records = find_learning_records(connection=connection, topic=topic, limit=limit)
    if not records:
        return f"No learning records found for topic '{topic}'"
    return json.dumps([record.model_dump() for record in records])


@tool
def save_quiz(
    topic: str,
    questions: list[dict[str, Any]],
    score: float | None = None,
) -> str:
    """Save a completed or in-progress quiz and its answers."""
    attempt = QuizAttempt(topic=topic, questions=questions, score=score)
    with closing(init_db()) as connection:
        saved_attempt = save_quiz_attempt(connection=connection, attempt=attempt)
    return f"Saved quiz attempt with ID {saved_attempt.id}"
