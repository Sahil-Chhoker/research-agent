import sqlite3
from typing import Callable

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from prompts import (
    build_instructions,
    build_researcher_instructions,
)
from settings import settings
from tools import find_saved_learning, save_learning, save_quiz, search_web

max_concurrent_research_units = 3
max_researcher_iterations = 3

def create_checkpointer(
    database_path: str = "checkpoint.db",
    encryption_key: str | None = None,
) -> SqliteSaver:
    """Create the encrypted SQLite checkpointer used by the agent."""
    key = encryption_key or settings.LANGGRAPH_AES_KEY
    if not key:
        raise ValueError("LANGGRAPH_AES_KEY must be configured")

    serde = EncryptedSerializer.from_pycryptodome_aes(key=key.encode('utf-8'))
    connection = sqlite3.connect(database_path, check_same_thread=False)
    return SqliteSaver(connection, serde=serde)


def create_agent(
    *,
    model_name: str = settings.MODEL,
    database_path: str = "checkpoint.db",
    encryption_key: str | None = None,
):
    """Assemble the model, tools, middleware, sub-agent, and checkpointer."""
    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent. Give one topic at a time.",
        "system_prompt": build_researcher_instructions(),
        "tools": [search_web],
    }
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_deep_agent(
        model=model,
        tools=[search_web, save_learning, find_saved_learning, save_quiz],
        system_prompt=build_instructions(
            max_concurrent_research_units=max_concurrent_research_units,
            max_researcher_iterations=max_researcher_iterations,
        ),
        subagents=[research_sub_agent],
        middleware=[TodoListMiddleware(), SummarizationMiddleware(model=model)],
        checkpointer=create_checkpointer(database_path, encryption_key),
    )


def run_cli(
    agent,
    *,
    thread_id: str = "1",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive interface against an already-configured agent."""
    thread_config = {"configurable": {"thread_id": thread_id}}
    while True:
        try:
            query = input_fn("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return

        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            thread_config,
        )
        output_fn(str(result["messages"][-1].content))


def main() -> None:
    run_cli(create_agent())


if __name__ == "__main__":
    main()
