from deepagents import create_deep_agent
import sqlite3
from settings import settings
from tools import web_search_tool, save_learnings_tool, load_learnings_tool
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

research_agent_prompt = ""
with open('agent_system_prompt.txt', mode='r', encoding='utf-8') as file:
    research_agent_prompt = file.read()


aes_key = settings.LANGGRAPH_AES_KEY.encode('utf-8')

serde = EncryptedSerializer.from_pycryptodome_aes(key=aes_key)
connection = sqlite3.connect('checkpoint.db', check_same_thread=False)
checkpointer = SqliteSaver(connection, serde=serde)

agent = create_deep_agent(
    model="ollama:qwen3:1.7b",
    tools=[web_search_tool, save_learnings_tool, load_learnings_tool],
    system_prompt=research_agent_prompt,
    checkpointer=checkpointer
)

thread_config = {"configurable": {"thread_id": "1"}}
while True:
    query = input("User: ")
    if query.lower() in ['exit', 'quit']:
        break
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, thread_config)["messages"][-1].content
    print(result)
