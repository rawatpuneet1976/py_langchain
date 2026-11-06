import importlib
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

GoogleSerperAPIWrapper = None
for module_name, class_name in [
    ("langchain_serper", "GoogleSerperAPIWrapper"),
    ("langchain_serper", "SerperAPIWrapper"),
    ("langchain_community.utilities", "GoogleSerperAPIWrapper"),
]:
    try:
        module = importlib.import_module(module_name)
        GoogleSerperAPIWrapper = getattr(module, class_name)
        break
    except (ImportError, AttributeError):
        continue

if GoogleSerperAPIWrapper is None:
    raise ImportError(
        "No compatible Serper wrapper was found. Install the standalone Serper package or keep using langchain-community."
    )

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY is missing. Add it to your .env file before running this notebook.")

model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
memory = MemorySaver()


agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a agent and can search for any question on google."
)


while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Bye 👋")
        break

    response = agent.invoke(
                {"messages":[{"role":"user", "content":query}]}, 
                {"configurable": {"thread_id": "1"}}, 
            )
    print("AI:",  response["messages"][-1].content)