from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()


agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a helpful assistant that can perform Google searches. You have access to the following tool: search. Use this tool to answer questions about any topic."
)


while True:
    query = input("user: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the agent. Goodbye!👋")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}}
    )

    print("agent: ", response["messages"][-1].text)