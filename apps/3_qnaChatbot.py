### LLM
### TOOL  Google Search
### Agent 
### Memory
### Streaming
###  WEB interface

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b" , streaming= True)
search = GoogleSerperAPIWrapper()
tools = [search.run]


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an AI Agent that can Search on Google and get updated data to understand the user"
)


## web Interface
st.subheader("QnA Chat Bot with Google Search")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask Anything!")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({
        "role": "user",
        "content": query
    })

    res = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
                
                )
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message= ""

        for chunk in res:
            if isinstance(chunk[0], AIMessageChunk) and chunk[0].content:
                message = message + chunk[0].content
                space.write(message)
        
        st.session_state.history.append({
                "role": "assistant",
                "content": message
            })

# ### LLM
# ### TOOL - Google Search
# ### Agent
# ### Memory
# ### WEB interface

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_groq import ChatGroq
# from langchain_community.utilities import GoogleSerperAPIWrapper
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import MemorySaver
# import streamlit as st


# # ---------------- PAGE ----------------

# st.set_page_config(
#     page_title="QnA Chatbot",
#     page_icon="🤖"
# )


# # ---------------- INITIALIZE ONCE ----------------

# if "memory" not in st.session_state:

#     st.session_state.memory = MemorySaver()

#     st.session_state.llm = ChatGroq(
#         model="openai/gpt-oss-20b"
#     )

#     st.session_state.search = GoogleSerperAPIWrapper()

#     st.session_state.tools = [
#         st.session_state.search.run
#     ]

#     st.session_state.agent = create_agent(
#         model=st.session_state.llm,
#         tools=st.session_state.tools,
#         checkpointer=st.session_state.memory,
#         system_prompt=(
#             "You are an AI Agent that can search Google and get "
#             "updated information. Answer the user's questions accurately. "
#             "Use previous conversation context when the user refers to "
#             "something discussed earlier."
#         )
#     )


# # ---------------- CHAT HISTORY FOR UI ----------------

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# # ---------------- WEB INTERFACE ----------------

# st.title("🤖 QnA Chatbot with Google Search")


# # Display previous messages
# for message in st.session_state.messages:

#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# # ---------------- USER INPUT ----------------

# query = st.chat_input("Ask Anything!")


# if query:

#     # Show user message
#     with st.chat_message("user"):
#         st.markdown(query)

#     # Save user message for UI
#     st.session_state.messages.append({
#         "role": "user",
#         "content": query
#     })


#     # ---------------- AGENT ----------------

#     res = st.session_state.agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": query
#                 }
#             ]
#         },
#         {
#             "configurable": {
#                 "thread_id": "user-1"
#             }
#         }
#     )


#     # ---------------- AI RESPONSE ----------------

#     answer = res["messages"][-1].text


#     # Show AI response
#     with st.chat_message("assistant"):
#         st.markdown(answer)


#     # Save AI response for UI
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer
#     })