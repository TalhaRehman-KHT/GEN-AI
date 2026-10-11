from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("🤖 Q&A Bot")
st.markdown("This is a simple Q&A bot that uses the Google Generative AI API to answer questions. You can ask any question, and the bot will provide an answer based on its knowledge.")

if "messages" not in st.session_state:
    st.session_state.messages = []

    # Clear chat button in the sidebar
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Show the saved history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat box pinned to the bottom
query = st.chat_input("Ask any question...")
if query:
    # Save and show the user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    # Get the reply (full history is sent, so the bot remembers the conversation)
    with st.spinner("Thinking..."):
        res = llm.invoke(st.session_state.messages)

    # Save and show the AI message
    st.session_state.messages.append({"role": "assistant", "content": res.text})
    st.chat_message("assistant").markdown(res.text)