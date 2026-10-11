import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


### data in st session
if "document_upload" not in st.session_state:
    st.session_state.document_upload = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    ### load the documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ### split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    ### Embedding & Vector_db
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    st.session_state.vector_store = vector_db

    ### Create an agent + tool
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve documents relevant to a query from the knowledge base."""
        results = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    system_prompt = (
        "You are a helpful assistant that answers questions using retrieved context. "
        "Your knowledge base consists of the details from the uploaded document. "
        "ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."
    )

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    st.session_state.agent = agent
    st.session_state.document_upload = True


### upload UI
if not st.session_state.document_upload:
    uploaded = st.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_file"
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
        st.rerun()


### Chat UI
if st.session_state.document_upload and st.session_state.agent:
    # show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about the uploaded PDF files")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"configurable": {"thread_id": "1"}},
                )
                answer = response["messages"][-1].content
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# while true:
#       query= input("User:")
#       if(query=="quit"):
#             break

#       response = agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         config={"configurable": {"thread_id": "1"}},
#             )
#       result = response["messages"][-1].content

#       print("Ai:", result)


