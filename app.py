import streamlit as st
import asyncio
from pipeline.graph import build_rag_graph
from pipeline.nodes import PipelineState
from langchain.memory import ConversationBufferWindowMemory
from utils.logging import setup_logger, log_exception_sync

logger = setup_logger("app")  #

st.set_page_config(page_title="Financial News RAG Chatbot")

# -----------------
# Session State
# -----------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------
# Chat Input
# -----------------
st.title("Financial News RAG Chatbot")
direct_mode = st.toggle("Direct Summarize Mode (paste article)", value=False)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter financial query or article:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        memory = st.session_state.memory.load_memory_variables({})
        chat_history_str = "\n".join([f"Human: {m.content}\nAssistant: {m.content}" for m in memory.get("history", [])])
        initial_state = {
            "query": prompt,
            "chat_history": chat_history_str,
            "articles": [],
            "summary": "",
            "ticker": ""
        }
        graph = build_rag_graph()
        with st.spinner("Thinking..."):
            final_state = asyncio.run(graph.ainvoke(initial_state))
        summary = final_state.get("summary", "No response")
        st.markdown(summary)
        st.session_state.memory.save_context({"input": prompt}, {"output": summary})
        st.session_state.messages.append({"role": "assistant", "content": summary})
