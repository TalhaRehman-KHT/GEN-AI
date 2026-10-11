import uuid

import streamlit as st
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from graph import build_blog_graph

st.set_page_config(page_title="Blog Generator", page_icon="📝")
st.title("📝 Multi-Agent Blog Generator")

ss = st.session_state
if "graph" not in ss:
    ss.graph = build_blog_graph()  # keeps the InMemorySaver checkpointer alive across reruns
    ss.thread_id = None


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": ss.thread_id}}


def run(payload):
    with st.spinner("Agents are working..."):
        ss.graph.invoke(payload, config())
    ss.step = ss.get("step", 0) + 1  # new key below -> fresh feedback box
    st.rerun()


# ---- Start ----
with st.form("start"):
    topic = st.text_input("Topic")
    audience = st.text_input("Audience", "general readers")
    if st.form_submit_button("Start", type="primary") and topic.strip():
        ss.thread_id = str(uuid.uuid4())
        run({"topics": topic.strip(), "audience": audience.strip()})

if not ss.thread_id:
    st.stop()

snap = ss.graph.get_state(config())
values = snap.values
pending = snap.interrupts[0].value if snap.interrupts else None
stage = pending["stage"] if pending else None

# ---- Stage outputs ----
if values.get("research"):
    with st.expander("🔎 Research", expanded=stage == "researcher_review"):
        st.markdown(values["research"])
if values.get("draft"):
    with st.expander(f"✍️ Draft (revisions: {values.get('revision_count', 0)})", expanded=stage == "draft_review"):
        st.markdown(values["draft"])
if values.get("final_blog"):
    with st.expander("✅ Edited blog", expanded=True):
        st.markdown(values["final_blog"])

# ---- Human review steps ----
if stage in ("researcher_review", "draft_review"):
    label = "research" if stage == "researcher_review" else "draft"
    st.subheader(f"Review the {label}")
    feedback = st.text_area("Feedback (what should change?)", key=f"fb_{ss.get('step', 0)}")
    c1, c2 = st.columns(2)
    if c1.button("Approve", type="primary", width="stretch"):
        run(Command(resume={"action": "approve"}))
    if c2.button("Give feedback", width="stretch", disabled=not feedback.strip()):
        run(Command(resume={"action": "revise", "feedback": feedback.strip()}))

elif stage == "publish":
    st.subheader("Publish this blog?")
    c1, c2 = st.columns(2)
    if c1.button("Yes, publish", type="primary", width="stretch"):
        run(Command(resume={"action": "yes"}))
    if c2.button("No", width="stretch"):
        run(Command(resume={"action": "no"}))

elif not snap.next:  # graph finished
    if values.get("published"):
        st.success("Blog published!")
    else:
        st.info("Blog not published.")

if values.get("final_blog"):
    st.download_button("Download blog (.md)", values["final_blog"],
                       file_name=f"{values['topics'][:40].replace(' ', '_')}.md", mime="text/markdown")
