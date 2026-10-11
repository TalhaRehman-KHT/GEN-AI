# Multi-Agent Blog Generator

A LangGraph pipeline with a Streamlit UI that researches, writes and edits a blog post, pausing for human review at each step.

```
researcher → review research → writer → review draft → editor → publish yes/no
     ↑______ feedback ______|      ↑_____ feedback _____|
```

- **Researcher**: builds research notes for the topic and audience.
- **Writer**: turns the notes into a draft.
- **Editor**: polishes the draft into the final blog.
- At each review you can approve or send feedback back to that agent. Draft revisions are capped at `MAX_REVISION` (3) in `graph.py`.

## Files

| File | Purpose |
|---|---|
| `agent.py` | LLM setup (Groq) and the three agent prompts/functions |
| `state.py` | `BlogState`, the shared graph state |
| `graph.py` | Nodes, human-review interrupts, routing, `build_blog_graph()` |
| `app.py` | Streamlit UI |

## Setup

1. Python 3.10+ recommended. Create and activate a virtual environment:
   ```bash
   python -m venv env
   env\Scripts\activate        # Windows
   source env/bin/activate     # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your Groq API key to a `.env` file in this folder (or any parent folder):
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get a key at https://console.groq.com/keys.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501:

1. Enter a topic and audience, then click **Start**.
2. Review the research: **Approve**, or type feedback and click **Give feedback**.
3. Review the draft the same way.
4. Read the edited blog and choose **Yes, publish** or **No**.
5. Use **Download blog (.md)** to save it.

To test the agents without the UI: `python agent.py`.

## Notes

- State is kept in memory (`InMemorySaver`), so a run is lost if you refresh the browser tab or restart the server.
- The default model is `openai/gpt-oss-20b` on Groq; change it in `get_llm()` in `agent.py`.
