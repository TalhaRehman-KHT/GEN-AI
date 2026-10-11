from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from typing import Literal
from state import BlogState
from agent import get_llm, researcher_agent, writer_agent, editor_agent


MAX_REVISION = 3


#### Nodes - Edges
### Defining the nodes

def researcher_node(state: BlogState):
    """
    Researcher Agent generates (or receives) the research outline.
    """
    llm = get_llm()
    feedback = state.research_feedback

    research_data = researcher_agent(
        llm=llm,
        topic=state.topics,
        audience=state.audience,
        feedback=feedback
    )

    state.research = research_data
    state.research_feedback = ""

    return state


def human_review_research_node(state: BlogState):
    """
    Pause and ask the human to approve the research or give feedback.
    """
    decision = interrupt({
        "stage": "researcher_review",
        "research": state.research,
        "instruction": (
            "Reply with approve to continue writing, "
            "or describe what to change to send it back to researcher."
        )
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = (
            "approve"
            if text.lower() in ["approve", "approved", "ok", "yes", "proceed", ""]
            else "revise"
        )
        feedback = "" if action == "approve" else text

    state.research_feedback = feedback

    return state


def writer_node(state: BlogState):
    """
    Writer Agent produces the full draft blog (or receives it).
    """
    llm = get_llm()

    draft_data = writer_agent(
        llm=llm,
        topic=state.topics,
        audience=state.audience,
        research=state.research,
        feedback=state.draft_feedback
    )

    state.draft = draft_data
    state.draft_feedback = ""

    return state


def human_review_draft_node(state: BlogState):
    """
    Pause and ask the human to approve the draft or give feedback.
    """
    decision = interrupt({
        "stage": "draft_review",
        "draft": state.draft,
        "instruction": (
            "Reply with approve to continue to send it to the editor, "
            "or describe what to change to send it back to writer."
        )
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = (
            "approve"
            if text.lower() in ["approve", "approved", "ok", "yes", "proceed", ""]
            else "revise"
        )
        feedback = "" if action == "approve" else text

    state.draft_feedback = feedback
    if feedback:
        state.revision_count += 1

    return state


def editor_node(state: BlogState):
    """
    Editor Agent produces the final blog.
    """
    llm = get_llm()

    final = editor_agent(
        llm=llm,
        topic=state.topics,
        draft=state.draft
    )

    state.final_blog = final

    return state


def publish_node(state: BlogState):
    """
    Pause and ask the human whether to publish the final blog (yes/no).
    """
    decision = interrupt({
        "stage": "publish",
        "final_blog": state.final_blog,
        "instruction": "Reply yes to publish or no to discard.",
    })

    if isinstance(decision, dict):
        decision = decision.get("action", "no")
    state.published = str(decision).lower() in ["yes", "y", "publish", "approve"]

    return state


### Conditional Edeges

def route_after_research_review(state:BlogState) ->Literal["researcher", "writer"]:
    if state.research_feedback:
        return "researcher"
    else:
        return "writer"



def route_after_draft_review(state:BlogState) ->Literal["writer", "editor"]:
    if state.draft_feedback and state.revision_count <= MAX_REVISION :
        return "writer"
    else:
        return "editor"




### Build and compile the graph

def build_blog_graph():
    builder = StateGraph(BlogState)

    ### Add nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("review_researcher", human_review_research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review", human_review_draft_node)
    builder.add_node("editor", editor_node)
    builder.add_node("publish", publish_node)

    ### Add edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "review_researcher")

    builder.add_conditional_edges(
        "review_researcher",
        route_after_research_review,
        {
            "researcher": "researcher",
            "writer": "writer"
        }
    )

    builder.add_edge("writer", "review")

    builder.add_conditional_edges(
        "review",
        route_after_draft_review,
        {
            "writer": "writer",
            "editor": "editor"
        }
    )

    builder.add_edge("editor", "publish")
    builder.add_edge("publish", END)

    GRAPH = builder.compile(checkpointer=InMemorySaver())

    return GRAPH

