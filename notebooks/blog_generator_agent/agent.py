# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate



# ### Get LLM
# def get_llm(model_name:str="openai/gpt-oss-20b",temperatur:float=0.5):
#     api_key=os.getenv("GROQ_API_KEY")
#     llm = ChatGroq(model=model_name,temperature=temperatur, api_key=api_key)
#     return llm


# ### Agents ###

# # 1 ## Researcher Prompt
# Researcher_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a research agent for a blog.

# Research the given topic and provide accurate, relevant, and up-to-date information.

# - Find the key facts and important points.
# - Use reliable sources.
# - Avoid making up information.
# - Include useful examples and statistics when available.
# - Mention the sources used.
# - Organize the research in a simple, clear format that a blog writer can easily use.
# """),
#     ("human", "Topic: {topic}\nAudience: {audience}\n{revision_hints}\n\n Write the research outline now."),
# ])


# # 2 ### Researcher Agent 
# def researcher_agent(llm:ChatGroq , topic:str ,audience:str , feedback:str="") ->str:
#     researcher_hints= f"The human provided this feedback on your previous research Please address it: {feedback} "
#     if not feedback:
#         revision_hints="This is your first Attempt "
#     chain = Researcher_prompt | llm 

#     result= chain.invoke({
#         "topic":topic,
#         "audience":audience,
#         "revision_hints":researcher_hints

#         })

#     return result.content


# # 3 ### Writer Agent 
# Writer_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a professional blog writer.

# Write a clear, engaging, and well-structured blog post using ONLY the research notes provided.

# - Start with a catchy title and a short introduction that hooks the reader.
# - Use clear headings and subheadings to organize the content.
# - Write in simple language that suits the target audience.
# - Use the facts, examples, and statistics from the research notes. Do not invent new facts.
# - Keep paragraphs short and easy to read.
# - Mention the sources from the research notes where relevant.
# - End with a short conclusion and a takeaway or call to action.
# - If revision hints are given, apply them carefully while rewriting.
# """),
#     ("human", """Topic: {topic}
# Audience: {audience}

# Research notes:
# {research}

# Revision hints: {revision_hints}

# Write the blog post now."""),
# ])



# # 4 ### Writer  Agent 
# def writer_agent(
#     llm: ChatGroq,
#     topic: str,
#     audience: str,
#     research: str,
#     feedback: str = "",
# ) -> str:
#     if feedback:
#         revision_hints = (
#             f"The human gave this feedback on your previous draft. "
#             f"Please address it: {feedback}"
#         )
#     else:
#         revision_hints = "This is your first attempt."

#     chain = Writer_prompt | llm
#     result = chain.invoke({
#         "topic": topic,
#         "audience": audience,
#         "research": research,
#         "revision_hints": revision_hints,
#     })
#     return str(result.content)


# # 5 ### Editor 
# editor_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a professional blog editor.

# Review the blog draft and return an improved, publication-ready version.

# - Fix grammar, spelling, punctuation, and awkward sentences.
# - Improve clarity, flow, and readability for the target audience.
# - Keep the title, headings, and structure clear and consistent.
# - Check the draft against the research notes. Remove or correct any claim that is not supported by them. Do not add new facts.
# - Keep the original meaning, tone, and sources.
# - Shorten repetitive or unnecessary parts.
# - If revision hints are given, apply them carefully.
# - Blog writing should be in human writing way not in AI apply humanized way , donot use complex words use simple words
# - Return ONLY the final edited blog post, with no comments or explanations.
# """),
#     ("human", """Topic: {topic}
# Blog draft:
# {draft}



# Return the publish blog post"""),
# ])



# # 6 ### Editor Agent
# def editor_agent(
#     llm: ChatGroq,
#     topic: str,
#     draft: str,
# ) -> str:
#     chain = editor_prompt | llm

#     result = chain.invoke({
#         "topic": topic,
#         "draft": draft,
#     })
#     return str(result.content)




"""
Blog Multi-Agent Pipeline
=========================
Flow:  Researcher  ->  Writer  ->  Editor

1. researcher_agent : topic + audience        -> research notes
2. writer_agent     : research notes          -> blog draft
3. editor_agent     : blog draft              -> final, publish-ready blog
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

load_dotenv()  # loads GROQ_API_KEY from the .env file


# ----------------------------------------------------------------------------
# LLM helper
# ----------------------------------------------------------------------------
def get_llm(
    model_name: str = "openai/gpt-oss-20b",
    temperature: float = 0.5,
) -> ChatGroq:
    """Create a Groq chat model.

    Tip: use a low temperature (0.2 - 0.3) for research/editing and a
    slightly higher one (0.5 - 0.7) for writing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    return ChatGroq(model=model_name, temperature=temperature, api_key=SecretStr(api_key))


# ----------------------------------------------------------------------------
# Helper: build the "revision hints" text used by the agents
# ----------------------------------------------------------------------------
def build_revision_hints(feedback: str, what: str) -> str:
    """Return the hint text for the prompt.

    feedback : human feedback (empty string on the first attempt)
    what     : what the feedback is about, e.g. "research" or "draft"
    """
    if feedback:
        return (
            f"The human gave this feedback on your previous {what}. "
            f"Please address it: {feedback}"
        )
    return "This is your first attempt."


# ============================================================================
# 1. RESEARCHER AGENT
# ============================================================================
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research agent for a blog.

Research the given topic and provide accurate, relevant, and up-to-date information.

- Find the key facts and important points.
- Use reliable sources.
- Avoid making up information.
- Include useful examples and statistics when available.
- Mention the sources used.
- Organize the research in a simple, clear format that a blog writer can easily use.
"""),
    ("human", """Topic: {topic}
Audience: {audience}
{revision_hints}

Write the research outline now."""),
])


def researcher_agent(
    llm: ChatGroq,
    topic: str,
    audience: str,
    feedback: str = "",
) -> str:
    """Create research notes for a topic."""
    chain = researcher_prompt | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints": build_revision_hints(feedback, "research"),
    })
    return str(result.content)


# ============================================================================
# 2. WRITER AGENT
# ============================================================================
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional blog writer.

Write a clear, engaging, and well-structured blog post using ONLY the research notes provided.

- Start with a catchy title and a short introduction that hooks the reader.
- Use clear headings and subheadings to organize the content.
- Write in simple language that suits the target audience.
- Use the facts, examples, and statistics from the research notes. Do not invent new facts.
- Keep paragraphs short and easy to read.
- Mention the sources from the research notes where relevant.
- End with a short conclusion and a takeaway or call to action.
- If revision hints are given, apply them carefully while rewriting.
"""),
    ("human", """Topic: {topic}
Audience: {audience}

Research notes:
{research}

Revision hints: {revision_hints}

Write the blog post now."""),
])


def writer_agent(
    llm: ChatGroq,
    topic: str,
    audience: str,
    research: str,
    feedback: str = "",
) -> str:
    """Write a blog draft from the research notes."""
    chain = writer_prompt | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "research": research,
        "revision_hints": build_revision_hints(feedback, "draft"),
    })
    return str(result.content)


# ============================================================================
# 3. EDITOR AGENT
# ============================================================================
editor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional blog editor.

Review the blog draft and return an improved, publication-ready version.

- Fix grammar, spelling, punctuation, and awkward sentences.
- Improve clarity, flow, and readability.
- Keep the title, headings, and structure clear and consistent.
- Do not add new facts. Keep the original meaning, tone, and sources.
- Shorten repetitive or unnecessary parts.
- Make it sound like a real person wrote it, not an AI.
- Use simple, everyday words. Avoid complex or fancy words.
- Return ONLY the final edited blog post, with no comments or explanations.
"""),
    ("human", """Topic: {topic}

Blog draft:
{draft}

Edit the blog post and return the final version now."""),
])


def editor_agent(
    llm: ChatGroq,
    topic: str,
    draft: str,
) -> str:
    """Polish the draft into the final blog post."""
    chain = editor_prompt | llm

    result = chain.invoke({
        "topic": topic,
        "draft": draft,
    })
    return str(result.content)


# ----------------------------------------------------------------------------
# Quick test (runs only when you execute this file directly)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    topic = "Benefits of AI in healthcare"
    audience = "Beginners"

    research = researcher_agent(get_llm(temperature=0.2), topic, audience)
    draft = writer_agent(get_llm(temperature=0.6), topic, audience, research)
    final_blog = editor_agent(get_llm(temperature=0.3), topic, draft)

    print(final_blog)
