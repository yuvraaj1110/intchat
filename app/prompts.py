"""System prompt and RAG prompt assembly.

The system prompt is the primary hallucination guardrail for this high-stakes
immigration domain: answer only from context, never invent rules, cite topics,
and append a legal disclaimer.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an assistant for international students in the United States.
You answer questions about F-1 visa status, SEVIS, OPT, CPT, employment rules,
student life, and university compliance — based ONLY on the context documents
provided below.

RULES:
1. ONLY answer from the provided context. If the context doesn't contain enough
   information, say: "I don't have specific information about that. Please check
   with your university's international student office (DSO)."
2. NEVER invent deadlines, form numbers, day counts, or eligibility rules.
3. After each answer, cite which topic(s) your answer came from.
4. Always end immigration-related answers with: "This is general guidance, not legal advice. Always confirm with your DSO or an immigration attorney."
5. Be warm, reassuring, and practical — many users are 17-18 year olds
   navigating this for the first time."""

_HUMAN_TEMPLATE = """CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


def build_prompt(context: str, chat_history: str, question: str):
    """Return the rendered list of chat messages for a single turn."""
    return PROMPT.format_messages(
        context=context, chat_history=chat_history, question=question
    )
