#interviewer.py
from ast import Store
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

INTERVIEWER_SYSTEM_PROMPT = """You are an expert technical interviewer.

Your role:
- Ask one clear, focused question at a time
- Reference previous answers when relevant
- Build on the conversation naturally
- Be professional but encouraging

Interview type: {interview_type}
Position title: {position}
Position level: {level}
Focus area: {focus_area}

Use the exact role, level, and focus area provided above.
Do not switch to a different domain or job type.

Remember: You have access to the full conversation history.
Use it to avoid repeating questions and to ask follow-ups.
"""

def create_interviewer_chain_with_memory(memory):
    """Create interviewer chain with conversation memory."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEWER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

    # Chain that loads memory before processing
    chain = (
        RunnablePassthrough.assign(
            history=lambda x: memory.chat_memory.messages
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Store for multiple sessions
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def create_interviewer_with_history():
    """Create interviewer with automatic history management."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEWER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    # Wrap with history management
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain_with_history

