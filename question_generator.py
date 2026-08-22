# chains/question_generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_question_generator(retriever):
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical interviewer.

Generate ONE clear interview question based on the job requirements below.
The style should vary: sometimes conceptual ("Explain X"), sometimes scenario ("Imagine Y"),
sometimes troubleshooting ("How would you fix Z"), sometimes design ("How would you build...").
Always phrase it naturally as an interviewer would ask.
Do not include rubrics, breakdowns, or multiple levels.
Output only the question text.

Job Requirements Context:
{context}
"""),
    ("human", """Write a {difficulty} level interview question about {topic}.
It must be different from previous questions: {previous_questions}.""")
])

#     """Create a RAG chain for generating interview questions."""

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are an expert technical interviewer.

# Generate ONE clear, scenario-based interview question based on the job requirements below.
# Do not include rubrics, breakdowns, or multiple levels.
# Output only the question text.

# Job Requirements Context:
# {context}
# """),
#         ("human", """Write a {difficulty} level interview question about {topic}.
# It must be different from previous questions: {previous_questions}.""")
#     ])

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["topic"])),
            "difficulty": lambda x: x["difficulty"],
            "topic": lambda x: x["topic"],
            "previous_questions": lambda x: x.get("previous_questions", "None")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain