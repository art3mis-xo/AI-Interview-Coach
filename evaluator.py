# chains/evaluator.py
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

class AnswerFeedback(BaseModel):
    """Structured feedback for a single answer."""

    score: int = Field(
        description="Score from 1-10 (1=poor, 10=excellent)",
        ge=1,
        le=10
    )
    understanding: str = Field(
        description="Assessment of conceptual understanding"
    )
    communication: str = Field(
        description="How well they explained their answer"
    )
    strengths: List[str] = Field(
        description="Specific things the candidate did well"
    )
    improvements: List[str] = Field(
        description="Specific areas to improve"
    )
    follow_up_question: Optional[str] = Field(
        description="A follow-up question to probe deeper",
        default=None
    )

class InterviewReport(BaseModel):
    """Final interview evaluation report."""

    overall_score: int = Field(ge=1, le=10)
    recommendation: str = Field(
        description="hire / maybe / no_hire"
    )
    summary: str = Field(
        description="2-3 sentence overall assessment"
    )
    technical_skills: int = Field(ge=1, le=10)
    communication_skills: int = Field(ge=1, le=10)
    problem_solving: int = Field(ge=1, le=10)
    strengths: List[str]
    areas_to_improve: List[str]
    suggested_topics_to_study: List[str]

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def create_evaluator_chain():
    """Create a chain that returns structured feedback."""

    # Create parser from our Pydantic model
    parser = PydanticOutputParser(pydantic_object=AnswerFeedback)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interview evaluator.

Evaluate the candidate's answer to the given question.
Be specific and constructive in your feedback.

Question context: {question}
Position level: {level}

{format_instructions}
"""),
        ("human", "Candidate's answer: {answer}")
    ])

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    chain = prompt | llm | parser

    return chain, parser

def create_evaluator_simple():
    """Use with_structured_output for cleaner code."""

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    # Automatically handles the output parsing
    structured_llm = llm.with_structured_output(AnswerFeedback)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interview evaluator.
Evaluate the candidate's answer to the given question.
Be specific and constructive in your feedback.

Question: {question}
Position level: {level}"""),
        ("human", "{answer}")
    ])

    chain = prompt | structured_llm

    return chain

def create_report_generator():
    """Generate final interview report."""

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    structured_llm = llm.with_structured_output(InterviewReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are generating a final interview evaluation report.

Based on the interview transcript below, provide a comprehensive assessment.
Be fair, specific, and constructive.

Position: {position}
Level: {level}
Interview type: {interview_type}
"""),
        ("human", """Interview transcript:
{transcript}

Individual question scores: {scores}

Generate the final report.""")
    ])

    return prompt | structured_llm


