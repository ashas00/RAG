"""Pydantic data models exchanged between RAG pipeline stages."""
import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A span of characters inside one corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources plus a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset."""

    search_results: list[MinimalAnswer]
    k: int
