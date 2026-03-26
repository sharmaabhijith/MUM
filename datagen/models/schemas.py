from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from datagen.models.enums import (
    AuthorityLevel,
    EvalQuestionCategory,
    RelationshipType,
)


class UserProfile(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    user_id: str
    display_name: str
    scenario_id: str
    authority_level: AuthorityLevel
    authority_weight: float
    expertise: str
    focus_areas: list[str]
    biases: list[str]
    prompt_behavior_notes: str
    domain_authority: str | None = None
    side: str | None = None
    sequence_order: int | None = None

    # Detailed roleplay fields
    communication_style: str
    document_reading_pattern: str
    reaction_to_corrections: str
    knowledge_gaps: list[str]
    misconceptions: list[str]
    emotional_tendencies: str
    reference_style: str
    session_evolution: dict[str, str]
    example_utterances: list[str]


class ConversationTurn(BaseModel):
    turn_number: int
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class ConversationSession(BaseModel):
    session_id: str
    scenario_id: str
    user_id: str
    session_number: int
    session_timestamp: str
    turns: list[ConversationTurn]
    target_conflicts: list[str] = []
    metadata: dict[str, Any] = {}


class SessionSummary(BaseModel):
    """Per-session summary — mid-grain annotation."""

    session_id: str
    scenario_id: str
    user_id: str
    session_number: int
    summary: str
    key_facts: list[str]
    positions_taken: list[str]
    positions_changed: list[str] = []


class EvidenceLink(BaseModel):
    """Session-level evidence reference."""

    user_id: str
    session_id: str


class EvalQuestion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    question_id: str
    scenario_id: str
    category: EvalQuestionCategory
    question: str
    gold_answer: str
    evidence: list[EvidenceLink]
    difficulty: Literal["easy", "medium", "hard"]


class ScenarioTimeline(BaseModel):
    start_date: str
    end_date: str
    description: str = ""
    session_schedule: dict[str, list[dict[str, Any]]]


class DocumentConfig(BaseModel):
    name: str
    filename: str
    target_tokens: int
    content_requirements: str = ""


class InjectedConflict(BaseModel):
    conflict_id: str
    users: list[str]
    topic: str
    nature: str
    resolution: str
    target_sessions: list[int]


class AnnotationTargets(BaseModel):
    eval_questions: int
    eval_breakdown: dict[str, int]


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    scenario_id: str
    name: str
    relationship_type: RelationshipType
    domain: str
    users: list[UserProfile]
    documents: list[DocumentConfig]
    sessions_per_user: int
    turns_per_session: int | dict[str, int]
    timeline: ScenarioTimeline
    injected_conflicts: list[InjectedConflict]
    annotation_targets: AnnotationTargets
    session_arc: dict[str, str] = {}

    def get_turns_for_session(self, session_number: int) -> int:
        """Return the turn count for a specific session.

        If turns_per_session is an int, every session gets that count.
        If it's a dict, keys are session numbers (as strings) with int values.
        Sessions not listed in the dict fall back to the "default" key,
        or 20 if no default is specified.
        """
        if isinstance(self.turns_per_session, int):
            return self.turns_per_session
        key = str(session_number)
        if key in self.turns_per_session:
            return self.turns_per_session[key]
        return self.turns_per_session.get("default", 20)


class GenerationReport(BaseModel):
    total_cost: float = 0.0
    total_tokens: dict[str, int] = {}
    timing: dict[str, float] = {}
    per_phase_breakdown: dict[str, Any] = {}


class ScenarioOutput(BaseModel):
    scenario_id: str
    config: ScenarioConfig
    conversations: list[ConversationSession]
    session_summaries: list[SessionSummary]
    eval_questions: list[EvalQuestion]


class BenchmarkDataset(BaseModel):
    version: str
    generated_at: str
    scenarios: list[ScenarioOutput]
    aggregate_stats: dict[str, Any] = {}
    generation_report: GenerationReport
