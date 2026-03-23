from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.models.enums import (
    AuthorityLevel,
    ConflictResolution,
    ConflictType,
    EvalQuestionCategory,
    MemoryStatus,
    MemoryType,
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


class ExtractedMemory(BaseModel):
    """Cross-session durable memory — session-level attribution."""

    model_config = ConfigDict(use_enum_values=True)

    memory_id: str
    scenario_id: str
    user_id: str
    session_id: str
    memory_type: MemoryType
    content: str
    status: MemoryStatus
    superseded_by: str | None = None
    authority_level: AuthorityLevel
    domain_tag: str | None = None
    side_tag: str | None = None
    timestamp: str


class EvidenceLink(BaseModel):
    """Session-level evidence reference."""

    user_id: str
    session_id: str


class ConflictAnnotation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    conflict_id: str
    scenario_id: str
    users_involved: list[str]
    topic: str
    conflict_type: ConflictType
    positions: dict[str, str]
    resolution: ConflictResolution
    resolution_detail: str
    evidence: list[EvidenceLink]
    first_surfaced: str
    last_updated: str


class EvalQuestion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    question_id: str
    scenario_id: str
    category: EvalQuestionCategory
    question: str
    gold_answer: str
    evidence: list[EvidenceLink]
    required_memories: list[str] = []
    required_conflicts: list[str] = []
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
    memories_per_session: float
    conflicts: int
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
    turns_per_session: int
    timeline: ScenarioTimeline
    injected_conflicts: list[InjectedConflict]
    annotation_targets: AnnotationTargets
    session_arc: dict[str, str] = {}


class ValidationReport(BaseModel):
    scenario_id: str
    conflict_coverage: dict[str, Any] = {}
    memory_extractability: dict[str, Any] = {}
    evidence_validity: dict[str, Any] = {}
    question_answerability: dict[str, Any] = {}
    persona_fidelity: dict[str, Any] = {}
    overall_pass: bool = False


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
    memories: list[ExtractedMemory]
    conflicts: list[ConflictAnnotation]
    eval_questions: list[EvalQuestion]
    validation_report: ValidationReport


class BenchmarkDataset(BaseModel):
    version: str
    generated_at: str
    scenarios: list[ScenarioOutput]
    aggregate_stats: dict[str, Any] = {}
    generation_report: GenerationReport
