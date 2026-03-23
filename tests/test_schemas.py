import json

import pytest

from src.models.enums import (
    AuthorityLevel,
    ConflictResolution,
    ConflictType,
    EvalQuestionCategory,
    MemoryStatus,
    MemoryType,
    RelationshipType,
)
from src.models.schemas import (
    AnnotationTargets,
    BenchmarkDataset,
    ConflictAnnotation,
    ConversationSession,
    ConversationTurn,
    DocumentConfig,
    EvalQuestion,
    EvidenceLink,
    ExtractedMemory,
    GenerationReport,
    InjectedConflict,
    ScenarioConfig,
    ScenarioOutput,
    ScenarioTimeline,
    SessionSummary,
    UserProfile,
    ValidationReport,
)


def make_user_profile(**overrides) -> UserProfile:
    defaults = {
        "user_id": "student_a",
        "display_name": "Student A",
        "scenario_id": "1",
        "authority_level": AuthorityLevel.EQUAL,
        "authority_weight": 1.0,
        "expertise": "Theory",
        "focus_areas": ["clocks", "proofs"],
        "biases": ["prefers math"],
        "prompt_behavior_notes": "Asks precise questions.",
        "communication_style": "Formal and precise.",
        "document_reading_pattern": "Deep reader of theory.",
        "reaction_to_corrections": "Accepts gracefully.",
        "knowledge_gaps": ["weak on implementation"],
        "misconceptions": ["vector clocks are 30% of exam"],
        "emotional_tendencies": "Confident but anxious.",
        "reference_style": "Formal citations.",
        "session_evolution": {"session_1": "Tentative.", "session_2": "Confident."},
        "example_utterances": ["Can you walk me through the proof?"],
    }
    defaults.update(overrides)
    return UserProfile(**defaults)


class TestEnums:
    def test_relationship_types(self):
        assert len(RelationshipType) == 5
        assert RelationshipType.SYMMETRIC.value == "symmetric"
        assert RelationshipType.ADVERSARIAL.value == "adversarial"

    def test_authority_levels(self):
        assert len(AuthorityLevel) == 4
        assert AuthorityLevel.HIGH.value == "high"

    def test_conflict_types(self):
        assert len(ConflictType) == 4
        assert ConflictType.FACTUAL.value == "factual"

    def test_conflict_resolution(self):
        assert len(ConflictResolution) == 4
        assert ConflictResolution.TEMPORAL_SUPERSESSION.value == "temporal_supersession"

    def test_memory_types(self):
        assert len(MemoryType) == 9
        assert MemoryType.CUSTOMER_COMM.value == "customer_communication"

    def test_memory_status(self):
        assert len(MemoryStatus) == 3
        assert MemoryStatus.SUPERSEDED.value == "superseded"

    def test_eval_question_categories(self):
        assert len(EvalQuestionCategory) == 12
        assert EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION.value == "adversarial_side_isolation"
        assert EvalQuestionCategory.SEQUENTIAL_HANDOFF.value == "sequential_handoff"


class TestUserProfile:
    def test_create_with_all_fields(self):
        user = make_user_profile()
        assert user.user_id == "student_a"
        assert user.authority_level == "equal"
        assert len(user.session_evolution) == 2
        assert len(user.example_utterances) == 1

    def test_optional_fields(self):
        user = make_user_profile()
        assert user.domain_authority is None
        assert user.side is None
        assert user.sequence_order is None

    def test_side_field(self):
        user = make_user_profile(side="buyer", user_id="buyer_cfo")
        assert user.side == "buyer"


class TestConversationModels:
    def test_conversation_turn(self):
        turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="Hello",
            timestamp="2025-04-01T09:00:00Z",
        )
        assert turn.role == "user"

    def test_conversation_session(self):
        session = ConversationSession(
            session_id="s1_student_a_1",
            scenario_id="1",
            user_id="student_a",
            session_number=1,
            session_timestamp="2025-04-01T09:00:00Z",
            turns=[
                ConversationTurn(
                    turn_number=1, role="user", content="Hi", timestamp="2025-04-01T09:00:00Z"
                ),
                ConversationTurn(
                    turn_number=1,
                    role="assistant",
                    content="Hello!",
                    timestamp="2025-04-01T09:01:00Z",
                ),
            ],
        )
        assert len(session.turns) == 2
        assert session.target_conflicts == []


class TestAnnotationModels:
    def test_session_summary(self):
        summary = SessionSummary(
            session_id="s1_student_a_1",
            scenario_id="1",
            user_id="student_a",
            session_number=1,
            summary="Student A explored vector clocks.",
            key_facts=["Vector clocks provide partial ordering"],
            positions_taken=["Vector clocks are heavily tested"],
        )
        assert summary.positions_changed == []

    def test_extracted_memory(self):
        mem = ExtractedMemory(
            memory_id="m1",
            scenario_id="1",
            user_id="student_a",
            session_id="s1_student_a_1",
            memory_type=MemoryType.OPINION,
            content="Vector clocks are 30% of the exam.",
            status=MemoryStatus.ACTIVE,
            authority_level=AuthorityLevel.EQUAL,
            timestamp="2025-04-01T09:00:00Z",
        )
        assert mem.superseded_by is None

    def test_memory_supersession(self):
        mem_old = ExtractedMemory(
            memory_id="m1",
            scenario_id="1",
            user_id="student_a",
            session_id="s1_student_a_1",
            memory_type=MemoryType.OPINION,
            content="Vector clocks are 30% of the exam.",
            status=MemoryStatus.SUPERSEDED,
            superseded_by="m2",
            authority_level=AuthorityLevel.EQUAL,
            timestamp="2025-04-01T09:00:00Z",
        )
        assert mem_old.status == "superseded"
        assert mem_old.superseded_by == "m2"

    def test_evidence_link_session_level(self):
        link = EvidenceLink(user_id="student_a", session_id="s1_student_a_3")
        assert link.user_id == "student_a"
        assert link.session_id == "s1_student_a_3"
        # No turn_number field
        assert not hasattr(link, "turn_number")

    def test_conflict_annotation(self):
        conflict = ConflictAnnotation(
            conflict_id="C1-1",
            scenario_id="1",
            users_involved=["student_b", "student_d"],
            topic="Paxos vs Raft",
            conflict_type=ConflictType.INTERPRETIVE,
            positions={"student_b": "Raft is simpler", "student_d": "Paxos is more elegant"},
            resolution=ConflictResolution.PRESERVE_BOTH,
            resolution_detail="Both views preserved as peer opinions.",
            evidence=[
                EvidenceLink(user_id="student_b", session_id="s1_student_b_3"),
                EvidenceLink(user_id="student_d", session_id="s1_student_d_4"),
            ],
            first_surfaced="2025-04-03T14:00:00Z",
            last_updated="2025-04-08T16:00:00Z",
        )
        assert len(conflict.evidence) == 2


class TestEvalQuestion:
    def test_create(self):
        q = EvalQuestion(
            question_id="q1",
            scenario_id="1",
            category=EvalQuestionCategory.USER_ATTRIBUTION,
            question="Which student identified the notation inconsistency?",
            gold_answer="Student D",
            evidence=[EvidenceLink(user_id="student_d", session_id="s1_student_d_1")],
            required_memories=["m5"],
            difficulty="easy",
        )
        assert q.category == "user_attribution"
        assert q.difficulty == "easy"


class TestConfigModels:
    def test_scenario_config_roundtrip(self):
        config = ScenarioConfig(
            scenario_id="1",
            name="Study Group",
            relationship_type=RelationshipType.SYMMETRIC,
            domain="Education / CS",
            users=[make_user_profile()],
            documents=[
                DocumentConfig(name="Textbook", filename="textbook.pdf", target_tokens=8000)
            ],
            sessions_per_user=7,
            turns_per_session=18,
            timeline=ScenarioTimeline(
                start_date="2025-04-01",
                end_date="2025-04-14",
                session_schedule={"student_a": [{"session": 1, "date": "2025-04-01"}]},
            ),
            injected_conflicts=[
                InjectedConflict(
                    conflict_id="C1-1",
                    users=["student_b", "student_d"],
                    topic="Paxos vs Raft",
                    nature="B prefers Raft, D prefers Paxos",
                    resolution="Preserve both",
                    target_sessions=[3, 4, 5],
                )
            ],
            annotation_targets=AnnotationTargets(
                memories_per_session=3.5,
                conflicts=3,
                eval_questions=175,
                eval_breakdown={"user_attribution": 20},
            ),
        )
        # Roundtrip through JSON
        json_str = config.model_dump_json()
        restored = ScenarioConfig.model_validate_json(json_str)
        assert restored.scenario_id == "1"
        assert restored.relationship_type == "symmetric"
        assert len(restored.users) == 1
        assert restored.users[0].user_id == "student_a"


class TestBenchmarkDataset:
    def test_nested_structure(self):
        dataset = BenchmarkDataset(
            version="0.1.0",
            generated_at="2025-04-15T00:00:00Z",
            scenarios=[],
            generation_report=GenerationReport(),
        )
        assert dataset.version == "0.1.0"
        assert dataset.scenarios == []
