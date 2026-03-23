import pytest

from src.llm.client import MockLLMClient
from src.llm.cost_tracker import CostTracker
from src.llm.token_counter import TokenCounter
from src.models.schemas import ConversationSession, ConversationTurn
from src.pipeline.phase1_document_prep import DocumentContext, DocumentPreparer
from src.pipeline.phase2_conversation import ConversationGenerator


class TestDocumentPreparer:
    def test_build_context_block(self):
        preparer = DocumentPreparer()
        documents = {"Doc A": "Content of doc A.", "Doc B": "Content of doc B."}
        block = preparer._build_context_block(documents)
        assert "=== DOCUMENT: Doc A ===" in block
        assert "Content of doc A." in block
        assert "=== END DOCUMENT ===" in block


class TestConversationGenerator:
    def test_validate_session_correct(self):
        cost_tracker = CostTracker()
        client = MockLLMClient(cost_tracker=cost_tracker)
        generator = ConversationGenerator(llm_client=client)

        session = ConversationSession(
            session_id="test_s1",
            scenario_id="1",
            user_id="user_a",
            session_number=1,
            session_timestamp="2025-04-01T09:00:00Z",
            turns=[
                ConversationTurn(turn_number=1, role="user", content="Hello", timestamp="T1"),
                ConversationTurn(turn_number=1, role="assistant", content="Hi", timestamp="T2"),
                ConversationTurn(turn_number=2, role="user", content="Question", timestamp="T3"),
                ConversationTurn(turn_number=2, role="assistant", content="Answer", timestamp="T4"),
            ],
        )
        warnings = generator._validate_session(session, expected_turns=2)
        assert len(warnings) == 0

    def test_validate_session_wrong_count(self):
        cost_tracker = CostTracker()
        client = MockLLMClient(cost_tracker=cost_tracker)
        generator = ConversationGenerator(llm_client=client)

        session = ConversationSession(
            session_id="test_s1",
            scenario_id="1",
            user_id="user_a",
            session_number=1,
            session_timestamp="2025-04-01T09:00:00Z",
            turns=[
                ConversationTurn(turn_number=1, role="user", content="Hello", timestamp="T1"),
            ],
        )
        warnings = generator._validate_session(session, expected_turns=2)
        assert len(warnings) > 0
        assert "Expected" in warnings[0]

    def test_parse_conversation_response(self):
        cost_tracker = CostTracker()
        client = MockLLMClient(cost_tracker=cost_tracker)
        generator = ConversationGenerator(llm_client=client)

        response = {
            "turns": [
                {"turn": 1, "role": "user", "content": "Hello", "timestamp": "2025-04-01T09:00:00Z"},
                {"turn": 1, "role": "assistant", "content": "Hi!", "timestamp": "2025-04-01T09:02:00Z"},
            ]
        }

        session = generator._parse_conversation_response(
            response=response,
            user_id="test_user",
            scenario_id="1",
            session_number=1,
            session_timestamp="2025-04-01T09:00:00Z",
            target_conflicts=["C1-1"],
            authority_level="equal",
        )

        assert session.session_id == "s1_test_user_1"
        assert len(session.turns) == 2
        assert session.turns[0].role == "user"
        assert session.turns[1].role == "assistant"


class TestCostTracker:
    def test_record_and_total(self):
        tracker = CostTracker()
        tracker.record_call("gpt-4o-mini", 1000, 500, "test")
        cost = tracker.get_total_cost()
        # 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003 = 0.00045
        assert abs(cost - 0.00045) < 0.0001

    def test_phase_breakdown(self):
        tracker = CostTracker()
        tracker.record_call("gpt-4o-mini", 1000, 500, "phase_a")
        tracker.record_call("gpt-4o-mini", 2000, 1000, "phase_b")
        phases = tracker.get_phase_breakdown()
        assert "phase_a" in phases
        assert "phase_b" in phases
        assert phases["phase_a"]["calls"] == 1
        assert phases["phase_b"]["calls"] == 1

    def test_summary(self):
        tracker = CostTracker()
        tracker.record_call("gpt-4o-mini", 1000, 500, "test")
        summary = tracker.summary()
        assert "Total cost" in summary
        assert "Total tokens" in summary
