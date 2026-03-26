import pytest

from datagen.models.enums import AuthorityLevel, EvalQuestionCategory
from datagen.models.schemas import (
    ConversationSession,
    ConversationTurn,
    InjectedConflict,
    UserProfile,
)
from datagen.pipeline.phase1_document_prep import DocumentContext
from datagen.prompts.base import PromptBuilder
from datagen.prompts.conversation_system import build_conversation_system_prompt
from datagen.prompts.conversation_user import build_conversation_user_prompt
from datagen.prompts.eval_question_gen import CATEGORY_INSTRUCTIONS
from datagen.prompts.session_summary import build_summary_prompt


def make_user() -> UserProfile:
    return UserProfile(
        user_id="test_user",
        display_name="Test User",
        scenario_id="1",
        authority_level=AuthorityLevel.EQUAL,
        authority_weight=1.0,
        expertise="Testing",
        focus_areas=["unit tests", "integration"],
        biases=["prefers pytest"],
        prompt_behavior_notes="Asks testing questions.",
        communication_style="Direct and concise.",
        document_reading_pattern="Reads test files first.",
        reaction_to_corrections="Accepts readily.",
        knowledge_gaps=["weak on mocking"],
        misconceptions=["thinks coverage = quality"],
        emotional_tendencies="Calm and methodical.",
        reference_style="File paths and line numbers.",
        session_evolution={"session_1": "Initial exploration.", "session_2": "Deep dive."},
        example_utterances=["Can you show me how to test this?", "What's the coverage?"],
    )


def make_doc_context() -> DocumentContext:
    return DocumentContext(
        scenario_id="1",
        documents={"test_doc.pdf": "This is a test document with some content."},
        token_counts={"test_doc.pdf": 10},
        total_tokens=10,
        context_block="\n=== DOCUMENT: test_doc.pdf ===\nTest content.\n=== END ===\n",
    )


def make_session() -> ConversationSession:
    return ConversationSession(
        session_id="s1_test_user_1",
        scenario_id="1",
        user_id="test_user",
        session_number=1,
        session_timestamp="2025-04-01T09:00:00Z",
        turns=[
            ConversationTurn(
                turn_number=1, role="user", content="Hello", timestamp="2025-04-01T09:00:00Z"
            ),
            ConversationTurn(
                turn_number=1,
                role="assistant",
                content="Hi there!",
                timestamp="2025-04-01T09:01:00Z",
            ),
        ],
    )


class TestPromptBuilder:
    def test_build_user_persona_section(self):
        user = make_user()
        section = PromptBuilder.build_user_persona_section(user)
        assert "Test User" in section
        assert "Communication Style" in section
        assert "Document Reading Pattern" in section
        assert "Knowledge Gaps and Misconceptions" in section
        assert "Example Messages" in section
        assert "Can you show me how to test this?" in section

    def test_build_document_section(self):
        doc_context = make_doc_context()
        section = PromptBuilder.build_document_section(doc_context)
        assert "test_doc.pdf" in section
        assert "Test content." in section


class TestConversationPrompt:
    def test_no_unfilled_variables(self):
        """Ensure no {unfilled_variable} patterns remain."""

        class MockScenario:
            class config:
                scenario_id = "1"
                sessions_per_user = 7
                turns_per_session = 18
                timeline = type("T", (), {"start_date": "2025-04-01"})()

            def get_authority_context(self):
                return "All users are equal."

            def get_session_timestamp(self, user_id, session_number):
                return "2025-04-01T09:00:00Z"

        user = make_user()
        doc_context = make_doc_context()
        conflict = InjectedConflict(
            conflict_id="C1-1",
            users=["test_user", "other"],
            topic="Test topic",
            nature="Disagreement",
            resolution="Preserve both",
            target_sessions=[1],
        )

        prompt = build_conversation_system_prompt(
            user=user,
            scenario=MockScenario(),
            session_number=1,
            session_timestamp="2025-04-01T09:00:00Z",
            prior_summaries="",
            target_conflicts=[conflict],
            doc_context=doc_context,
        )

        # Check no unfilled template variables (simple braces with word chars)
        import re
        unfilled = re.findall(r"\{[a-z_]+\}", prompt)
        # Filter out JSON format examples
        json_patterns = {"{", "}", '{"turns"', '{"turn"'}
        unfilled = [u for u in unfilled if u not in json_patterns and "turn" not in u and "index" not in u]
        assert not unfilled, f"Unfilled variables found: {unfilled}"

    def test_user_prompt_contains_session_info(self):
        prompt = build_conversation_user_prompt(session_number=3, turns_per_session=18)
        assert "Session 3" in prompt
        assert "36 messages" in prompt
        assert "18 user messages" in prompt


class TestSummaryPrompt:
    def test_summary_prompt_structure(self):
        session = make_session()
        prompt = build_summary_prompt(session, "Prior summary text.")
        assert "s1_test_user_1" in prompt
        assert "Prior summary text." in prompt
        assert "150-250 word" in prompt


class TestEvalCategories:
    def test_all_categories_have_instructions(self):
        for category in EvalQuestionCategory:
            assert category in CATEGORY_INSTRUCTIONS, f"Missing instructions for {category}"
