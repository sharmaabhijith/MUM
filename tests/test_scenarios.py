import pytest

from datagen.models.enums import EvalQuestionCategory, RelationshipType
from datagen.scenarios import load_scenario, create_scenario


class TestScenarioLoading:
    def test_load_scenario_1(self):
        config = load_scenario("1")
        assert config.scenario_id == "1"
        assert config.relationship_type == RelationshipType.SYMMETRIC
        assert len(config.users) == 4
        assert config.sessions_per_user == 7
        assert config.get_turns_for_session(1) == 50
        assert config.get_turns_for_session(2) == 50
        assert config.get_turns_for_session(3) == 20

    def test_load_scenario_2(self):
        config = load_scenario("2")
        assert config.scenario_id == "2"
        assert config.relationship_type == RelationshipType.HIERARCHICAL
        assert len(config.users) == 4
        assert config.get_turns_for_session(1) == 50

    def test_load_scenario_3(self):
        config = load_scenario("3")
        assert config.scenario_id == "3"
        assert config.relationship_type == RelationshipType.CROSS_FUNCTIONAL
        assert len(config.users) == 4

    def test_load_scenario_4(self):
        config = load_scenario("4")
        assert config.scenario_id == "4"
        assert config.relationship_type == RelationshipType.ADVERSARIAL
        assert len(config.users) == 4
        # Check sides
        buyer_users = [u for u in config.users if u.side == "buyer"]
        seller_users = [u for u in config.users if u.side == "seller"]
        assert len(buyer_users) == 2
        assert len(seller_users) == 2

    def test_load_scenario_5(self):
        config = load_scenario("5")
        assert config.scenario_id == "5"
        assert config.relationship_type == RelationshipType.SEQUENTIAL
        assert len(config.users) == 4
        # Check sequence order
        orders = [u.sequence_order for u in config.users]
        assert sorted(orders) == [1, 2, 3, 4]


class TestScenarioClasses:
    def test_study_group_eval_categories(self):
        config = load_scenario("1")
        scenario = create_scenario(config)
        categories = scenario.get_applicable_eval_categories()
        assert EvalQuestionCategory.USER_ATTRIBUTION in categories
        assert EvalQuestionCategory.TEMPORAL_CORRECTION in categories
        assert EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION not in categories
        assert EvalQuestionCategory.SEQUENTIAL_HANDOFF not in categories

    def test_negotiation_eval_categories(self):
        config = load_scenario("4")
        scenario = create_scenario(config)
        categories = scenario.get_applicable_eval_categories()
        assert EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION in categories
        assert EvalQuestionCategory.AUTHORITY_HIERARCHY in categories
        assert EvalQuestionCategory.SEQUENTIAL_HANDOFF not in categories

    def test_support_eval_categories(self):
        config = load_scenario("5")
        scenario = create_scenario(config)
        categories = scenario.get_applicable_eval_categories()
        assert EvalQuestionCategory.SEQUENTIAL_HANDOFF in categories
        assert EvalQuestionCategory.TEMPORAL_CORRECTION in categories
        assert EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION not in categories

    def test_conflict_filtering(self):
        config = load_scenario("1")
        scenario = create_scenario(config)
        # C1-1 targets sessions [3, 4, 5] for student_b
        conflicts_s1 = scenario.get_conflicts_for_session("student_b", 1)
        conflicts_s3 = scenario.get_conflicts_for_session("student_b", 3)
        assert len(conflicts_s1) == 0
        assert len(conflicts_s3) >= 1

    def test_authority_context(self):
        config = load_scenario("2")
        scenario = create_scenario(config)
        context = scenario.get_authority_context()
        assert "Commissioner" in context
        assert "hierarchical" in context.lower() or "Hierarchical" in context


class TestUserProfiles:
    def test_user_has_all_roleplay_fields(self):
        config = load_scenario("1")
        for user in config.users:
            assert user.communication_style, f"{user.user_id} missing communication_style"
            assert user.document_reading_pattern, f"{user.user_id} missing document_reading_pattern"
            assert user.reaction_to_corrections, f"{user.user_id} missing reaction_to_corrections"
            assert user.emotional_tendencies, f"{user.user_id} missing emotional_tendencies"
            assert user.reference_style, f"{user.user_id} missing reference_style"
            assert len(user.session_evolution) >= 7, f"{user.user_id} missing session_evolution entries"
            assert len(user.example_utterances) >= 3, f"{user.user_id} needs more example_utterances"

    def test_annotation_targets(self):
        config = load_scenario("1")
        targets = config.annotation_targets
        assert targets.eval_questions == 175
        assert sum(targets.eval_breakdown.values()) == 175
