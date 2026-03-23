from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


class SupportScenario(BaseScenario):
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [
            EvalQuestionCategory.TEMPORAL_CORRECTION,
            EvalQuestionCategory.SEQUENTIAL_HANDOFF,
        ]

    def get_authority_context(self) -> str:
        return (
            "Sequential relationship with temporal authority. "
            "Agent A (Tier 1, LOW authority) → Agent B (Tier 2, MEDIUM authority) → "
            "Agent C (Tier 2 Resolution, MEDIUM authority). "
            "Later agents' findings supersede earlier agents' diagnoses when supported "
            "by evidence. Agent B's connection pool diagnosis overrides Agent A's DNS diagnosis. "
            "However, the full audit trail must be preserved — including the incorrect "
            "diagnosis and what was communicated to the customer at each stage."
        )
