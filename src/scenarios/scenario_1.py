from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


class StudyGroupScenario(BaseScenario):
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [EvalQuestionCategory.TEMPORAL_CORRECTION]

    def get_authority_context(self) -> str:
        return (
            "All users are equal peers (symmetric relationship). "
            "No user has authority over another. "
            "When users disagree, both views should be preserved — neither is 'correct' by authority. "
            "Factual corrections (e.g., BFT threshold) are resolved by evidence, not authority."
        )
