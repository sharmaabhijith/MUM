from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


class ExecTeamScenario(BaseScenario):
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [
            EvalQuestionCategory.AUTHORITY_HIERARCHY,
            EvalQuestionCategory.TEMPORAL_CORRECTION,
        ]

    def get_authority_context(self) -> str:
        return (
            "Hierarchical relationship: VP of Operations (HIGH authority) > "
            "Senior Analyst (MEDIUM) = Marketing Lead (MEDIUM) > Junior Associate (LOW). "
            "The VP has authority over narrative tone and board messaging. "
            "However, factual corrections override authority — if the Junior Associate "
            "finds a numerical error, the correct number prevails regardless of who found it. "
            "Authority wins on opinions and framing; facts are flat."
        )
