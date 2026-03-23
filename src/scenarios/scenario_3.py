from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


class ContractReviewScenario(BaseScenario):
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [EvalQuestionCategory.AUTHORITY_HIERARCHY]

    def get_authority_context(self) -> str:
        return (
            "Cross-functional relationship with rotating domain authority. "
            "All four experts have MEDIUM authority overall, but each has domain authority "
            "in their area: Legal Counsel owns legal risk assessment, Finance Lead owns "
            "financial terms and cost analysis, Engineering Manager owns technical feasibility "
            "and SLA assessment, Procurement Lead owns vendor benchmarks and negotiability. "
            "When experts disagree across domains, each expert's assessment prevails "
            "for their domain. When they disagree within a domain (rare), the domain expert wins."
        )
