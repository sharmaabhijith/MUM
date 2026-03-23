from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


class NegotiationScenario(BaseScenario):
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [
            EvalQuestionCategory.AUTHORITY_HIERARCHY,
            EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION,
        ]

    def get_authority_context(self) -> str:
        return (
            "Adversarial relationship with dual hierarchies. "
            "Buyer side: Buyer's CFO (HIGH authority on price/valuation) > "
            "Buyer's Counsel (MEDIUM authority on legal terms). "
            "Seller side: Seller's CEO (HIGH authority on price/strategy) > "
            "Seller's Counsel (MEDIUM authority on legal terms). "
            "CRITICAL: No one on the buyer side outranks anyone on the seller side "
            "and vice versa — they are adversarial peers across sides. "
            "Conflicts between sides are preserved as adversarial positions — "
            "the conflict IS the data, never merge or average opposing positions. "
            "Never leak one side's strategy to the other."
        )
