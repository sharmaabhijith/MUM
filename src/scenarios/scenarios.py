"""
Scenario implementations for the MUM benchmark.

This module defines the five scenario classes that customize evaluation behavior
for each relationship type in the benchmark. Each scenario extends BaseScenario
(defined in base.py) with two responsibilities:

1. ``_get_extra_eval_categories()`` — Returns additional EvalQuestionCategory
   values beyond the 8 universal categories. These enable scenario-specific
   evaluation questions that test behaviors unique to the relationship type.

2. ``get_authority_context()`` — Returns a natural-language description of the
   authority structure, injected into conversation generation prompts so the
   LLM understands how to handle disagreements between users.

Architecture Overview
---------------------

::

    BaseScenario (base.py)
    ├── StudyGroupScenario          — Scenario 1: Ficlandia country research (symmetric)
    ├── MunicipalCommitteeScenario  — Scenario 2: FicTown waste management (hierarchical)
    ├── ProductStrategyScenario     — Scenario 3: Ficduct platform review (cross-functional)
    ├── NegotiationScenario         — Scenario 4: Project Ficguard M&A (adversarial)
    └── IncidentResponseScenario    — Scenario 5: Meridian payment failures (sequential)

The mapping from ``RelationshipType`` enum → scenario class is maintained in
``__init__.py`` via the ``SCENARIO_CLASSES`` dict, which ``create_scenario()``
uses to instantiate the right subclass based on the YAML config's
``relationship_type`` field.

Relationship Types
------------------
- **Symmetric**: All users are equal peers. Disagreements are resolved by
  document evidence alone; no user's opinion outranks another's.
- **Hierarchical**: Users have explicit authority levels (HIGH > MEDIUM > LOW).
  Authority wins on framing/opinion; facts override authority.
- **Cross-functional**: Users have equal overall authority but each owns a
  domain. Within-domain authority resolves domain-specific disagreements.
- **Adversarial**: Two opposing sides, each with internal hierarchy. Conflicts
  between sides are preserved as adversarial positions (never merged).
- **Sequential**: Users work in temporal sequence with expanding evidence.
  Later findings supersede earlier diagnoses when supported by new evidence.
"""

from __future__ import annotations

from src.models.enums import EvalQuestionCategory
from src.scenarios.base import BaseScenario


# ---------------------------------------------------------------------------
# Scenario 1 — Symmetric: Study Group
# ---------------------------------------------------------------------------


class StudyGroupScenario(BaseScenario):
    """Scenario 1 — Study Group: Country Research on Ficlandia.

    Relationship type: SYMMETRIC (all users are equal peers)
    Domain: Education / Area Studies
    Users: 4 graduate students researching the fictional Republic of Ficlandia
    Documents: 7 comprehensive reference documents covering Ficlandia's history,
        geography, demographics, economics, culture, civil society, and future

    Authority structure:
        All students have equal authority (weight 1.0). No user's opinion
        overrides another's. When students disagree, both views are preserved.
        Factual corrections are resolved by document evidence, not authority.

    Special evaluation categories:
        - TEMPORAL_CORRECTION: Students may hold misconceptions that get
          corrected across sessions as they read more deeply into the documents.

    What makes this scenario distinctive:
        - Ficlandia is entirely fictional — all facts must come from documents
        - Each student has different expertise (history vs economics vs culture
          vs geography)
        - Students have planted misconceptions about Ficlandia that may be
          corrected by the AI or by cross-referencing documents
        - No authority gradient means disagreements are purely evidence-based
    """

    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [EvalQuestionCategory.TEMPORAL_CORRECTION]

    def get_authority_context(self) -> str:
        return (
            "All users are equal peers (symmetric relationship). "
            "No user has authority over another. "
            "When users disagree, both views should be preserved — neither is "
            "'correct' by authority. "
            "Factual corrections are resolved by document evidence, not authority."
        )


# ---------------------------------------------------------------------------
# Scenario 2 — Hierarchical: Municipal Committee
# ---------------------------------------------------------------------------


class MunicipalCommitteeScenario(BaseScenario):
    """Scenario 2 — Municipal Committee: Solid Waste Management Reform in FicTown.

    Relationship type: HIERARCHICAL
    Domain: Municipal Governance / Public Policy
    Users: 4 municipal committee members with clear authority hierarchy
    Documents: 6 internal reports on FicTown's solid waste management crisis
        (ground audit, public health study, contractor review, citizen
        grievances, budget utilization, proposed technical upgrade)

    Authority structure:
        Commissioner (HIGH) > Health Officer (MEDIUM) = Public Works Engineer
        (MEDIUM) > Ward Councillor (LOW). The Commissioner has authority over
        policy direction and committee recommendations. However, factual
        corrections override authority — if the Ward Councillor finds a data
        error, the correct number prevails regardless of who found it.
        Authority wins on policy framing; facts are flat.

    Special evaluation categories:
        - AUTHORITY_HIERARCHY: Tests whether the system respects the municipal
          governance hierarchy when users disagree on policy framing.
        - TEMPORAL_CORRECTION: Committee members may revise positions as they
          cross-reference documents and discover systemic connections.

    What makes this scenario distinctive:
        - Data-heavy documents with specific INR figures, tonnages, percentages
        - Cross-references between documents reveal systemic failures
          (health ↔ landfill ↔ budget ↔ contractor)
        - The Proposed Technical Upgrade synthesizes evidence from all others
        - Tensions between cost constraints, public health urgency, and
          contractor accountability
    """

    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [
            EvalQuestionCategory.AUTHORITY_HIERARCHY,
            EvalQuestionCategory.TEMPORAL_CORRECTION,
        ]

    def get_authority_context(self) -> str:
        return (
            "Hierarchical relationship: Commissioner (HIGH authority) > "
            "Health Officer (MEDIUM) = Public Works Engineer (MEDIUM) > "
            "Ward Councillor (LOW). "
            "The Commissioner has authority over policy direction and "
            "committee recommendations. "
            "However, factual corrections override authority — if the Ward "
            "Councillor finds a data error in the Ground Audit or Budget "
            "Utilization report, the correct number prevails regardless of "
            "who found it. "
            "Authority wins on policy framing and prioritization; facts are flat."
        )


# ---------------------------------------------------------------------------
# Scenario 3 — Cross-Functional: Product Strategy
# ---------------------------------------------------------------------------


class ProductStrategyScenario(BaseScenario):
    """Scenario 3 — Product Strategy: Ficduct Enterprise Platform Review.

    Relationship type: CROSS_FUNCTIONAL
    Domain: Technology / Product Strategy
    Users: 4 cross-functional domain experts reviewing the Ficduct platform
    Documents: 7 documents covering Ficduct's product overview, technical
        architecture, marketing & sales, financial report, customer feedback,
        strategy & operations, and v2 planning

    Authority structure:
        Cross-functional with rotating domain authority. All four reviewers
        have MEDIUM authority overall, but each owns their domain:
        CTO/Engineering Lead → technical architecture and engineering roadmap,
        CFO/Finance Lead → financial analysis and unit economics,
        CMO/Marketing Lead → go-to-market strategy and competitive positioning,
        VP Product → product strategy and roadmap prioritization.
        Cross-domain disagreements: each expert's assessment prevails in their
        domain. Within-domain disagreements (rare): the domain expert wins.

    Special evaluation categories:
        - AUTHORITY_HIERARCHY: Tests whether the system correctly applies
          rotating domain authority rather than a fixed hierarchy.

    What makes this scenario distinctive:
        - No single expert can override another's domain assessment
        - Technical, financial, market, and customer perspectives may conflict
          on priorities
        - Documents cross-reference each other (customer feedback informs v2)
        - Mix of quantitative metrics and qualitative assessments
    """

    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [EvalQuestionCategory.AUTHORITY_HIERARCHY]

    def get_authority_context(self) -> str:
        return (
            "Cross-functional relationship with rotating domain authority. "
            "All four reviewers have MEDIUM authority overall, but each has "
            "domain authority in their area: CTO/Engineering Lead owns "
            "technical architecture and engineering roadmap assessment, "
            "CFO/Finance Lead owns financial analysis and unit economics, "
            "CMO/Marketing Lead owns go-to-market strategy and competitive "
            "positioning, VP Product owns product strategy and roadmap "
            "prioritization. "
            "When experts disagree across domains, each expert's assessment "
            "prevails for their domain. When they disagree within a domain "
            "(rare), the domain expert wins."
        )


# ---------------------------------------------------------------------------
# Scenario 4 — Adversarial: M&A Negotiation
# ---------------------------------------------------------------------------


class NegotiationScenario(BaseScenario):
    """Scenario 4 — M&A Due Diligence: Project Ficguard.

    Relationship type: ADVERSARIAL
    Domain: Finance / M&A
    Users: 4 participants in two adversarial teams
        Buyer (Ficguard Industries): CFO (HIGH) + Counsel (MEDIUM)
        Seller (Veloxra Labs): CEO (HIGH) + Counsel (MEDIUM)
    Documents: 8 deal materials (term sheet, acquisition agreement, valuation
        report, financial due diligence, legal memorandum, negotiation strategy
        memo, IP/technology assessment, regulatory compliance review)

    Authority structure:
        Adversarial dual hierarchies. Buyer side: CFO (HIGH on
        price/valuation) > Counsel (MEDIUM on legal terms). Seller side:
        CEO (HIGH on price/strategy) > Counsel (MEDIUM on legal terms).
        CRITICAL: No one on the buyer side outranks anyone on the seller
        side and vice versa. Conflicts between sides are preserved as
        adversarial positions — the conflict IS the data. The AI must never
        leak one side's strategy to the other.

    Special evaluation categories:
        - AUTHORITY_HIERARCHY: Tests within-side hierarchy.
        - ADVERSARIAL_SIDE_ISOLATION: Tests that buyer-side strategy is never
          disclosed to seller-side users and vice versa.

    What makes this scenario distinctive:
        - Both sides access the SAME documents but selectively cite favorable
          data
        - Internal strategy documents inform buyer-side analysis
        - The AI must help each side build THEIR case, not a neutral analysis
        - Legal, financial, technical, and regulatory perspectives all feed
          into deal positions
    """

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
            "CRITICAL: No one on the buyer side outranks anyone on the seller "
            "side and vice versa — they are adversarial peers across sides. "
            "Conflicts between sides are preserved as adversarial positions — "
            "the conflict IS the data, never merge or average opposing "
            "positions. "
            "Never leak one side's strategy to the other."
        )


# ---------------------------------------------------------------------------
# Scenario 5 — Sequential: Incident Response
# ---------------------------------------------------------------------------


class IncidentResponseScenario(BaseScenario):
    """Scenario 5 — Incident Response: Cascading Payment Failures at Meridian.

    Relationship type: SEQUENTIAL
    Domain: Technology / Incident Management
    Users: 4 analysts working sequentially across shifts (A → B → C → D)
        with expanding evidence corpus at each handoff
    Documents: 5 documents (main incident report + 4 log packets, one per
        shift)

    Authority structure:
        Sequential with temporal authority. All four analysts have EQUAL
        authority (the Main Incident Report specifies "four analysts of equal
        authority"). Later analysts' findings supersede earlier analysts'
        diagnoses ONLY when supported by new evidence. The full audit trail
        must be preserved — including incorrect diagnoses and what was
        assessed at each stage.

        Diagnostic evolution across shifts:
        - Shift A (Analyst 1): DB pool exhaustion diagnosis based on
          PgBouncer saturation and connection-wait timeouts
        - Shift B (Analyst 2): DB-pool theory weakened — FSS (no DB
          dependency) symptomatic; TLS handshake failures emerge
        - Shift C (Analyst 3): ROOT CAUSE identified — CLS SAN-mismatch in
          mTLS cert rotation; 119/312 certs (38.1%) deployed with wrong SAN
        - Shift D (Analyst 4): CONFIRMED — PR #4781 race condition in CLS
          caused wrong SAN template; all 119 certs replaced; incident closed

    Special evaluation categories:
        - TEMPORAL_CORRECTION: Earlier diagnoses must be updated as new
          evidence arrives across shifts.
        - SEQUENTIAL_HANDOFF: Tests whether the system correctly tracks the
          evolving diagnosis across shift handoffs and preserves the full
          audit trail.

    What makes this scenario distinctive:
        - Evidence accumulates: each analyst reads all prior packets plus
          their new one
        - Red flags tracked across shifts (A-RF1 through C-RF2, all resolved
          by Shift D)
        - Shift A's diagnosis was reasonable given available evidence but
          ultimately incorrect
        - The distinction between valid data vs. invalid interpretation is
          critical for temporal reasoning
    """

    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        return [
            EvalQuestionCategory.TEMPORAL_CORRECTION,
            EvalQuestionCategory.SEQUENTIAL_HANDOFF,
        ]

    def get_authority_context(self) -> str:
        return (
            "Sequential relationship with temporal authority. "
            "Analyst 1 (Shift A, EQUAL authority) → Analyst 2 (Shift B, "
            "EQUAL authority) → Analyst 3 (Shift C, EQUAL authority) → "
            "Analyst 4 (Shift D, EQUAL authority). "
            "All four analysts have equal authority. Later analysts' findings "
            "supersede earlier analysts' diagnoses when supported by new "
            "evidence. "
            "Shift A's DB-pool-exhaustion diagnosis is superseded by Shift B's "
            "TLS-anomaly finding, which is refined by Shift C's CLS "
            "SAN-mismatch root cause identification, and confirmed by "
            "Shift D's PR #4781 race-condition analysis. "
            "However, the full audit trail must be preserved — including the "
            "incorrect diagnoses and what was assessed at each shift stage."
        )
