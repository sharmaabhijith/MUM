from __future__ import annotations

import json

from src.models.enums import EvalQuestionCategory
from src.models.schemas import (
    ConflictAnnotation,
    ConversationSession,
    ExtractedMemory,
)
from src.pipeline.phase1_document_prep import DocumentContext

# Category-specific instructions
CATEGORY_INSTRUCTIONS: dict[str, str] = {
    EvalQuestionCategory.USER_ATTRIBUTION: (
        'Generate questions of the form "Who [said/flagged/proposed/discovered] X?" '
        "that test whether the system correctly tracks which user stated specific information. "
        "Metric: exact-match accuracy on user attribution."
    ),
    EvalQuestionCategory.CROSS_USER_SYNTHESIS: (
        'Generate questions of the form "Combining all users\' analyses, what is the complete picture of X?" '
        "that test whether the system can merge complementary knowledge from different users. "
        "Metric: FactScore (atomic fact precision/recall) + LLM-as-judge completeness."
    ),
    EvalQuestionCategory.CONFLICT_RESOLUTION: (
        "Generate questions that test whether the system can identify disagreements between users "
        "and apply the correct resolution strategy for the relationship type. "
        "Include sub-tasks: conflict identification, characterization, and resolution. "
        "Metric: P/R/F1 on conflict identification + LLM-as-judge on resolution correctness."
    ),
    EvalQuestionCategory.INFORMATION_GAP: (
        'Generate questions of the form "What has [User A] discovered that [User B] needs but doesn\'t know?" '
        "that test cross-user information asymmetry detection. "
        "Metric: LLM-as-judge on whether gaps are genuine and actionable."
    ),
    EvalQuestionCategory.ROLE_APPROPRIATE_BRIEFING: (
        'Generate questions of the form "Generate a briefing for [User X] summarizing what others found, relevant to X\'s role." '
        "The briefing must be aware of who the recipient is, what they already know, and their authority level. "
        "For adversarial scenarios, test that briefings don't leak cross-side information. "
        "Metric: LLM-as-judge on relevance, role-appropriateness, and information leakage."
    ),
    EvalQuestionCategory.ADVERSARIAL_CONFUSION: (
        "Generate questions that deliberately swap users or positions to test misattribution resistance. "
        'The expected answer is ALWAYS "No" with a correction identifying the actual user. '
        'Format: "Did [wrong user] [action]?" Answer: "No — [correct user] did. [Wrong user] actually [their real position]." '
        "Metric: Binary accuracy."
    ),
    EvalQuestionCategory.DOCUMENT_COVERAGE: (
        'Generate questions of the form "Which sections of [document] were reviewed by which users? What was missed?" '
        "Test collective coverage tracking and blind spot identification. "
        "Metric: Precision/Recall of (user, document_section) pairs + Recall on blind spots."
    ),
    EvalQuestionCategory.CROSS_USER_PROVENANCE: (
        'Generate questions of the form "Trace the history of [fact/decision/error] across all users and sessions." '
        "Test whether the system can track how information was discovered, evolved, and propagated. "
        "Metric: LLM-as-judge on chain completeness and temporal accuracy."
    ),
    EvalQuestionCategory.AUTHORITY_HIERARCHY: (
        "Generate questions that test whether the system understands that different users carry "
        "different weight depending on the relationship type and topic. "
        "Must specify which users are involved and the correct authority relationship. "
        "Metric: LLM-as-judge on correct authority identification and application."
    ),
    EvalQuestionCategory.TEMPORAL_CORRECTION: (
        "Generate questions about the full lifecycle of corrections: original wrong claim, "
        "who made it, when corrected, by whom, whether accepted, and current state. "
        "Must reference the full correction chain with session timestamps. "
        "Metric: LLM-as-judge on chain completeness, temporal accuracy, current-state correctness."
    ),
    EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION: (
        "Generate questions testing information firewalls between adversarial sides. "
        "Three sub-types: (1) Side leakage detection — does a briefing contain other side's strategy? "
        "(2) Position preservation — are both sides' positions preserved without compromise? "
        "(3) Cross-side reasoning — can the system reason about both sides without merging? "
        "Metric: LLM-as-judge on information leakage + position accuracy per side."
    ),
    EvalQuestionCategory.SEQUENTIAL_HANDOFF: (
        "Generate questions testing handoff continuity across sequential agents. "
        "Three sub-types: (1) Handoff completeness — what information was available/lost? "
        "(2) Customer communication audit — what was the customer told at each stage? "
        "(3) Diagnosis chain accuracy — reconstruct the sequence of diagnoses. "
        "Metric: LLM-as-judge on handoff completeness + communication accuracy + temporal ordering."
    ),
}


def build_eval_question_prompt(
    scenario_id: str,
    category: EvalQuestionCategory,
    target_count: int,
    conversations_summary: str,
    memories: list[ExtractedMemory],
    conflicts: list[ConflictAnnotation],
    doc_context: DocumentContext,
    relationship_type: str,
    authority_context: str,
) -> str:
    memories_json = json.dumps(
        [m.model_dump() for m in memories], indent=2, ensure_ascii=False
    )
    conflicts_json = json.dumps(
        [c.model_dump() for c in conflicts], indent=2, ensure_ascii=False
    )

    category_instruction = CATEGORY_INSTRUCTIONS.get(
        category, "Generate evaluation questions for this category."
    )

    return f"""Generate exactly {target_count} evaluation questions for the category: {category.value}

## Category Description
{category_instruction}

## Scenario Context
Scenario ID: {scenario_id}
Relationship type: {relationship_type}
Authority context: {authority_context}

## Rules
1. Every question MUST have explicit evidence links: list of (user_id, session_id) pairs.
2. Gold answers must cite specific memory IDs and conflict IDs where relevant.
3. Assign difficulty: easy (single fact/user), medium (2-3 facts/users), hard (complex multi-user reasoning).
4. Questions must be answerable from the provided memories, conflicts, and conversation summaries.
5. Each question must require reasoning about multiple users — this is a MULTI-USER benchmark.

## Available Data

### Conversation Summaries
{conversations_summary[:4000]}

### Extracted Memories
{memories_json[:6000]}

### Detected Conflicts
{conflicts_json[:3000]}

## Output Format
Return a JSON object with a "questions" key:
{{"questions": [
  {{
    "question_id": "q_{scenario_id}_{category.value}_{{index}}",
    "scenario_id": "{scenario_id}",
    "category": "{category.value}",
    "question": "The evaluation question text",
    "gold_answer": "The expected correct answer",
    "evidence": [
      {{"user_id": "user_id", "session_id": "session_id"}}
    ],
    "required_memories": ["memory_id_1", "memory_id_2"],
    "required_conflicts": ["conflict_id_1"],
    "difficulty": "easy|medium|hard"
  }}
]}}

Generate exactly {target_count} questions."""
