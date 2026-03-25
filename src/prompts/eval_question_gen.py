from __future__ import annotations

from src.models.enums import EvalQuestionCategory
from src.models.schemas import ConversationSession, SessionSummary
from src.pipeline.phase1_document_prep import DocumentContext

# Category-specific instructions
CATEGORY_INSTRUCTIONS: dict[str, str] = {
    EvalQuestionCategory.USER_ATTRIBUTION: (
        'Generate questions of the form "Who [said/flagged/proposed/discovered] X?" '
        "that test whether the system correctly tracks which user stated specific information. "
        "Ground each question in a concrete claim, opinion, or discovery that appears in "
        "a specific user's conversation turns. "
        "Metric: exact-match accuracy on user attribution."
    ),
    EvalQuestionCategory.CROSS_USER_SYNTHESIS: (
        'Generate questions of the form "Combining all users\' analyses, what is the complete picture of X?" '
        "that test whether the system can merge complementary knowledge from different users. "
        "Each question should require information from at least 2 different users' conversations. "
        "Metric: FactScore (atomic fact precision/recall) + LLM-as-judge completeness."
    ),
    EvalQuestionCategory.CONFLICT_RESOLUTION: (
        "Generate questions that test whether the system can identify disagreements between users "
        "and apply the correct resolution strategy for the relationship type. "
        "Look for places where users express contradictory views, cite conflicting data, or "
        "interpret the same evidence differently. "
        "Metric: P/R/F1 on conflict identification + LLM-as-judge on resolution correctness."
    ),
    EvalQuestionCategory.INFORMATION_GAP: (
        'Generate questions of the form "What has [User A] discovered that [User B] needs but doesn\'t know?" '
        "that test cross-user information asymmetry detection. "
        "Identify facts or insights from one user's sessions that are relevant to another user's "
        "focus area but never appear in that second user's conversations. "
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
        "Base these swaps on real claims from the conversations. "
        "Metric: Binary accuracy."
    ),
    EvalQuestionCategory.DOCUMENT_COVERAGE: (
        'Generate questions of the form "Which sections of [document] were reviewed by which users? What was missed?" '
        "Test collective coverage tracking and blind spot identification. "
        "Look at which document sections each user references in their conversations. "
        "Metric: Precision/Recall of (user, document_section) pairs + Recall on blind spots."
    ),
    EvalQuestionCategory.CROSS_USER_PROVENANCE: (
        'Generate questions of the form "Trace the history of [fact/decision/error] across all users and sessions." '
        "Test whether the system can track how information was discovered, evolved, and propagated "
        "across different users' conversation sessions. "
        "Metric: LLM-as-judge on chain completeness and temporal accuracy."
    ),
    EvalQuestionCategory.AUTHORITY_HIERARCHY: (
        "Generate questions that test whether the system understands that different users carry "
        "different weight depending on the relationship type and topic. "
        "Must specify which users are involved and the correct authority relationship. "
        "Look for disagreements where authority level should determine the resolution. "
        "Metric: LLM-as-judge on correct authority identification and application."
    ),
    EvalQuestionCategory.TEMPORAL_CORRECTION: (
        "Generate questions about the full lifecycle of corrections: original wrong claim, "
        "who made it, when corrected, by whom, whether accepted, and current state. "
        "Look for places where a user's position evolves across sessions as they encounter "
        "new evidence or corrections from the AI assistant. "
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


def _serialize_conversations(
    conversations: list[ConversationSession],
    max_tokens_estimate: int = 12000,
) -> str:
    """Serialize conversations to a compact text format for the prompt.

    Prioritizes including all users and sessions, truncating individual
    sessions if needed to stay within the token budget.
    """
    by_user: dict[str, list[ConversationSession]] = {}
    for c in conversations:
        by_user.setdefault(c.user_id, []).append(c)
    for sessions in by_user.values():
        sessions.sort(key=lambda s: s.session_number)

    # ~4 chars per token
    char_budget = max_tokens_estimate * 4
    n_sessions = len(conversations)
    per_session_budget = char_budget // max(n_sessions, 1)

    lines: list[str] = []
    for user_id in sorted(by_user):
        lines.append(f"=== User: {user_id} ===")
        for session in by_user[user_id]:
            lines.append(
                f"--- Session {session.session_number} "
                f"[{session.session_id}] ({session.session_timestamp}) ---"
            )
            session_chars = 0
            for turn in session.turns:
                turn_line = f"[{turn.role.upper()} T{turn.turn_number}] {turn.content}"
                if session_chars + len(turn_line) > per_session_budget:
                    lines.append("[... session truncated for prompt length ...]")
                    break
                lines.append(turn_line)
                session_chars += len(turn_line)
            lines.append("")

    return "\n".join(lines)


def _serialize_summaries(summaries: list[SessionSummary]) -> str:
    """Serialize session summaries into a compact text block."""
    lines: list[str] = []
    for s in sorted(summaries, key=lambda x: (x.user_id, x.session_number)):
        lines.append(f"[{s.user_id} — Session {s.session_number} ({s.session_id})]")
        lines.append(f"  Summary: {s.summary}")
        if s.key_facts:
            lines.append(f"  Key facts: {'; '.join(s.key_facts)}")
        if s.positions_taken:
            lines.append(f"  Positions: {'; '.join(s.positions_taken)}")
        if s.positions_changed:
            lines.append(f"  Changed: {'; '.join(s.positions_changed)}")
        lines.append("")
    return "\n".join(lines)


def build_eval_question_prompt(
    scenario_id: str,
    category: EvalQuestionCategory,
    target_count: int,
    conversations: list[ConversationSession],
    summaries: list[SessionSummary],
    doc_context: DocumentContext,
    relationship_type: str,
    authority_context: str,
    users_description: str,
) -> str:
    category_instruction = CATEGORY_INSTRUCTIONS.get(
        category, "Generate evaluation questions for this category."
    )

    conversations_text = _serialize_conversations(conversations)
    summaries_text = _serialize_summaries(summaries)

    return f"""Generate exactly {target_count} evaluation questions for the category: {category.value}

## Category Description
{category_instruction}

## Scenario Context
Scenario ID: {scenario_id}
Relationship type: {relationship_type}
Authority context: {authority_context}

## Users
{users_description}

## Rules
1. Every question MUST be grounded in specific conversation content — cite which user(s) and session(s) the answer draws from.
2. Provide evidence links as (user_id, session_id) pairs pointing to where the answer can be found.
3. Gold answers must cite specific content from the conversations.
4. Assign difficulty: easy (single fact/user), medium (2-3 facts/users), hard (complex multi-user reasoning).
5. Questions must require reasoning about multiple users — this is a MULTI-USER benchmark.
6. Do NOT generate questions that can be answered from general knowledge alone. Every question must require the conversation content.

## Session Summaries
{summaries_text}

## Full Conversations
{conversations_text}

## Output Format
Return a JSON object with a "questions" key:
{{"questions": [
  {{
    "question_id": "q_{scenario_id}_{category.value}_{{index}}",
    "scenario_id": "{scenario_id}",
    "category": "{category.value}",
    "question": "The evaluation question text",
    "gold_answer": "The expected correct answer, grounded in conversation content",
    "evidence": [
      {{"user_id": "user_id", "session_id": "session_id"}}
    ],
    "difficulty": "easy|medium|hard"
  }}
]}}

Generate exactly {target_count} questions."""
