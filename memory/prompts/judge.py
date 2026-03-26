"""Prompts for the LLM-as-judge evaluator."""

# Category-specific judging guidance so the judge knows what matters most
# for each evaluation dimension.
CATEGORY_JUDGE_GUIDANCE: dict[str, str] = {
    "user_attribution": (
        "This question tests whether the system correctly identifies WHICH USER "
        "said, claimed, or discovered something. Pay special attention to:\n"
        "- Whether the predicted answer names the correct user(s)\n"
        "- Whether it avoids confusing one user's statements with another's\n"
        "- Attribution is the most critical dimension for this category"
    ),
    "cross_user_synthesis": (
        "This question tests whether the system can merge complementary knowledge "
        "from multiple users into a coherent answer. Pay special attention to:\n"
        "- Whether information from all relevant users is included\n"
        "- Whether the synthesis is coherent, not just a list of per-user facts\n"
        "- Completeness is the most critical dimension for this category"
    ),
    "conflict_resolution": (
        "This question tests whether the system identifies disagreements between "
        "users and resolves them correctly. Pay special attention to:\n"
        "- Whether conflicting positions are identified and attributed to the right users\n"
        "- Whether the resolution is correct (using document evidence or authority hierarchy)\n"
        "- Correctness and attribution are both critical for this category"
    ),
    "information_gap": (
        "This question tests whether the system detects asymmetries — what one user "
        "knows that another user needs but doesn't have. Pay special attention to:\n"
        "- Whether the gap is accurately identified (User A knows X, User B doesn't)\n"
        "- Whether the information is genuinely relevant to the other user's needs\n"
        "- Correctness and completeness are both critical for this category"
    ),
    "role_appropriate_briefing": (
        "This question tests whether the system can generate a briefing tailored to "
        "a specific user's role, knowledge, and authority level. Pay special attention to:\n"
        "- Whether the briefing is relevant to the target user's role and expertise\n"
        "- Whether it avoids repeating what the user already knows\n"
        "- Whether it respects information boundaries (e.g., no cross-side leakage in adversarial scenarios)\n"
        "- Completeness and correctness are both critical for this category"
    ),
    "document_coverage": (
        "This question tests tracking of which users reviewed which document sections. "
        "Pay special attention to:\n"
        "- Whether user-to-document-section mappings are correct\n"
        "- Whether blind spots (unreviewd sections) are identified\n"
        "- Attribution and completeness are both critical for this category"
    ),
    "cross_user_provenance": (
        "This question tests tracing the full history of a fact or decision across "
        "users and sessions. Pay special attention to:\n"
        "- Whether the temporal chain is complete (who discovered → who propagated → who corrected)\n"
        "- Whether session ordering is accurate\n"
        "- Correctness and completeness are both critical for this category"
    ),
    "authority_hierarchy": (
        "This question tests whether the system understands that different users carry "
        "different authority weight. Pay special attention to:\n"
        "- Whether the correct authority relationship is identified\n"
        "- Whether disagreements are resolved in favor of the higher-authority user when appropriate\n"
        "- Correctness is the most critical dimension for this category"
    ),
    "temporal_correction": (
        "This question tests tracking the full lifecycle of a correction: original wrong "
        "claim, who made it, when corrected, whether accepted, and current state. "
        "Pay special attention to:\n"
        "- Whether the temporal chain is complete and in the right order\n"
        "- Whether the current (post-correction) state is correctly identified\n"
        "- Correctness and completeness are both critical for this category"
    ),
    "adversarial_side_isolation": (
        "This question tests information firewalls between adversarial sides. "
        "Pay special attention to:\n"
        "- Whether information from one side leaks into the other side's context\n"
        "- Whether both sides' positions are preserved without merging\n"
        "- Correctness and attribution are both critical for this category"
    ),
    "sequential_handoff": (
        "This question tests handoff continuity across sequential agents/shifts. "
        "Pay special attention to:\n"
        "- Whether information transfer across handoffs is accurately tracked\n"
        "- Whether any information loss during handoffs is identified\n"
        "- Completeness and correctness are both critical for this category"
    ),
}


def build_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    category: str,
) -> list[dict]:
    """Build the prompt for the LLM judge to score a predicted answer.

    The judge scores on multiple dimensions and returns structured JSON.
    """
    category_guidance = CATEGORY_JUDGE_GUIDANCE.get(category, "")
    category_block = (
        f"\n\nCategory-specific guidance for \"{category}\":\n{category_guidance}\n"
        if category_guidance
        else ""
    )

    system = (
        "You are an expert evaluator for the MUM (Multi-User Memory) Benchmark. "
        "This benchmark evaluates AI memory systems that manage independent, parallel "
        "conversation sessions with multiple users (typically 4 users, 7 sessions each). "
        "All conversations are grounded in fictional scenarios and documents.\n\n"
        "The system being evaluated must correctly:\n"
        "- Track which user said what (user attribution)\n"
        "- Synthesize information across users (cross-user reasoning)\n"
        "- Detect and resolve conflicts between users' claims\n"
        "- Track how users' beliefs evolved over time (corrections, retractions)\n"
        "- Respect authority hierarchies and information boundaries\n\n"
        "Your task is to compare a predicted answer against a gold (reference) answer "
        "and score the prediction on four dimensions.\n\n"
        "Score each dimension from 1 to 5:\n"
        "  1 = Completely wrong or missing\n"
        "  2 = Mostly wrong with minor correct elements\n"
        "  3 = Partially correct but with significant gaps or errors\n"
        "  4 = Mostly correct with minor gaps or imprecisions\n"
        "  5 = Fully correct and complete\n\n"
        "Scoring dimensions:\n"
        "  - correctness: Are the core facts in the predicted answer correct? "
        "Compare specific claims, numbers, names, and conclusions against the gold answer.\n"
        "  - completeness: Does the predicted answer cover ALL key points in the gold answer? "
        "A correct but partial answer should score lower here.\n"
        "  - attribution: Does it correctly attribute information to the right users? "
        "Naming the wrong user, omitting user attribution, or conflating users' positions "
        "should lower this score.\n"
        "  - hallucination: Does it avoid fabricating facts not in the gold answer? "
        "(5 = no hallucinations, 1 = severe hallucinations). Extra details that are "
        "plausible but not in the gold answer count as mild hallucination.\n\n"
        "Scoring guidelines:\n"
        "- Be strict: a score of 5 means the predicted answer is essentially equivalent "
        "to the gold answer on that dimension.\n"
        "- Partial credit is appropriate: if the answer gets the main point but misses "
        "details, score 3-4 depending on severity.\n"
        "- The predicted answer does not need to use the exact same wording as the gold "
        "answer — semantic equivalence is sufficient.\n"
        "- If the predicted answer says 'insufficient information' or similar when the "
        "gold answer contains a concrete answer, score correctness and completeness as 1."
        f"{category_block}\n"
        "Respond with ONLY valid JSON in this format:\n"
        "{\n"
        '  "correctness": <1-5>,\n'
        '  "completeness": <1-5>,\n'
        '  "attribution": <1-5>,\n'
        '  "hallucination": <1-5>,\n'
        '  "reasoning": "<brief explanation of each score>"\n'
        "}"
    )

    user_msg = (
        f"## Evaluation Category: {category}\n\n"
        f"## Question\n{question}\n\n"
        f"## Gold Answer (Reference)\n{gold_answer}\n\n"
        f"## Predicted Answer\n{predicted_answer}\n\n"
        f"Score the predicted answer against the gold answer on all four dimensions. "
        f"Respond with JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def build_binary_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
) -> list[dict]:
    """Build prompt for binary (correct/incorrect) evaluation.

    Used for adversarial_confusion where the answer is binary (Yes/No with correction).
    """
    system = (
        "You are an expert evaluator for the MUM (Multi-User Memory) Benchmark.\n\n"
        "You are scoring an 'adversarial confusion' question. These questions "
        "deliberately swap users or misattribute claims to test whether the system "
        "resists misattribution. The format is typically:\n"
        '  Question: "Did [wrong user] [action]?"\n'
        '  Correct answer: "No — [correct user] did. [Wrong user] actually [their real position]."\n\n'
        "The predicted answer is CORRECT if it:\n"
        "- Correctly rejects the false attribution (says No or equivalent)\n"
        "- Identifies the actual user who performed the action\n"
        "The predicted answer is INCORRECT if it:\n"
        "- Agrees with the false attribution\n"
        "- Fails to identify the correct user\n"
        "- Expresses uncertainty without correcting the misattribution\n\n"
        "Respond with ONLY valid JSON:\n"
        "{\n"
        '  "correct": true or false,\n'
        '  "reasoning": "<brief explanation>"\n'
        "}"
    )

    user_msg = (
        f"## Question (adversarial confusion — deliberately misattributes a claim)\n"
        f"{question}\n\n"
        f"## Gold Answer (correct response)\n{gold_answer}\n\n"
        f"## Predicted Answer\n{predicted_answer}\n\n"
        f"Does the predicted answer correctly reject the false attribution and "
        f"identify the right user? Respond with JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
