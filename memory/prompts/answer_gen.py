"""Prompts for generating answers to evaluation questions using memory context."""


def build_answer_prompt(question: str, context: str, method_name: str) -> list[dict]:
    """Build the prompt for answering an eval question given memory context.

    Returns a list of messages suitable for the LLM client.
    """
    system = (
        "You are an AI assistant being evaluated on the MUM (Multi-User Memory) "
        "Benchmark. In this benchmark, you serve as a shared AI assistant that has "
        "conducted independent, parallel conversation sessions with multiple users "
        "(typically 4) over time (typically 7 sessions each). Each user has their own "
        "expertise, biases, misconceptions, and communication style. The conversations "
        "are grounded in fictional scenarios and documents — all facts come from these "
        "documents, not from general world knowledge.\n\n"
        "Your task is to answer evaluation questions that test your ability to manage "
        "and reason over this multi-user conversational memory. Questions may require:\n"
        "- **User attribution**: Identifying which specific user said, claimed, or "
        "discovered something.\n"
        "- **Cross-user synthesis**: Combining information from multiple users' "
        "conversations to form a complete picture.\n"
        "- **Conflict resolution**: Recognizing disagreements between users and "
        "determining the correct answer (using document evidence or authority hierarchy).\n"
        "- **Temporal tracking**: Understanding how a user's beliefs evolved across "
        "sessions (e.g., a misconception that was corrected).\n"
        "- **Information gap detection**: Identifying what one user knows that another "
        "user needs but doesn't have.\n"
        "- **Provenance tracing**: Reconstructing the history of a fact or decision "
        "across users and sessions.\n\n"
        "Critical rules:\n"
        "- Use ONLY the provided memory context to answer. Never use general knowledge.\n"
        "- Always attribute information to specific users by name/ID when relevant.\n"
        "- Distinguish carefully between different users' statements, beliefs, and "
        "positions — do not conflate them.\n"
        "- When users disagree, state both positions and explain which is correct "
        "and why (based on document evidence or authority).\n"
        "- Track temporal evolution: note when a user held an incorrect belief, when "
        "they were corrected, and whether they accepted the correction.\n"
        "- Be precise about facts, numbers, names, dates, and session references.\n"
        "- If the memory context is insufficient to answer, say so explicitly rather "
        "than guessing."
    )

    if context:
        user_msg = (
            f"## Memory Context ({method_name})\n\n"
            f"{context}\n\n"
            f"## Question\n\n"
            f"{question}\n\n"
            f"## Answer\n\n"
            f"Answer based ONLY on the memory context above. Be specific about which "
            f"users said what, include relevant session details, and address all aspects "
            f"of the question."
        )
    else:
        user_msg = (
            f"## Question\n\n"
            f"{question}\n\n"
            f"## Answer\n\n"
            f"No conversation memory is available. State that the information needed "
            f"to answer this question is not available in the provided context."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
