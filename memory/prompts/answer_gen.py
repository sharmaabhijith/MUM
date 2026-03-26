"""Prompts for generating answers to evaluation questions using memory context."""


def build_answer_prompt(question: str, context: str, method_name: str) -> list[dict]:
    """Build the prompt for answering an eval question given memory context.

    Returns a list of messages suitable for the LLM client.
    """
    system = (
        "You are an AI assistant that manages conversations with multiple users. "
        "You have access to a memory store that contains information from past "
        "conversations with different users. Use ONLY the provided memory context "
        "to answer the question. If the memory context does not contain enough "
        "information to answer, say so explicitly.\n\n"
        "Important guidelines:\n"
        "- Attribute information to specific users when asked.\n"
        "- Distinguish between what different users said or believe.\n"
        "- Track how information evolved across sessions.\n"
        "- Never fabricate information not present in the memory context.\n"
        "- Be precise about facts, numbers, and names."
    )

    if context:
        user_msg = (
            f"## Memory Context ({method_name})\n\n"
            f"{context}\n\n"
            f"## Question\n\n"
            f"{question}\n\n"
            f"## Answer\n\n"
            f"Provide a thorough, accurate answer based on the memory context above."
        )
    else:
        user_msg = (
            f"## Question\n\n"
            f"{question}\n\n"
            f"## Answer\n\n"
            f"Answer this question to the best of your ability. "
            f"No conversation memory is available."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
