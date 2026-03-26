from __future__ import annotations


def build_conversation_user_prompt(session_number: int, turns_per_session: int) -> str:
    total_messages = turns_per_session * 2
    return (
        f"Generate Session {session_number} now.\n\n"
        f"Produce exactly {total_messages} messages in the JSON 'turns' array: "
        f"{turns_per_session} user messages and {turns_per_session} assistant responses, "
        f"strictly alternating (user, assistant, user, assistant, ...). "
        f"The array must contain exactly {total_messages} objects.\n\n"
        f"REMINDERS:\n"
        f"- Every AI assistant response MUST cite specific document locations "
        f"(section numbers, table numbers, dates, figure references, page references, "
        f"named entities, statistics, or other document-internal identifiers) from the "
        f"provided source documents.\n"
        f"- Do NOT use general knowledge or external information not found in the documents.\n"
        f"- User messages must reference the documents using the user's reference_style.\n"
        f"- Stay in character for the user persona throughout.\n"
        f"- Follow the session-specific behavior described in the system prompt.\n\n"
        f"Output valid JSON with a 'turns' key containing the array of turn objects. "
        f"Each turn object must have: turn (int), role ('user' or 'assistant'), "
        f"content (str), and timestamp (ISO 8601 string). "
        f"Pair user and assistant messages with the same turn number "
        f"(turn 1 for the first user+assistant pair, turn 2 for the second, etc. up to turn {turns_per_session}). "
        f"Ensure timestamps advance by 2-5 minutes between messages."
    )
