from __future__ import annotations


def build_conversation_user_prompt(session_number: int, turns_per_session: int) -> str:
    return (
        f"Generate Session {session_number} now.\n\n"
        f"Produce exactly {turns_per_session} turns "
        f"(each turn = 1 user message + 1 assistant response = {turns_per_session * 2} total messages).\n\n"
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
        f"Ensure timestamps advance by 2-5 minutes between messages."
    )
