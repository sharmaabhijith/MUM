from __future__ import annotations


def build_conversation_user_prompt(session_number: int, turns_per_session: int) -> str:
    return (
        f"Generate Session {session_number} now. "
        f"Produce exactly {turns_per_session} turns "
        f"(each turn = 1 user message + 1 assistant response). "
        f"Output valid JSON with a 'turns' key containing the array of turns. "
        f"Each turn object must have: turn (int), role ('user' or 'assistant'), "
        f"content (str), and timestamp (ISO 8601 string). "
        f"Ensure timestamps advance by 2-5 minutes between messages."
    )
