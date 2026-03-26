from __future__ import annotations

import json

from datagen.models.schemas import ConversationSession


def build_summary_prompt(session: ConversationSession, prior_summary: str) -> str:
    turns_json = json.dumps(
        [t.model_dump() for t in session.turns], indent=2, ensure_ascii=False
    )

    return f"""You are an expert conversation summarizer. Generate a session summary for the following conversation between a user and an AI assistant.

## Session Info
- Session ID: {session.session_id}
- User: {session.user_id}
- Session number: {session.session_number}

## Prior Sessions Summary
{prior_summary if prior_summary else "This is the first session."}

## Conversation
{turns_json}

## Instructions
Generate a summary of 150-250 words that captures:
1. What the user discussed and asked about
2. Key conclusions or understanding reached
3. Any positions taken, opinions expressed, or strategies stated
4. Any positions that changed from prior sessions (if applicable)
5. Any misconceptions stated or corrections accepted

## Output Format
Return a JSON object:
{{
  "summary": "150-250 word summary text",
  "key_facts": ["fact 1", "fact 2", ...],
  "positions_taken": ["position 1", "position 2", ...],
  "positions_changed": ["change 1", ...]
}}

The key_facts should be specific, atomic statements about what the user learned or discussed.
The positions_taken should capture opinions, preferences, or strategies the user expressed.
The positions_changed should note any shifts from prior sessions (empty list if none)."""
