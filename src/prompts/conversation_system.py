from __future__ import annotations

import json

from src.models.schemas import InjectedConflict, UserProfile
from src.pipeline.phase1_document_prep import DocumentContext
from src.prompts.base import PromptBuilder
from src.scenarios.base import BaseScenario


def build_conversation_system_prompt(
    user: UserProfile,
    scenario: BaseScenario,
    session_number: int,
    session_timestamp: str,
    prior_summaries: str,
    target_conflicts: list[InjectedConflict],
    doc_context: DocumentContext,
) -> str:
    config = scenario.config
    persona_section = PromptBuilder.build_user_persona_section(user)
    authority_section = PromptBuilder.build_authority_section(
        user, scenario.get_authority_context()
    )
    document_section = PromptBuilder.build_document_section(doc_context)

    # Session-specific behavior
    session_key = f"session_{session_number}"
    session_behavior = user.session_evolution.get(session_key, "Continue naturally.")

    # Format conflict seeding
    conflict_section = ""
    if target_conflicts:
        conflict_lines = []
        for c in target_conflicts:
            conflict_lines.append(f"- Topic: {c.topic}\n  Nature: {c.nature}")
        conflict_section = "\n".join(conflict_lines)

    prompt = f"""You are simulating a conversation between a human user and an AI assistant.

## Context
The human user ("{user.display_name}") is interacting with an AI assistant that has access
to the following shared documents. The AI can reference specific sections, theorems, tables, etc.

{persona_section}

{authority_section}

## Temporal Context
Current date/time: {session_timestamp}
This is Session {session_number} of {config.sessions_per_user}.
The user has been working on this material since {config.timeline.start_date}.

## Session-Specific Behavior
{session_behavior}

## Previous Session Summary
{prior_summaries if prior_summaries else "This is the first session. No prior context."}

## Conflict Seeding (INTERNAL — do not reference directly)
The following disagreements should emerge NATURALLY from the user's persona.
Do NOT force these — let them arise from the user asking questions consistent with
their expertise and biases:
{conflict_section if conflict_section else "No specific conflicts to seed in this session."}

## Continuity Requirements
- The user may reference things they discussed in prior sessions.
- The user's understanding should BUILD across sessions — not restart from scratch.
- Opinions and positions should evolve: the user can change their mind.
- If the user held a position in a prior session, they should reference it
  (reinforcing, updating, or correcting it).
- The user's emotional state and confidence should match the session_evolution description.

## Generation Instructions
Generate a realistic conversation of exactly {config.turns_per_session} turns
(1 turn = 1 user message + 1 AI response).

Rules:
1. USER messages MUST match their communication_style, reference_style, and example_utterances.
   Read the example_utterances carefully — they define the user's VOICE.
2. USER messages must reflect their knowledge_gaps and misconceptions where relevant.
   If the user has a misconception, they should state it confidently (until corrected).
3. AI ASSISTANT responses must be grounded in the provided documents with specific citations.
4. Conversations must feel natural — tangents, follow-ups, confusion, topic changes.
5. Each turn should include a simulated timestamp (advancing ~2-5 minutes per turn).
6. The user must NOT know about other users' conversations.
7. The user's reaction_to_corrections should be realistic when the AI pushes back.
8. Session-specific behavior (from session_evolution) should be reflected in the conversation arc.

## Documents
{document_section}

## Output Format
Output a JSON object with a "turns" key containing an array of turns:
{{"turns": [
  {{"turn": 1, "role": "user", "content": "...", "timestamp": "{session_timestamp}"}},
  {{"turn": 1, "role": "assistant", "content": "...", "timestamp": "..."}},
  ...
]}}"""

    return prompt
