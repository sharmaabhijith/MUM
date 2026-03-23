from __future__ import annotations

import json

from src.models.schemas import ConversationSession, ExtractedMemory
from src.pipeline.phase1_document_prep import DocumentContext


def build_memory_extraction_prompt(
    session: ConversationSession,
    prior_memories: list[ExtractedMemory],
    doc_context: DocumentContext,
) -> str:
    turns_json = json.dumps(
        [t.model_dump() for t in session.turns], indent=2, ensure_ascii=False
    )

    prior_memories_json = "[]"
    if prior_memories:
        prior_memories_json = json.dumps(
            [m.model_dump() for m in prior_memories], indent=2, ensure_ascii=False
        )

    return f"""You are an expert memory extraction system. Given the following conversation
between a user and an AI assistant (with the source documents for grounding),
extract all distinct durable memories — facts, opinions, preferences, strategies,
decisions, assessments, and diagnoses stated by the USER.

## Rules
1. Each memory must be a single, atomic statement.
2. Tag each memory with:
   - memory_type: one of [fact, opinion, strategy, directive, risk_flag, assessment, position, diagnosis, customer_communication]
   - status: active, superseded, or corrected
   - The session_id where the memory originates: "{session.session_id}"
   - The session timestamp: "{session.session_timestamp}"
3. Target: ~3-4 memories per session.
4. Only extract memories from the USER's statements, not the AI's.
   Exception: If the AI corrects a factual error and the user accepts the
   correction, extract BOTH the original (mark corrected) and the accepted
   correction (mark active).
5. Do NOT extract general knowledge — only user-specific views, preferences,
   decisions, and conclusions.
6. If a memory contradicts or updates an earlier memory from a prior session,
   mark the earlier one's ID in superseded_by of the NEW memory, and set
   the new memory's status to "active". The old memory should have its status
   changed to "superseded" — but since you cannot modify prior memories,
   note supersession in the "supersession_notes" field.

## Session Info
- Session ID: {session.session_id}
- Scenario ID: {session.scenario_id}
- User ID: {session.user_id}
- Session Number: {session.session_number}
- Timestamp: {session.session_timestamp}

## Conversation
{turns_json}

## Prior Sessions' Memories (for supersession detection)
{prior_memories_json}

## Documents (for grounding verification)
{doc_context.context_block[:3000]}
[Documents truncated for prompt length]

## Output Format
Return a JSON object with a "memories" key containing an array:
{{"memories": [
  {{
    "memory_id": "m_{{scenario_id}}_{{user_id}}_{{session_number}}_{{index}}",
    "scenario_id": "{session.scenario_id}",
    "user_id": "{session.user_id}",
    "session_id": "{session.session_id}",
    "memory_type": "fact|opinion|strategy|...",
    "content": "atomic memory statement",
    "status": "active|superseded|corrected",
    "superseded_by": null,
    "authority_level": "{session.metadata.get('authority_level', 'equal')}",
    "domain_tag": null,
    "side_tag": null,
    "timestamp": "{session.session_timestamp}",
    "supersession_notes": "optional note if this supersedes a prior memory"
  }}
]}}"""
