from __future__ import annotations

import json

from src.models.schemas import ExtractedMemory, InjectedConflict


def build_conflict_detection_prompt(
    all_memories: list[ExtractedMemory],
    injected_conflicts: list[InjectedConflict],
    relationship_type: str,
    authority_context: str,
    scenario_id: str,
) -> str:
    memories_json = json.dumps(
        [m.model_dump() for m in all_memories], indent=2, ensure_ascii=False
    )

    expected_conflicts = []
    for c in injected_conflicts:
        expected_conflicts.append(
            f"- [{c.conflict_id}] {c.topic}: {c.nature} (Resolution: {c.resolution})"
        )
    expected_section = "\n".join(expected_conflicts) if expected_conflicts else "None specified."

    return f"""You are an expert conflict detection system. Given the extracted memories
from ALL users in this scenario, identify disagreements between users on
the same topic.

## Expected Conflicts (from scenario design)
{expected_section}

## All Extracted Memories (from all users)
{memories_json}

## Rules
1. Identify all conflicts — both the pre-specified ones AND any organic
   ones that emerged naturally from persona differences.
2. For each conflict, determine:
   - conflict_type: factual (disagreement about facts), interpretive (different
     readings of the same data), strategic (different approaches/preferences),
     or authority (who has the right to decide)
   - resolution: preserve_both (keep both views, neither is wrong),
     supersede (later/better evidence replaces earlier), authority_wins
     (higher authority's view prevails on opinions), or temporal_supersession
     (later finding overrides earlier)
3. Map each conflict to evidence links (session-level references where positions
   are expressed). Use the session_id from the memories.
4. Record first_surfaced (timestamp of earliest relevant memory)
   and last_updated (timestamp of latest relevant memory).
5. Include evidence links for EACH user's position — the session(s)
   where they express their view.

## Scenario Context
Scenario ID: {scenario_id}
Relationship type: {relationship_type}
{authority_context}

## Output Format
Return a JSON object with a "conflicts" key:
{{"conflicts": [
  {{
    "conflict_id": "C{{scenario_id}}-{{index}}",
    "scenario_id": "{scenario_id}",
    "users_involved": ["user_id_1", "user_id_2"],
    "topic": "description of the disagreement topic",
    "conflict_type": "factual|interpretive|strategic|authority",
    "positions": {{"user_id_1": "their position", "user_id_2": "their position"}},
    "resolution": "preserve_both|supersede|authority_wins|temporal_supersession",
    "resolution_detail": "explanation of why this resolution applies",
    "evidence": [
      {{"user_id": "user_id_1", "session_id": "session where position is expressed"}},
      {{"user_id": "user_id_2", "session_id": "session where position is expressed"}}
    ],
    "first_surfaced": "ISO 8601 timestamp",
    "last_updated": "ISO 8601 timestamp"
  }}
]}}"""
