from __future__ import annotations

import json

from src.models.schemas import ConversationSession, EvalQuestion, ExtractedMemory, UserProfile


def build_persona_fidelity_prompt(
    session: ConversationSession, user: UserProfile
) -> str:
    user_turns = [t for t in session.turns if t.role == "user"]
    turns_text = "\n".join(
        f"[Turn {t.turn_number}] {t.content}" for t in user_turns
    )

    return f"""Rate the persona fidelity of this conversation session on a scale of 1-5.

## Expected Persona
- Name: {user.display_name}
- Communication style: {user.communication_style}
- Reference style: {user.reference_style}
- Emotional tendencies: {user.emotional_tendencies}
- Session behavior (session {session.session_number}): {user.session_evolution.get(f'session_{session.session_number}', 'N/A')}

## Example utterances (for voice matching):
{chr(10).join(f'- "{u}"' for u in user.example_utterances)}

## Actual User Messages in This Session
{turns_text}

## Rating Criteria
1 = Completely off-character, wrong voice, wrong topics
2 = Partially in character but major deviations
3 = Generally in character with some inconsistencies
4 = Strong persona match with minor deviations
5 = Perfect persona fidelity — voice, topics, emotional state all match

## Output Format
Return JSON: {{"score": 1-5, "reasoning": "explanation"}}"""


def build_question_answerability_prompt(
    question: EvalQuestion, relevant_data: dict
) -> str:
    return f"""Assess whether this evaluation question is answerable from the provided data.

## Question
{question.question}

## Gold Answer
{question.gold_answer}

## Available Evidence
{json.dumps(relevant_data, indent=2, ensure_ascii=False, default=str)[:4000]}

## Assessment Criteria
1. Is the gold answer supported by the evidence?
2. Are the evidence links valid and relevant?
3. Could a system with access to the conversations reasonably answer this?

## Output Format
Return JSON: {{"answerable": true/false, "reasoning": "explanation", "confidence": 0.0-1.0}}"""


def build_memory_extractability_prompt(
    memory: ExtractedMemory, session: ConversationSession
) -> str:
    user_turns = [t for t in session.turns if t.role == "user"]
    turns_text = "\n".join(
        f"[Turn {t.turn_number}] {t.content}" for t in user_turns
    )

    return f"""Verify that this extracted memory is grounded in the source session.

## Memory
- ID: {memory.memory_id}
- Type: {memory.memory_type}
- Content: {memory.content}
- Status: {memory.status}

## Source Session User Messages
{turns_text}

## Assessment
Is this memory explicitly stated or clearly implied by the user in this session?

## Output Format
Return JSON: {{"grounded": true/false, "reasoning": "explanation"}}"""
