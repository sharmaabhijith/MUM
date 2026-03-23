from __future__ import annotations

from src.models.schemas import UserProfile
from src.pipeline.phase1_document_prep import DocumentContext


class PromptBuilder:
    @staticmethod
    def build_document_section(doc_context: DocumentContext) -> str:
        return doc_context.context_block

    @staticmethod
    def build_user_persona_section(user: UserProfile) -> str:
        sections = [
            f"## User Persona — Core",
            f"- Name: {user.display_name}",
            f"- Role/Expertise: {user.expertise}",
            f"- Focus areas: {', '.join(user.focus_areas)}",
            f"- Known biases: {', '.join(user.biases)}",
        ]

        if user.domain_authority:
            sections.append(f"- Domain authority: {user.domain_authority}")
        if user.side:
            sections.append(f"- Side: {user.side}")
        if user.sequence_order is not None:
            sections.append(f"- Sequence order: {user.sequence_order}")

        sections.extend([
            f"\n## User Persona — Communication Style",
            user.communication_style,
            f"\n## User Persona — How They Read Documents",
            user.document_reading_pattern,
            f"\n## User Persona — How They React When Corrected",
            user.reaction_to_corrections,
            f"\n## User Persona — What They Don't Know",
            f"Knowledge gaps: {', '.join(user.knowledge_gaps)}",
            f"Misconceptions they hold: {', '.join(user.misconceptions) if user.misconceptions else 'None'}",
            f"\n## User Persona — Emotional Tendencies",
            user.emotional_tendencies,
            f"\n## User Persona — How They Reference Documents",
            user.reference_style,
            f"\n## User Persona — Example Messages (match this voice and style)",
        ])
        for i, utterance in enumerate(user.example_utterances, 1):
            sections.append(f'{i}. "{utterance}"')

        return "\n".join(sections)

    @staticmethod
    def build_authority_section(user: UserProfile, authority_context: str) -> str:
        lines = [
            f"## Authority Context",
            f"Authority level: {user.authority_level} (weight: {user.authority_weight})",
            authority_context,
        ]
        return "\n".join(lines)
