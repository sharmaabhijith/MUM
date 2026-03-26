from __future__ import annotations

from datagen.models.schemas import UserProfile
from datagen.pipeline.phase1_document_prep import DocumentContext


class PromptBuilder:
    @staticmethod
    def build_document_section(doc_context: DocumentContext) -> str:
        """Build the full document text section with clear boundaries.

        Each document is wrapped in clear markers so the LLM can identify which
        document a piece of content comes from when generating citations.
        """
        blocks = []
        for name, text in doc_context.documents.items():
            token_count = doc_context.token_counts.get(name, 0)
            blocks.append(
                f"\n{'=' * 80}\n"
                f"DOCUMENT: {name}\n"
                f"Token count: ~{token_count}\n"
                f"{'=' * 80}\n\n"
                f"{text}\n\n"
                f"{'=' * 80}\n"
                f"END OF: {name}\n"
                f"{'=' * 80}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def build_user_persona_section(user: UserProfile) -> str:
        sections = [
            f"## User Persona — Core Identity",
            f"- Name: {user.display_name}",
            f"- Role/Expertise: {user.expertise}",
            f"- Focus areas: {', '.join(user.focus_areas)}",
            f"- Known biases: {', '.join(user.biases)}",
        ]

        if user.domain_authority:
            sections.append(f"- Domain authority: {user.domain_authority}")
        if user.side:
            sections.append(f"- Side/Team: {user.side}")
        if user.sequence_order is not None:
            sections.append(f"- Sequence position: {user.sequence_order}")

        sections.extend([
            f"\n## User Persona — Communication Style",
            f"How this user writes messages:",
            user.communication_style,
            f"\n## User Persona — Document Reading Pattern",
            f"Which documents/sections this user reads deeply, skims, or skips entirely:",
            user.document_reading_pattern,
            f"\n## User Persona — Reaction to Corrections",
            f"How this user responds when the AI corrects them or challenges their view:",
            user.reaction_to_corrections,
            f"\n## User Persona — Knowledge Gaps and Misconceptions",
            f"Knowledge gaps (topics they don't understand well):",
        ])
        for gap in user.knowledge_gaps:
            sections.append(f"  - {gap}")
        sections.append(f"Misconceptions they hold (may state these confidently as fact):")
        if user.misconceptions:
            for m in user.misconceptions:
                sections.append(f"  - {m}")
        else:
            sections.append(f"  - None")
        sections.extend([
            f"\n## User Persona — Emotional Tendencies",
            user.emotional_tendencies,
            f"\n## User Persona — How They Reference Documents",
            f"Citation style this user uses (match this exactly in generated messages):",
            user.reference_style,
            f"\n## User Persona — Example Messages (MATCH THIS VOICE AND STYLE)",
            f"Study these examples carefully. The user's messages must sound like these:",
        ])
        for i, utterance in enumerate(user.example_utterances, 1):
            sections.append(f'{i}. "{utterance}"')

        return "\n".join(sections)

    @staticmethod
    def build_authority_section(user: UserProfile, authority_context: str) -> str:
        lines = [
            f"## Authority Context",
            f"This user's authority level: {user.authority_level} (weight: {user.authority_weight})",
            f"Authority structure for this scenario:",
            authority_context,
        ]
        return "\n".join(lines)
