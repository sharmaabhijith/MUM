from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.llm.token_counter import TokenCounter
from src.models.schemas import (
    ConversationSession,
    ConversationTurn,
    SessionSummary,
)
from src.pipeline.phase1_document_prep import DocumentContext
from src.prompts.conversation_system import build_conversation_system_prompt
from src.prompts.conversation_user import build_conversation_user_prompt
from src.prompts.session_summary import build_summary_prompt
from src.scenarios.base import BaseScenario
from src.utils.io import write_json

logger = logging.getLogger("mum")


class ConversationGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        token_counter: TokenCounter | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.llm_client = llm_client
        self.token_counter = token_counter or TokenCounter()
        self.cost_tracker = cost_tracker or llm_client.cost_tracker

    def generate_scenario(
        self,
        scenario: BaseScenario,
        doc_context: DocumentContext,
        output_dir: Path | None = None,
    ) -> tuple[list[ConversationSession], list[SessionSummary]]:
        config = scenario.config
        all_sessions: list[ConversationSession] = []
        all_summaries: list[SessionSummary] = []

        # Determine save directory for incremental saves
        save_dir = None
        if output_dir is not None:
            save_dir = Path(output_dir) / f"scenario_{config.scenario_id}"

        for user in config.users:
            # Check if this user's data already exists (resume support)
            if save_dir and self._user_data_exists(save_dir, user.user_id, config.sessions_per_user):
                logger.info(
                    f"Skipping {user.display_name} — "
                    f"all {config.sessions_per_user} sessions already saved"
                )
                loaded_sessions, loaded_summaries = self._load_user_data(
                    save_dir, user.user_id, config.sessions_per_user
                )
                all_sessions.extend(loaded_sessions)
                all_summaries.extend(loaded_summaries)
                continue

            logger.info(
                f"Generating conversations for {user.display_name} "
                f"({config.sessions_per_user} sessions)"
            )
            accumulated_summary = ""
            user_sessions: list[ConversationSession] = []
            user_summaries: list[SessionSummary] = []

            for session_num in range(1, config.sessions_per_user + 1):
                # Get session timestamp
                session_timestamp = scenario.get_session_timestamp(
                    user.user_id, session_num
                )

                # Get conflicts to seed
                target_conflicts = scenario.get_conflicts_for_session(
                    user.user_id, session_num
                )

                # Build prompts
                turns_this_session = config.get_turns_for_session(session_num)
                system_prompt = build_conversation_system_prompt(
                    user=user,
                    scenario=scenario,
                    session_number=session_num,
                    session_timestamp=session_timestamp,
                    prior_summaries=accumulated_summary,
                    target_conflicts=target_conflicts,
                    doc_context=doc_context,
                    turns_this_session=turns_this_session,
                )
                user_prompt = build_conversation_user_prompt(
                    session_number=session_num,
                    turns_per_session=turns_this_session,
                )

                # Generate conversation
                logger.info(
                    f"  Session {session_num}/{config.sessions_per_user} "
                    f"for {user.display_name}"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = self.llm_client.generate_json(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=16384,
                    phase="conversation_generation",
                )

                # Parse into ConversationSession
                session = self._parse_conversation_response(
                    response=response,
                    user_id=user.user_id,
                    scenario_id=config.scenario_id,
                    session_number=session_num,
                    session_timestamp=session_timestamp,
                    target_conflicts=[c.conflict_id for c in target_conflicts],
                    authority_level=user.authority_level,
                )

                # Validate
                warnings = self._validate_session(session, turns_this_session)
                for w in warnings:
                    logger.warning(f"  Validation: {w}")

                user_sessions.append(session)

                # Generate session summary immediately (feed-forward)
                summary = self._generate_summary(session, accumulated_summary)
                user_summaries.append(summary)

                # Accumulate summaries for next session
                accumulated_summary += (
                    f"\n\n--- Session {session_num} Summary ---\n{summary.summary}"
                )

            # Save after all sessions for this user are complete
            if save_dir:
                self._save_user_data(
                    save_dir, user.user_id, user_sessions, user_summaries
                )
                logger.info(
                    f"  Saved {len(user_sessions)} sessions for {user.display_name}"
                )

            all_sessions.extend(user_sessions)
            all_summaries.extend(user_summaries)

        logger.info(
            f"Generated {len(all_sessions)} sessions, "
            f"{len(all_summaries)} summaries for scenario {config.scenario_id}"
        )
        return all_sessions, all_summaries

    def _user_data_exists(
        self, save_dir: Path, user_id: str, expected_sessions: int
    ) -> bool:
        """Check if all sessions for a user already exist on disk."""
        conv_dir = save_dir / "conversations"
        summ_dir = save_dir / "summaries"
        for sn in range(1, expected_sessions + 1):
            conv_file = conv_dir / f"{user_id}_session_{sn}.json"
            summ_file = summ_dir / f"{user_id}_session_{sn}_summary.json"
            if not conv_file.exists() or not summ_file.exists():
                return False
        return True

    def _load_user_data(
        self, save_dir: Path, user_id: str, expected_sessions: int
    ) -> tuple[list[ConversationSession], list[SessionSummary]]:
        """Load previously saved sessions and summaries for a user."""
        import json as _json

        sessions = []
        summaries = []
        for sn in range(1, expected_sessions + 1):
            conv_file = save_dir / "conversations" / f"{user_id}_session_{sn}.json"
            summ_file = save_dir / "summaries" / f"{user_id}_session_{sn}_summary.json"
            with open(conv_file) as f:
                sessions.append(ConversationSession.model_validate(_json.load(f)))
            with open(summ_file) as f:
                summaries.append(SessionSummary.model_validate(_json.load(f)))
        return sessions, summaries

    def _save_user_data(
        self,
        save_dir: Path,
        user_id: str,
        sessions: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        """Save all sessions and summaries for a user to disk."""
        conv_dir = save_dir / "conversations"
        summ_dir = save_dir / "summaries"
        for session in sessions:
            write_json(
                session.model_dump(),
                conv_dir / f"{user_id}_session_{session.session_number}.json",
            )
        for summary in summaries:
            write_json(
                summary.model_dump(),
                summ_dir / f"{user_id}_session_{summary.session_number}_summary.json",
            )

    def _generate_summary(
        self, session: ConversationSession, prior_summary: str
    ) -> SessionSummary:
        prompt = build_summary_prompt(session, prior_summary)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate_json(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            phase="session_summary",
        )

        return SessionSummary(
            session_id=session.session_id,
            scenario_id=session.scenario_id,
            user_id=session.user_id,
            session_number=session.session_number,
            summary=response.get("summary", ""),
            key_facts=response.get("key_facts", []),
            positions_taken=response.get("positions_taken", []),
            positions_changed=response.get("positions_changed", []),
        )

    def _parse_conversation_response(
        self,
        response: dict | list,
        user_id: str,
        scenario_id: str,
        session_number: int,
        session_timestamp: str,
        target_conflicts: list[str],
        authority_level: str,
    ) -> ConversationSession:
        session_id = f"s{scenario_id}_{user_id}_{session_number}"

        # Handle both {"turns": [...]} and direct array format
        turns_data = response if isinstance(response, list) else response.get("turns", [])

        turns = []
        for t in turns_data:
            # Handle combined format: {"turn": 1, "user": "...", "assistant": "..."}
            if "user" in t and "assistant" in t and "role" not in t:
                turn_num = t.get("turn", len(turns) // 2 + 1)
                ts = t.get("timestamp", session_timestamp)
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="user",
                        content=t["user"],
                        timestamp=ts,
                    )
                )
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="assistant",
                        content=t["assistant"],
                        timestamp=ts,
                    )
                )
            # Handle combined format: {"turn": 1, "user_message": "...", "assistant_response": "..."}
            elif "user_message" in t and "assistant_response" in t:
                turn_num = t.get("turn", len(turns) // 2 + 1)
                ts = t.get("timestamp", session_timestamp)
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="user",
                        content=t["user_message"],
                        timestamp=ts,
                    )
                )
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="assistant",
                        content=t["assistant_response"],
                        timestamp=ts,
                    )
                )
            # Standard format: {"turn": 1, "role": "user", "content": "..."}
            else:
                turns.append(
                    ConversationTurn(
                        turn_number=t.get("turn", len(turns) // 2 + 1),
                        role=t.get("role", "user"),
                        content=t.get("content", ""),
                        timestamp=t.get("timestamp", session_timestamp),
                    )
                )

        return ConversationSession(
            session_id=session_id,
            scenario_id=scenario_id,
            user_id=user_id,
            session_number=session_number,
            session_timestamp=session_timestamp,
            turns=turns,
            target_conflicts=target_conflicts,
            metadata={"authority_level": authority_level},
        )

    def _validate_session(
        self, session: ConversationSession, expected_turns: int
    ) -> list[str]:
        warnings = []
        expected_messages = expected_turns * 2  # user + assistant per turn

        if len(session.turns) != expected_messages:
            warnings.append(
                f"Expected {expected_messages} messages ({expected_turns} turns), "
                f"got {len(session.turns)}"
            )

        # Check alternating roles
        for i, turn in enumerate(session.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if turn.role != expected_role:
                warnings.append(
                    f"Turn {i}: expected role '{expected_role}', got '{turn.role}'"
                )
                break

        # Check non-empty content
        empty_turns = [i for i, t in enumerate(session.turns) if not t.content.strip()]
        if empty_turns:
            warnings.append(f"Empty content in turns: {empty_turns}")

        return warnings
