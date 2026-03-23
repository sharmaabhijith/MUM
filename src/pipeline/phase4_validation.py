from __future__ import annotations

import logging
import random

from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.models.schemas import (
    ConversationSession,
    ScenarioOutput,
    UserProfile,
    ValidationReport,
)
from src.prompts.quality_check import (
    build_memory_extractability_prompt,
    build_persona_fidelity_prompt,
    build_question_answerability_prompt,
)

logger = logging.getLogger("mum")

# Default thresholds for passing validation
DEFAULT_THRESHOLDS = {
    "conflict_coverage": 0.8,
    "memory_extractability": 0.8,
    "evidence_validity": 0.8,
    "question_answerability": 0.8,
    "persona_fidelity": 3.5,  # Average score out of 5
}


class ValidationPipeline:
    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker | None = None,
        thresholds: dict | None = None,
        max_samples: int = 50,
    ):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker or llm_client.cost_tracker
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.max_samples = max_samples

    def validate_scenario(self, scenario_output: ScenarioOutput) -> ValidationReport:
        config = scenario_output.config
        logger.info(f"Validating scenario {config.scenario_id}")

        # 1. Conflict Coverage
        conflict_result = self._check_conflict_coverage(scenario_output)

        # 2. Memory Extractability (LLM-as-judge, sampled)
        memory_result = self._check_memory_extractability(scenario_output)

        # 3. Evidence Validity
        evidence_result = self._check_evidence_validity(scenario_output)

        # 4. Question Answerability (LLM-as-judge, sampled)
        answerability_result = self._check_question_answerability(scenario_output)

        # 5. Persona Fidelity (LLM-as-judge, sampled)
        persona_result = self._check_persona_fidelity(scenario_output)

        # Determine overall pass
        overall_pass = (
            conflict_result.get("score", 0) >= self.thresholds["conflict_coverage"]
            and memory_result.get("score", 0) >= self.thresholds["memory_extractability"]
            and evidence_result.get("score", 0) >= self.thresholds["evidence_validity"]
            and answerability_result.get("score", 0) >= self.thresholds["question_answerability"]
            and persona_result.get("score", 0) >= self.thresholds["persona_fidelity"]
        )

        report = ValidationReport(
            scenario_id=config.scenario_id,
            conflict_coverage=conflict_result,
            memory_extractability=memory_result,
            evidence_validity=evidence_result,
            question_answerability=answerability_result,
            persona_fidelity=persona_result,
            overall_pass=overall_pass,
        )

        logger.info(f"Validation {'PASSED' if overall_pass else 'FAILED'} for scenario {config.scenario_id}")
        return report

    def _check_conflict_coverage(self, scenario_output: ScenarioOutput) -> dict:
        """Check that every injected conflict appears in annotations."""
        config = scenario_output.config
        detected_conflicts = scenario_output.conflicts
        injected = config.injected_conflicts

        found = 0
        missing = []
        for ic in injected:
            matched = any(
                ic.topic.lower() in dc.topic.lower()
                or any(u in dc.users_involved for u in ic.users)
                for dc in detected_conflicts
            )
            if matched:
                found += 1
            else:
                missing.append(ic.conflict_id)

        total = len(injected) if injected else 1
        score = found / total

        return {
            "score": score,
            "found": found,
            "total": len(injected),
            "missing": missing,
            "pass": score >= self.thresholds["conflict_coverage"],
        }

    def _check_memory_extractability(self, scenario_output: ScenarioOutput) -> dict:
        """LLM-as-judge: verify memories are grounded in source sessions."""
        memories = scenario_output.memories
        conversations = scenario_output.conversations

        # Sample memories
        sample = random.sample(memories, min(self.max_samples, len(memories)))
        session_map = {c.session_id: c for c in conversations}

        grounded = 0
        checked = 0

        for memory in sample:
            session = session_map.get(memory.session_id)
            if not session:
                continue

            prompt = build_memory_extractability_prompt(memory, session)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate_json(
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                phase="validation_memory",
            )

            if response.get("grounded", False):
                grounded += 1
            checked += 1

        score = grounded / checked if checked > 0 else 0

        return {
            "score": score,
            "grounded": grounded,
            "checked": checked,
            "pass": score >= self.thresholds["memory_extractability"],
        }

    def _check_evidence_validity(self, scenario_output: ScenarioOutput) -> dict:
        """Check that eval question evidence links resolve to real sessions."""
        questions = scenario_output.eval_questions
        session_ids = {c.session_id for c in scenario_output.conversations}

        valid = 0
        invalid = 0
        invalid_questions = []

        for q in questions:
            all_valid = all(e.session_id in session_ids for e in q.evidence)
            if all_valid and q.evidence:
                valid += 1
            else:
                invalid += 1
                invalid_questions.append(q.question_id)

        total = valid + invalid
        score = valid / total if total > 0 else 0

        return {
            "score": score,
            "valid": valid,
            "invalid": invalid,
            "invalid_questions": invalid_questions[:20],
            "pass": score >= self.thresholds["evidence_validity"],
        }

    def _check_question_answerability(self, scenario_output: ScenarioOutput) -> dict:
        """LLM-as-judge: are questions answerable from annotations?"""
        questions = scenario_output.eval_questions
        sample = random.sample(questions, min(self.max_samples, len(questions)))

        # Build lookup structures
        memory_map = {m.memory_id: m for m in scenario_output.memories}
        conflict_map = {c.conflict_id: c for c in scenario_output.conflicts}

        answerable = 0
        checked = 0

        for q in sample:
            relevant_data = {
                "memories": [
                    memory_map[mid].model_dump()
                    for mid in q.required_memories
                    if mid in memory_map
                ],
                "conflicts": [
                    conflict_map[cid].model_dump()
                    for cid in q.required_conflicts
                    if cid in conflict_map
                ],
            }

            prompt = build_question_answerability_prompt(q, relevant_data)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate_json(
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                phase="validation_answerability",
            )

            if response.get("answerable", False):
                answerable += 1
            checked += 1

        score = answerable / checked if checked > 0 else 0

        return {
            "score": score,
            "answerable": answerable,
            "checked": checked,
            "pass": score >= self.thresholds["question_answerability"],
        }

    def _check_persona_fidelity(self, scenario_output: ScenarioOutput) -> dict:
        """LLM-as-judge: rate persona consistency per session."""
        config = scenario_output.config
        conversations = scenario_output.conversations
        user_map = {u.user_id: u for u in config.users}

        # Sample sessions
        sample = random.sample(
            conversations, min(self.max_samples, len(conversations))
        )

        scores = []
        for session in sample:
            user = user_map.get(session.user_id)
            if not user:
                continue

            prompt = build_persona_fidelity_prompt(session, user)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate_json(
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                phase="validation_persona",
            )

            score = response.get("score", 3)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "score": avg_score,
            "scores": scores,
            "sessions_checked": len(scores),
            "pass": avg_score >= self.thresholds["persona_fidelity"],
        }
