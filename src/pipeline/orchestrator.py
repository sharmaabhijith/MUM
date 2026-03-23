from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.llm.client import LLMClient, MockLLMClient
from src.llm.cost_tracker import CostTracker
from src.llm.token_counter import TokenCounter
from src.models.schemas import (
    BenchmarkDataset,
    GenerationReport,
    ScenarioOutput,
    ValidationReport,
)
from src.pipeline.phase1_document_prep import DocumentPreparer
from src.pipeline.phase2_conversation import ConversationGenerator
from src.pipeline.phase3_annotation import AnnotationPipeline
from src.pipeline.phase4_validation import ValidationPipeline
from src.scenarios import create_scenario, load_scenario
from src.utils.io import write_json

logger = logging.getLogger("mum")

OUTPUT_DIR = Path("output")


class PipelineOrchestrator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        annotation_model: str | None = None,
        dry_run: bool = False,
        output_dir: Path | str = OUTPUT_DIR,
    ):
        self.cost_tracker = CostTracker()
        self.token_counter = TokenCounter(model=model)
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

        if dry_run:
            self.llm_client = MockLLMClient(
                model=model, cost_tracker=self.cost_tracker
            )
            self.annotation_client = self.llm_client
        else:
            self.llm_client = LLMClient(
                model=model, cost_tracker=self.cost_tracker
            )
            ann_model = annotation_model or model
            if ann_model != model:
                self.annotation_client = LLMClient(
                    model=ann_model, cost_tracker=self.cost_tracker
                )
            else:
                self.annotation_client = self.llm_client

        self.doc_preparer = DocumentPreparer(token_counter=self.token_counter)
        self.conv_generator = ConversationGenerator(
            llm_client=self.llm_client,
            token_counter=self.token_counter,
            cost_tracker=self.cost_tracker,
        )
        self.annotation_pipeline = AnnotationPipeline(
            llm_client=self.annotation_client,
            cost_tracker=self.cost_tracker,
        )
        self.validation_pipeline = ValidationPipeline(
            llm_client=self.annotation_client,
            cost_tracker=self.cost_tracker,
        )

    def run_scenario(self, scenario_id: str) -> ScenarioOutput:
        start_time = time.time()
        logger.info(f"=== Running scenario {scenario_id} ===")

        # Load config
        config = load_scenario(scenario_id)

        # Phase 1: Document preparation
        logger.info("Phase 1: Document Preparation")
        doc_context = self.doc_preparer.prepare_scenario(config)

        # Create scenario instance
        scenario = create_scenario(config, doc_context)

        # Phase 2: Conversation generation
        logger.info("Phase 2: Conversation Generation")
        conversations, summaries = self.conv_generator.generate_scenario(
            scenario, doc_context
        )

        # Phase 3: Annotation
        logger.info("Phase 3: Annotation")
        memories, conflicts, eval_questions = self.annotation_pipeline.annotate_scenario(
            scenario, conversations, summaries, doc_context
        )

        # Build scenario output
        scenario_output = ScenarioOutput(
            scenario_id=scenario_id,
            config=config,
            conversations=conversations,
            session_summaries=summaries,
            memories=memories,
            conflicts=conflicts,
            eval_questions=eval_questions,
            validation_report=ValidationReport(scenario_id=scenario_id),
        )

        # Phase 4: Validation
        logger.info("Phase 4: Validation")
        validation_report = self.validation_pipeline.validate_scenario(scenario_output)
        scenario_output.validation_report = validation_report

        # Save outputs
        self._save_scenario_output(scenario_output)

        elapsed = time.time() - start_time
        logger.info(
            f"=== Scenario {scenario_id} complete in {elapsed:.1f}s ===\n"
            f"{self.cost_tracker.summary()}"
        )
        return scenario_output

    def run_all(self) -> BenchmarkDataset:
        start_time = time.time()
        logger.info("=== Running all scenarios ===")

        scenario_outputs = []
        for sid in ["1", "2", "3", "4", "5"]:
            try:
                output = self.run_scenario(sid)
                scenario_outputs.append(output)
            except Exception as e:
                logger.error(f"Failed scenario {sid}: {e}")

        # Build aggregate stats
        total_conversations = sum(len(s.conversations) for s in scenario_outputs)
        total_memories = sum(len(s.memories) for s in scenario_outputs)
        total_conflicts = sum(len(s.conflicts) for s in scenario_outputs)
        total_questions = sum(len(s.eval_questions) for s in scenario_outputs)

        dataset = BenchmarkDataset(
            version="0.1.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
            scenarios=scenario_outputs,
            aggregate_stats={
                "total_scenarios": len(scenario_outputs),
                "total_conversations": total_conversations,
                "total_memories": total_memories,
                "total_conflicts": total_conflicts,
                "total_eval_questions": total_questions,
            },
            generation_report=GenerationReport(
                total_cost=self.cost_tracker.get_total_cost(),
                total_tokens=self.cost_tracker.get_total_tokens(),
                timing={"total_seconds": time.time() - start_time},
                per_phase_breakdown=self.cost_tracker.get_phase_breakdown(),
            ),
        )

        self._save_benchmark(dataset)
        return dataset

    def run_conversations_only(self, scenario_id: str) -> None:
        """Phase 2 only — generate conversations for a scenario."""
        config = load_scenario(scenario_id)
        doc_context = self.doc_preparer.prepare_scenario(config)
        scenario = create_scenario(config, doc_context)
        conversations, summaries = self.conv_generator.generate_scenario(
            scenario, doc_context
        )
        self._save_phase_output(scenario_id, "conversations", conversations)
        self._save_phase_output(scenario_id, "summaries", summaries)

    def run_annotation_only(self, scenario_id: str) -> None:
        """Phase 3 only — requires existing conversations."""
        from src.utils.io import read_json

        config = load_scenario(scenario_id)
        doc_context = self.doc_preparer.prepare_scenario(config)
        scenario = create_scenario(config, doc_context)

        # Load existing conversations and summaries
        conv_path = self.output_dir / f"scenario_{scenario_id}" / "conversations"
        summ_path = self.output_dir / f"scenario_{scenario_id}" / "summaries"

        # This would need to deserialize — simplified here
        logger.info("Loading existing conversations for annotation...")
        # In practice, load from saved JSON files
        raise NotImplementedError(
            "Run full pipeline or implement conversation loading from output files"
        )

    def run_validation_only(self, scenario_id: str) -> None:
        """Phase 4 only — requires existing annotations."""
        raise NotImplementedError(
            "Run full pipeline or implement scenario output loading from files"
        )

    def _save_scenario_output(self, output: ScenarioOutput) -> None:
        scenario_dir = self.output_dir / f"scenario_{output.scenario_id}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Save conversations
        conv_dir = scenario_dir / "conversations"
        conv_dir.mkdir(exist_ok=True)
        for conv in output.conversations:
            write_json(
                conv.model_dump(),
                conv_dir / f"{conv.user_id}_session_{conv.session_number}.json",
            )

        # Save summaries
        write_json(
            [s.model_dump() for s in output.session_summaries],
            scenario_dir / "summaries" / "session_summaries.json",
        )

        # Save memories
        write_json(
            [m.model_dump() for m in output.memories],
            scenario_dir / "memories" / "extracted_memories.json",
        )

        # Save conflicts
        write_json(
            [c.model_dump() for c in output.conflicts],
            scenario_dir / "conflicts" / "conflict_annotations.json",
        )

        # Save eval questions
        write_json(
            [q.model_dump() for q in output.eval_questions],
            scenario_dir / "evaluation" / "eval_questions.json",
        )

        # Save validation report
        write_json(
            output.validation_report.model_dump(),
            scenario_dir / "validation" / "validation_report.json",
        )

        # Save complete bundle
        write_json(
            output.model_dump(),
            scenario_dir / f"scenario_{output.scenario_id}_complete.json",
        )

        logger.info(f"Saved scenario {output.scenario_id} to {scenario_dir}")

    def _save_phase_output(self, scenario_id: str, phase: str, data: list) -> None:
        output_dir = self.output_dir / f"scenario_{scenario_id}" / phase
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            [item.model_dump() for item in data],
            output_dir / f"{phase}.json",
        )

    def _save_benchmark(self, dataset: BenchmarkDataset) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(dataset.model_dump(), self.output_dir / "benchmark.json")
        write_json(
            dataset.generation_report.model_dump(),
            self.output_dir / "generation_report.json",
        )
        logger.info(f"Saved benchmark to {self.output_dir / 'benchmark.json'}")
