from __future__ import annotations

from pathlib import Path

from src.models.enums import RelationshipType
from src.models.schemas import ScenarioConfig
from src.pipeline.phase1_document_prep import DocumentContext
from src.scenarios.base import BaseScenario
from src.scenarios.scenario_1 import StudyGroupScenario
from src.scenarios.scenario_2 import ExecTeamScenario
from src.scenarios.scenario_3 import ContractReviewScenario
from src.scenarios.scenario_4 import NegotiationScenario
from src.scenarios.scenario_5 import SupportScenario
from src.utils.io import read_yaml

CONFIG_DIR = Path("config/scenarios")

SCENARIO_CLASSES: dict[str, type[BaseScenario]] = {
    RelationshipType.SYMMETRIC: StudyGroupScenario,
    RelationshipType.HIERARCHICAL: ExecTeamScenario,
    RelationshipType.CROSS_FUNCTIONAL: ContractReviewScenario,
    RelationshipType.ADVERSARIAL: NegotiationScenario,
    RelationshipType.SEQUENTIAL: SupportScenario,
}


def load_scenario(scenario_id: str) -> ScenarioConfig:
    config_path = CONFIG_DIR / f"scenario_{scenario_id}.yaml"
    raw = read_yaml(config_path)
    return ScenarioConfig.model_validate(raw)


def create_scenario(
    config: ScenarioConfig, doc_context: DocumentContext | None = None
) -> BaseScenario:
    cls = SCENARIO_CLASSES.get(config.relationship_type)
    if cls is None:
        raise ValueError(f"Unknown relationship type: {config.relationship_type}")
    return cls(config=config, doc_context=doc_context)


__all__ = [
    "BaseScenario",
    "StudyGroupScenario",
    "ExecTeamScenario",
    "ContractReviewScenario",
    "NegotiationScenario",
    "SupportScenario",
    "load_scenario",
    "create_scenario",
]
