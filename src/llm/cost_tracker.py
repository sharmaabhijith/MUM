from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("mum")

# Prices per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    phase: str
    cost: float


@dataclass
class CostTracker:
    records: list[CallRecord] = field(default_factory=list)

    def record_call(
        self, model: str, input_tokens: int, output_tokens: int, phase: str
    ) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        record = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            phase=phase,
            cost=cost,
        )
        self.records.append(record)
        return cost

    def get_total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    def get_total_tokens(self) -> dict[str, int]:
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        return {"input": total_input, "output": total_output, "total": total_input + total_output}

    def get_phase_breakdown(self) -> dict[str, dict]:
        phases: dict[str, dict] = {}
        for r in self.records:
            if r.phase not in phases:
                phases[r.phase] = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            phases[r.phase]["cost"] += r.cost
            phases[r.phase]["input_tokens"] += r.input_tokens
            phases[r.phase]["output_tokens"] += r.output_tokens
            phases[r.phase]["calls"] += 1
        return phases

    def summary(self) -> str:
        total = self.get_total_cost()
        tokens = self.get_total_tokens()
        phases = self.get_phase_breakdown()
        lines = [
            f"Total cost: ${total:.4f}",
            f"Total tokens: {tokens['total']:,} (input: {tokens['input']:,}, output: {tokens['output']:,})",
            f"Total API calls: {len(self.records)}",
            "",
            "Per-phase breakdown:",
        ]
        for phase, data in sorted(phases.items()):
            lines.append(
                f"  {phase}: ${data['cost']:.4f} | {data['calls']} calls | "
                f"{data['input_tokens']:,} in / {data['output_tokens']:,} out"
            )
        return "\n".join(lines)

    def to_report_dict(self) -> dict:
        return {
            "total_cost": self.get_total_cost(),
            "total_tokens": self.get_total_tokens(),
            "per_phase_breakdown": self.get_phase_breakdown(),
        }
