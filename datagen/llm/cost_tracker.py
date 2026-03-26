from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("mum")

# Prices per 1M tokens (USD) — DeepInfra pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    # DeepSeek
    "deepseek-ai/DeepSeek-V3.2": {"input": 0.30, "output": 0.88},
    "deepseek-ai/DeepSeek-V3.1": {"input": 0.30, "output": 0.88},
    "deepseek-ai/DeepSeek-V3": {"input": 0.30, "output": 0.88},
    "deepseek-ai/DeepSeek-R1": {"input": 0.75, "output": 2.19},
    # Google Gemini
    "google/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "google/gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "google/gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Qwen3
    "Qwen/Qwen3-14B": {"input": 0.10, "output": 0.30},
    "Qwen/Qwen3-32B": {"input": 0.15, "output": 0.45},
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {"input": 0.18, "output": 0.54},
    # Meta Llama 3.1
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"input": 0.03, "output": 0.05},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"input": 0.23, "output": 0.40},
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
