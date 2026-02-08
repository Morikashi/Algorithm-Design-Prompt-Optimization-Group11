from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Prompt:
    """
    Structured prompt representation for prompt optimization.
    Deterministic rendering is critical for hashing, dedup, and reproducibility.
    """
    task: str  # "qa" or "summarization"
    instruction: str
    constraints: List[str] = field(default_factory=list)
    output_format: Optional[str] = None  # e.g., "plain", "json", "bullets"
    style: Optional[str] = None  # e.g., "concise", "formal", "step_by_step"
    verification: bool = False  # whether to include a "double-check" step

    def render(self) -> str:
        """Deterministically render to a single text prompt."""
        # Important: stable ordering and consistent separators.
        parts: List[str] = []

        # Task tag (useful for analysis & debugging)
        parts.append(f"[TASK: {self.task}]")

        # Core instruction
        parts.append(self.instruction.strip())

        # Style (optional)
        if self.style:
            parts.append(f"Style: {self.style}")

        # Constraints (stable order)
        if self.constraints:
            parts.append("Constraints:")
            for c in self.constraints:
                parts.append(f"- {c.strip()}")

        # Output format (optional)
        if self.output_format:
            parts.append(f"Output format: {self.output_format}")

        # Verification step (optional)
        if self.verification:
            parts.append("Before finalizing, double-check the answer for correctness and format compliance.")

        return "\n".join(parts).strip()
