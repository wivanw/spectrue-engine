from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Literal, TYPE_CHECKING
from spectrue_core.domain.verification.verdict.model import AnalysisMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineMode:
    """
    Frozen configuration for a pipeline mode.
    """
    name: Literal["general", "deep", "deep_v2"]
    allow_batch: bool
    allow_clustering: bool
    require_single_language: bool
    require_metering: bool
    max_claims_for_scoring: int
    search_depth: Literal["basic", "advanced"]

    def __str__(self) -> str:
        return f"PipelineMode({self.name})"

    def __repr__(self) -> str:
        return (
            f"PipelineMode(name={self.name!r}, allow_batch={self.allow_batch}, "
            f"allow_clustering={self.allow_clustering}, "
            f"require_single_language={self.require_single_language}, "
            f"max_claims={self.max_claims_for_scoring}, "
            f"search_depth={self.search_depth!r})"
        )

    @property
    def api_analysis_mode(self) -> AnalysisMode:
        """Get API-facing analysis mode name.
        
        Maps internal mode name to frontend-compatible AnalysisMode enum.
        Use this for all API responses instead of raw mode.name.
        """
        try:
            return AnalysisMode(self.name)
        except ValueError:
            return AnalysisMode.GENERAL


@dataclass
class PipelineContext:
    """
    Immutable context passed through pipeline steps.
    """
    mode: PipelineMode
    claims: list[dict[str, Any]] = field(default_factory=list)
    lang: str = "en"
    trace: Any | None = None  # Using Any to avoid circular import with Trace
    sources: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    progress_callback: Any | None = None

    def with_update(self, **kwargs: Any) -> PipelineContext:
        current = {
            "mode": self.mode,
            "claims": self.claims,
            "lang": self.lang,
            "trace": self.trace,
            "sources": self.sources,
            "evidence": self.evidence,
            "verdict": self.verdict,
            "extras": self.extras,
            "progress_callback": self.progress_callback,
        }
        current.update(kwargs)
        return PipelineContext(**current)

    def set_extra(self, key: str, value: Any) -> PipelineContext:
        new_extras = {**self.extras, key: value}
        return self.with_update(extras=new_extras)

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)


@runtime_checkable
class Step(Protocol):
    """
    Protocol for pipeline steps.
    """
    name: str
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ...
