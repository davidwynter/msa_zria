from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from msa_zria.config import KGScope, ZRIAConfig
from msa_zria.data import EvaluationTarget, ParseTarget
from msa_zria.zria_backend import BaseZRIABackend, RuleBasedZRIABackend, ZRIAPredictionTrace, load_zria_backend


class BaseZRIAAdapter(ABC):
    @abstractmethod
    def predict(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> EvaluationTarget:
        raise NotImplementedError

    @abstractmethod
    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> ZRIAPredictionTrace:
        raise NotImplementedError


class ConfiguredZRIAAdapter(BaseZRIAAdapter):
    def __init__(self, backend: BaseZRIABackend) -> None:
        self.backend = backend

    @classmethod
    def from_config(cls, config: ZRIAConfig) -> "ConfiguredZRIAAdapter":
        return cls(load_zria_backend(config))

    def predict(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> EvaluationTarget:
        return self.backend.predict(query, parsed=parsed, kg_scope=kg_scope)

    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> ZRIAPredictionTrace:
        return self.backend.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope)


class RuleBasedZRIAAdapter(ConfiguredZRIAAdapter):
    @classmethod
    def from_rules_path(cls, path: str | Path) -> "RuleBasedZRIAAdapter":
        return cls(RuleBasedZRIABackend.from_path(path))


class HeuristicZRIAAdapter(RuleBasedZRIAAdapter):
    def __init__(self) -> None:
        super().__init__(
            RuleBasedZRIABackend.from_path(
                Path(__file__).resolve().parents[2] / "examples" / "zria_rules.json"
            )
        )
