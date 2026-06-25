from __future__ import annotations

import re
from dataclasses import dataclass

from msa_zria.config import KGConfig, KGScope
from msa_zria.data import ParseTarget, Triple
from msa_zria.kg import retrieve_neighborhood


_TOKEN_RE = re.compile(r"[a-z0-9_:/.-]+")


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    score: float


class EvidenceRetriever:
    def retrieve(
        self,
        query: str,
        *,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> list[EvidenceSnippet]:
        raise NotImplementedError


class KGEvidenceRetriever(EvidenceRetriever):
    def __init__(
        self,
        kg_config: KGConfig,
        *,
        top_k: int = 5,
        candidate_limit: int = 32,
        min_score: float = 0.5,
    ) -> None:
        self.kg_config = kg_config
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> list[EvidenceSnippet]:
        scoped_kg = self.kg_config.model_copy(
            update={
                "workspace": kg_scope.workspace if kg_scope and kg_scope.workspace is not None else self.kg_config.workspace,
                "branch": kg_scope.branch if kg_scope and kg_scope.branch is not None else self.kg_config.branch,
                "commit": kg_scope.commit if kg_scope and kg_scope.commit is not None else self.kg_config.commit,
                "as_of": kg_scope.as_of if kg_scope and kg_scope.as_of is not None else self.kg_config.as_of,
            }
        )
        triples = retrieve_neighborhood(
            scoped_kg,
            query,
            parsed,
            limit=max(self.top_k, self.candidate_limit),
        )
        if not triples:
            return []

        ranked: list[EvidenceSnippet] = []
        for triple in triples:
            score = _score_triple(triple, query, parsed)
            if score < self.min_score:
                continue
            ranked.append(EvidenceSnippet(text=triple.as_sentence(), score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.top_k]


def render_evidence_context(snippets: list[EvidenceSnippet]) -> str | None:
    if not snippets:
        return None
    lines = ["Retrieved evidence:"]
    for index, snippet in enumerate(snippets, start=1):
        lines.append(f"{index}. {snippet.text}")
    return "\n".join(lines)


def _score_triple(triple: Triple, query: str, parsed: ParseTarget | None) -> float:
    query_tokens = set(_tokenize(query))
    triple_text = f"{triple.subject} {triple.predicate} {triple.object}"
    triple_tokens = set(_tokenize(triple_text))
    overlap = len(query_tokens & triple_tokens)
    score = float(overlap)

    if parsed is None:
        return score

    for value, weight in (
        (parsed.device, 3.0),
        (parsed.issue, 3.0),
        (parsed.cause, 2.0),
        (parsed.severity, 1.5),
    ):
        if value and _contains_phrase(triple_text, value):
            score += weight
    return score


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _contains_phrase(haystack: str, needle: str) -> bool:
    normalized_haystack = haystack.lower()
    normalized_needle = needle.strip().lower()
    if not normalized_needle:
        return False
    return normalized_needle in normalized_haystack
