from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder, event_report_id, sha256_directory, sha256_json, stable_dataset_version
from msa_zria.config import KGScope
from msa_zria.data import EvaluationCase, EvaluationResult, EvaluationTarget, ParseTarget, Triple, evaluate_case

if TYPE_CHECKING:
    import torch

_PAD = "<pad>"
_UNK = "<unk>"
_REL_PAD = "<rel_pad>"
_GRAPH_NODE_TOKEN_LENGTH = 12
_GRAPH_ROLE_GENERAL = 0
_GRAPH_ROLE_DEVICE = 1
_GRAPH_ROLE_ISSUE = 2
_GRAPH_ROLE_CAUSE = 3
_GRAPH_ROLE_SEVERITY = 4
_GRAPH_ROLE_ANCHOR = 5
_GRAPH_BATCH_FIELDS = (
    "query_input_ids",
    "query_attention_mask",
    "node_input_ids",
    "node_attention_mask",
    "node_role_ids",
    "node_mask",
    "edge_index",
    "edge_type_ids",
    "edge_mask",
    "spectral_positions",
    "anchor_mask",
)


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The learned ZRIA backend requires torch to be installed."
        ) from exc
    return torch, nn, DataLoader


class ZRIAExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    query: str
    parsed: ParseTarget
    target: EvaluationTarget
    kg_scope: KGScope | None = None
    neighborhood: list[Triple] = Field(default_factory=list)


class ZRIALearnedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    backend_type: Literal["learned", "learned_graph"] = "learned"
    vocab: dict[str, int]
    labels: list[EvaluationTarget]
    max_length: int
    embedding_dim: int
    spectral_k: int = 0
    hidden_dim: int
    dropout: float = 0.0
    graph_feature_dim: int = 0
    relation_vocab: dict[str, int] = Field(default_factory=dict)
    graph_node_token_length: int = _GRAPH_NODE_TOKEN_LENGTH
    graph_max_nodes: int = 0
    graph_max_edges: int = 0
    graph_num_layers: int = 2
    harmonic_reg_weight: float = 5e-4
    calibration_temperature: float = 1.0
    calibration_source: Literal["none", "train", "eval"] = "none"
    self_supervised_tasks: list[str] = Field(default_factory=list)
    memory_enabled: bool = True
    memory_momentum: float = 0.9
    memory_vectors: list[list[float]] = Field(default_factory=list)
    memory_counts: list[float] = Field(default_factory=list)
    state_dict_path: str


class ZRIAGraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    score: float


class ZRIAComparisonDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    backend: str
    confidence: float | None = None
    predicted: EvaluationTarget
    evaluation: EvaluationResult


class ZRIAComparisonSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    total_cases: int
    passed_cases: int
    accuracy: float
    average_score: float


class ZRIAComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summaries: list[ZRIAComparisonSummary]
    details: list[ZRIAComparisonDetail]


class ZRIASweepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spectral_k: int
    dropout: float
    artifact_path: str
    eval_accuracy: float
    average_score: float
    epochs_trained: int
    best_epoch: int
    early_stopped: bool


class ZRIASweepReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ZRIASweepRecord]
    best_record: ZRIASweepRecord


def load_zria_examples(path: str | Path) -> list[ZRIAExample]:
    examples_path = Path(path)
    examples: list[ZRIAExample] = []
    with examples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(ZRIAExample.model_validate_json(line))
    return examples


def render_feature_text(
    query: str,
    parsed: ParseTarget,
    kg_scope: KGScope | None = None,
) -> str:
    parts = [
        f"query {query}",
        f"device {parsed.device}",
        f"issue {parsed.issue}",
    ]
    if parsed.cause:
        parts.append(f"cause {parsed.cause}")
    if parsed.severity:
        parts.append(f"severity {parsed.severity}")
    if kg_scope and kg_scope.workspace:
        parts.append(f"workspace {kg_scope.workspace}")
    if kg_scope and kg_scope.branch:
        parts.append(f"branch {kg_scope.branch}")
    if kg_scope and kg_scope.commit:
        parts.append(f"commit {kg_scope.commit}")
    return " ".join(parts)


def render_neighborhood_text(neighborhood: list[Triple]) -> str:
    return " ".join(
        f"subject {triple.subject} predicate {triple.predicate} object {triple.object}"
        for triple in neighborhood
    )


def render_graph_feature_text(
    query: str,
    parsed: ParseTarget,
    neighborhood: list[Triple],
    kg_scope: KGScope | None = None,
) -> str:
    parts = [render_feature_text(query, parsed, kg_scope)]
    neighborhood_text = render_neighborhood_text(neighborhood)
    if neighborhood_text:
        parts.append(neighborhood_text)
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_:/.-]+", text.lower())


def build_vocab(examples: list[ZRIAExample], *, include_neighborhood: bool = False) -> dict[str, int]:
    vocab = {_PAD: 0, _UNK: 1}
    for example in examples:
        feature_text = render_feature_text(example.query, example.parsed, example.kg_scope)
        if include_neighborhood:
            feature_text = render_graph_feature_text(
                example.query,
                example.parsed,
                example.neighborhood,
                example.kg_scope,
            )
        for token in tokenize(feature_text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_label_set(examples: list[ZRIAExample]) -> list[EvaluationTarget]:
    labels: list[EvaluationTarget] = []
    seen: set[str] = set()
    for example in examples:
        key = json.dumps(example.target.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        labels.append(example.target)
    return labels


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab[_UNK]) for token in tokenize(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[_PAD]] * (max_length - len(token_ids)))
    return token_ids


def label_index(target: EvaluationTarget, labels: list[EvaluationTarget]) -> int:
    serialized = _label_key(target)
    for index, label in enumerate(labels):
        if serialized == _label_key(label):
            return index
    raise ValueError("Encountered target that is not present in the label set.")


def _label_key(target: EvaluationTarget) -> str:
    return json.dumps(target.model_dump(), sort_keys=True)


def graph_feature_dim(spectral_k: int) -> int:
    return max(1, spectral_k)


def _relation_key(predicate: str, *, reverse: bool = False) -> str:
    return f"{predicate}#reverse" if reverse else predicate


def build_relation_vocab(examples: list[ZRIAExample]) -> dict[str, int]:
    vocab = {_REL_PAD: 0}
    for example in examples:
        for triple in example.neighborhood:
            for relation in (_relation_key(triple.predicate), _relation_key(triple.predicate, reverse=True)):
                if relation not in vocab:
                    vocab[relation] = len(vocab)
    return vocab


def _graph_shape_limits(examples: list[ZRIAExample]) -> tuple[int, int]:
    max_nodes = 2
    max_edges = 2
    for example in examples:
        node_labels = _graph_node_labels(example.neighborhood, example.parsed)
        max_nodes = max(max_nodes, len(node_labels))
        max_edges = max(max_edges, max(2, len(example.neighborhood) * 2))
    return max_nodes, max_edges


def _graph_node_labels(neighborhood: list[Triple], parsed: ParseTarget) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if value is None:
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        labels.append(normalized)

    for triple in neighborhood:
        add(triple.subject)
        add(triple.object)
    add(parsed.device)
    add(parsed.issue)
    add(parsed.cause)
    add(parsed.severity)
    return labels or [parsed.device, parsed.issue]


def _graph_role(node_label: str, parsed: ParseTarget) -> int:
    normalized_node = node_label.strip().lower()

    def matches(value: str | None) -> bool:
        if value is None:
            return False
        normalized = value.strip().lower()
        return normalized == normalized_node or normalized in normalized_node or normalized_node in normalized

    if matches(parsed.device):
        return _GRAPH_ROLE_DEVICE
    if matches(parsed.issue):
        return _GRAPH_ROLE_ISSUE
    if matches(parsed.cause):
        return _GRAPH_ROLE_CAUSE
    if matches(parsed.severity):
        return _GRAPH_ROLE_SEVERITY
    return _GRAPH_ROLE_GENERAL


def _spectral_positions(
    node_count: int,
    undirected_edges: list[tuple[int, int]],
    spectral_k: int,
) -> list[list[float]]:
    if spectral_k <= 0:
        return [[] for _ in range(node_count)]
    if node_count <= 0:
        return []

    torch, _, _ = _require_torch()
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32)
    for source, target in undirected_edges:
        if source >= node_count or target >= node_count:
            continue
        adjacency[source, target] += 1.0
        adjacency[target, source] += 1.0
    degrees = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.where(
        degrees > 0,
        degrees.rsqrt(),
        torch.zeros_like(degrees),
    )
    normalized = inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)
    laplacian = torch.eye(node_count, dtype=torch.float32) - normalized
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    del eigenvalues
    positions = eigenvectors[:, : min(spectral_k, eigenvectors.size(1))].real
    if positions.size(1) < spectral_k:
        padding = torch.zeros((node_count, spectral_k - positions.size(1)), dtype=torch.float32)
        positions = torch.cat([positions, padding], dim=1)
    return positions.tolist()


def build_graph_features(
    neighborhood: list[Triple],
    parsed: ParseTarget,
    vocab: dict[str, int],
    relation_vocab: dict[str, int],
    spectral_k: int,
    max_nodes: int,
    max_edges: int,
    node_token_length: int = _GRAPH_NODE_TOKEN_LENGTH,
    *,
    include_metadata: bool = False,
) -> dict[str, Any]:
    node_labels = _graph_node_labels(neighborhood, parsed)[:max_nodes]
    node_lookup = {label: index for index, label in enumerate(node_labels)}
    max_forward_edges = max(1, max_edges // 2)
    forward_edges: list[Triple] = []
    undirected_edges: list[tuple[int, int]] = []
    for triple in neighborhood:
        if len(forward_edges) >= max_forward_edges:
            break
        if triple.subject not in node_lookup or triple.object not in node_lookup:
            continue
        forward_edges.append(triple)
        undirected_edges.append((node_lookup[triple.subject], node_lookup[triple.object]))

    spectral_positions = _spectral_positions(len(node_labels), undirected_edges, spectral_k)
    node_input_ids: list[list[int]] = []
    node_attention_mask: list[list[float]] = []
    node_role_ids: list[int] = []
    node_mask: list[float] = []
    anchor_mask: list[float] = []
    padded_spectral: list[list[float]] = []

    for index, label in enumerate(node_labels):
        token_ids = encode_text(label, vocab, node_token_length)
        node_input_ids.append(token_ids)
        node_attention_mask.append([1.0 if token_id != vocab[_PAD] else 0.0 for token_id in token_ids])
        role_id = _graph_role(label, parsed)
        node_role_ids.append(role_id)
        node_mask.append(1.0)
        anchor_mask.append(1.0 if role_id != _GRAPH_ROLE_GENERAL else 0.0)
        padded_spectral.append(spectral_positions[index] if index < len(spectral_positions) else [0.0] * spectral_k)

    while len(node_input_ids) < max_nodes:
        node_input_ids.append([vocab[_PAD]] * node_token_length)
        node_attention_mask.append([0.0] * node_token_length)
        node_role_ids.append(_GRAPH_ROLE_GENERAL)
        node_mask.append(0.0)
        anchor_mask.append(0.0)
        padded_spectral.append([0.0] * spectral_k)

    edge_index: list[list[int]] = []
    edge_type_ids: list[int] = []
    edge_mask: list[float] = []
    for triple in forward_edges:
        source_idx = node_lookup[triple.subject]
        target_idx = node_lookup[triple.object]
        edge_index.append([source_idx, target_idx])
        edge_type_ids.append(relation_vocab.get(_relation_key(triple.predicate), relation_vocab[_REL_PAD]))
        edge_mask.append(1.0)

    reverse_edges = list(forward_edges)
    for triple in reverse_edges:
        if len(edge_index) >= max_edges:
            break
        source_idx = node_lookup[triple.object]
        target_idx = node_lookup[triple.subject]
        edge_index.append([source_idx, target_idx])
        edge_type_ids.append(relation_vocab.get(_relation_key(triple.predicate, reverse=True), relation_vocab[_REL_PAD]))
        edge_mask.append(1.0)

    while len(edge_index) < max_edges:
        edge_index.append([0, 0])
        edge_type_ids.append(relation_vocab[_REL_PAD])
        edge_mask.append(0.0)

    payload = {
        "node_input_ids": node_input_ids,
        "node_attention_mask": node_attention_mask,
        "node_role_ids": node_role_ids,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_mask": edge_mask,
        "spectral_positions": padded_spectral,
        "anchor_mask": anchor_mask,
        "forward_edge_count": len(forward_edges),
    }
    if include_metadata:
        payload["forward_edges"] = forward_edges
    return payload


class _ZRIADataset:
    def __init__(
        self,
        examples: list[ZRIAExample],
        vocab: dict[str, int],
        labels: list[EvaluationTarget],
        max_length: int,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _, _ = _require_torch()
        example = self.examples[index]
        text = render_feature_text(example.query, example.parsed, example.kg_scope)
        token_ids = encode_text(text, self.vocab, self.max_length)
        attention_mask = [1 if token_id != self.vocab[_PAD] else 0 for token_id in token_ids]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "label": torch.tensor(label_index(example.target, self.labels), dtype=torch.long),
        }


class _ZRIAGraphDataset:
    def __init__(
        self,
        examples: list[ZRIAExample],
        vocab: dict[str, int],
        relation_vocab: dict[str, int],
        labels: list[EvaluationTarget],
        max_length: int,
        spectral_k: int,
        max_graph_nodes: int,
        max_graph_edges: int,
        node_token_length: int = _GRAPH_NODE_TOKEN_LENGTH,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.relation_vocab = relation_vocab
        self.labels = labels
        self.max_length = max_length
        self.spectral_k = spectral_k
        self.max_graph_nodes = max_graph_nodes
        self.max_graph_edges = max_graph_edges
        self.node_token_length = node_token_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _, _ = _require_torch()
        example = self.examples[index]
        query_text = render_feature_text(example.query, example.parsed, example.kg_scope)
        token_ids = encode_text(query_text, self.vocab, self.max_length)
        attention_mask = [1.0 if token_id != self.vocab[_PAD] else 0.0 for token_id in token_ids]
        graph_features = build_graph_features(
            example.neighborhood,
            example.parsed,
            self.vocab,
            self.relation_vocab,
            self.spectral_k,
            self.max_graph_nodes,
            self.max_graph_edges,
            self.node_token_length,
        )
        return {
            "query_input_ids": torch.tensor(token_ids, dtype=torch.long),
            "query_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "node_input_ids": torch.tensor(graph_features["node_input_ids"], dtype=torch.long),
            "node_attention_mask": torch.tensor(graph_features["node_attention_mask"], dtype=torch.float32),
            "node_role_ids": torch.tensor(graph_features["node_role_ids"], dtype=torch.long),
            "node_mask": torch.tensor(graph_features["node_mask"], dtype=torch.float32),
            "edge_index": torch.tensor(graph_features["edge_index"], dtype=torch.long),
            "edge_type_ids": torch.tensor(graph_features["edge_type_ids"], dtype=torch.long),
            "edge_mask": torch.tensor(graph_features["edge_mask"], dtype=torch.float32),
            "spectral_positions": torch.tensor(graph_features["spectral_positions"], dtype=torch.float32),
            "anchor_mask": torch.tensor(graph_features["anchor_mask"], dtype=torch.float32),
            "label": torch.tensor(label_index(example.target, self.labels), dtype=torch.long),
        }


def create_model(
    vocab: dict[str, int],
    labels: list[EvaluationTarget],
    embedding_dim: int,
    spectral_k: int,
    hidden_dim: int,
    dropout: float = 0.0,
) -> Any:
    torch, nn, _ = _require_torch()

    class ZRIALearnedClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.spectral_k = spectral_k
            self.hidden = nn.Linear(embedding_dim + spectral_k, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim, len(labels))
            self.register_buffer("label_memory", torch.zeros(len(labels), hidden_dim))
            self.register_buffer("label_memory_counts", torch.zeros(len(labels)))

        def encode(self, input_ids: Any, attention_mask: Any) -> Any:
            embeddings = self.embedding(input_ids)
            masked = embeddings * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            if self.spectral_k > 0:
                signal = masked.mean(dim=2)
                frequencies = torch.fft.rfft(signal, dim=1).abs()
                if frequencies.size(1) >= self.spectral_k:
                    spectral = frequencies[:, : self.spectral_k]
                else:
                    spectral = torch.zeros(
                        frequencies.size(0),
                        self.spectral_k,
                        dtype=frequencies.dtype,
                        device=frequencies.device,
                    )
                    spectral[:, : frequencies.size(1)] = frequencies
                pooled = torch.cat([pooled, spectral], dim=1)
            hidden = torch.relu(self.hidden(pooled))
            return self.dropout(hidden)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.encode(input_ids, attention_mask)
            logits = self.output(hidden)
            return _apply_memory_logits(hidden, logits, self.label_memory, self.label_memory_counts)

    return ZRIALearnedClassifier()


def create_graph_model(
    vocab: dict[str, int],
    relation_vocab: dict[str, int],
    labels: list[EvaluationTarget],
    embedding_dim: int,
    spectral_k: int,
    hidden_dim: int,
    dropout: float = 0.0,
    *,
    graph_num_layers: int = 2,
    harmonic_reg_weight: float = 5e-4,
) -> Any:
    torch, nn, _ = _require_torch()
    num_relations = max(1, len(relation_vocab))
    num_bases = min(4, num_relations)

    class _RGCNLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_loop = nn.Linear(hidden_dim, hidden_dim)
            self.bases = nn.Parameter(torch.empty(num_bases, hidden_dim, hidden_dim))
            self.coefficients = nn.Parameter(torch.empty(num_relations, num_bases))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.coefficients)

        def _weights(self) -> Any:
            return torch.einsum("rb,bij->rij", self.coefficients, self.bases)

        def forward(
            self,
            node_hidden: Any,
            edge_index: Any,
            edge_type_ids: Any,
            edge_mask: Any,
            node_mask: Any,
        ) -> tuple[Any, Any]:
            relation_weights = self._weights()
            batch_outputs: list[Any] = []
            batch_scores: list[Any] = []
            for batch_index in range(node_hidden.size(0)):
                node_count = int(node_mask[batch_index].sum().item())
                current_hidden = node_hidden[batch_index, :node_count]
                padded_hidden = torch.zeros_like(node_hidden[batch_index])
                padded_scores = torch.zeros(edge_mask.size(1), dtype=node_hidden.dtype, device=node_hidden.device)
                if node_count == 0:
                    batch_outputs.append(padded_hidden)
                    batch_scores.append(padded_scores)
                    continue

                output_hidden = self.self_loop(current_hidden)
                valid_edges = edge_mask[batch_index].bool()
                if bool(valid_edges.any()):
                    current_edges = edge_index[batch_index, valid_edges].long()
                    current_relations = edge_type_ids[batch_index, valid_edges].long()
                    source_hidden = current_hidden[current_edges[:, 0]]
                    current_weights = relation_weights[current_relations]
                    messages = torch.bmm(source_hidden.unsqueeze(1), current_weights).squeeze(1)
                    destinations = current_edges[:, 1]
                    aggregated = torch.zeros_like(current_hidden)
                    aggregated.index_add_(0, destinations, messages)
                    degrees = torch.zeros(node_count, dtype=node_hidden.dtype, device=node_hidden.device)
                    degrees.index_add_(
                        0,
                        destinations,
                        torch.ones(destinations.size(0), dtype=node_hidden.dtype, device=node_hidden.device),
                    )
                    output_hidden = output_hidden + aggregated / degrees.clamp_min(1.0).unsqueeze(-1)
                    padded_scores[valid_edges] = messages.norm(dim=1)

                output_hidden = torch.relu(output_hidden + self.bias)
                padded_hidden[:node_count] = output_hidden
                batch_outputs.append(padded_hidden)
                batch_scores.append(padded_scores)
            return torch.stack(batch_outputs, dim=0), torch.stack(batch_scores, dim=0)

    class ZRIAGraphClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.node_role_embedding = nn.Embedding(_GRAPH_ROLE_ANCHOR + 1, hidden_dim)
            self.query_projection = nn.Linear(embedding_dim, hidden_dim)
            if spectral_k > 0:
                self.node_projection = nn.Linear(embedding_dim + spectral_k, hidden_dim)
                self.harmonic_projection = nn.Linear(embedding_dim, hidden_dim)
                self.harmonic_scale = nn.Parameter(torch.ones(spectral_k))
            else:
                self.node_projection = nn.Linear(embedding_dim, hidden_dim)
                self.harmonic_projection = None
                self.harmonic_scale = None
            self.layers = nn.ModuleList(_RGCNLayer() for _ in range(max(1, graph_num_layers)))
            self.dropout = nn.Dropout(dropout)
            self.fusion_projection = nn.Linear(hidden_dim * 3, hidden_dim)
            self.fusion_gate = nn.Linear(hidden_dim * 3, hidden_dim)
            self.output = nn.Linear(hidden_dim, len(labels))
            self.relation_embedding = nn.Embedding(num_relations, hidden_dim, padding_idx=0)
            self.relation_decoder = nn.Linear(hidden_dim * 4, num_relations)
            self.edge_decoder = nn.Linear(hidden_dim * 4, 1)
            self.harmonic_reg_weight = harmonic_reg_weight
            self.register_buffer("label_memory", torch.zeros(len(labels), hidden_dim))
            self.register_buffer("label_memory_counts", torch.zeros(len(labels)))

        def _token_mean(self, embeddings: Any, attention_mask: Any) -> Any:
            weighted = embeddings * attention_mask.unsqueeze(-1)
            return weighted.sum(dim=-2) / attention_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        def _masked_mean(self, values: Any, mask: Any) -> Any:
            weighted = values * mask.unsqueeze(-1)
            return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        def _harmonic_terms(self, node_text: Any, spectral_positions: Any, node_mask: Any) -> tuple[Any, Any]:
            if spectral_k <= 0 or self.harmonic_projection is None or self.harmonic_scale is None:
                return torch.zeros(
                    node_text.size(0),
                    node_text.size(1),
                    hidden_dim,
                    dtype=node_text.dtype,
                    device=node_text.device,
                ), torch.tensor(0.0, dtype=node_text.dtype, device=node_text.device)

            scaled_positions = spectral_positions * self.harmonic_scale.view(1, 1, -1)
            coefficients = torch.matmul(scaled_positions.transpose(1, 2), node_text)
            harmonic_signal = torch.matmul(scaled_positions, coefficients)
            correction = self.harmonic_projection(harmonic_signal)

            ortho_sum = torch.tensor(0.0, dtype=node_text.dtype, device=node_text.device)
            energy_sum = torch.tensor(0.0, dtype=node_text.dtype, device=node_text.device)
            counted = 0
            for batch_index in range(node_text.size(0)):
                valid_nodes = node_mask[batch_index].bool()
                if not bool(valid_nodes.any()):
                    continue
                valid_positions = scaled_positions[batch_index, valid_nodes]
                gram = torch.matmul(valid_positions.transpose(0, 1), valid_positions)
                ortho_sum = ortho_sum + torch.nn.functional.mse_loss(
                    gram,
                    torch.eye(valid_positions.size(1), dtype=node_text.dtype, device=node_text.device),
                )
                energy_sum = energy_sum + coefficients[batch_index].pow(2).mean()
                counted += 1
            if counted == 0:
                regularization = torch.tensor(0.0, dtype=node_text.dtype, device=node_text.device)
            else:
                regularization = self.harmonic_reg_weight * (
                    (ortho_sum / counted) + (1e-3 * energy_sum / counted)
                )
            return correction, regularization

        def encode_graph(
            self,
            query_input_ids: Any,
            query_attention_mask: Any,
            node_input_ids: Any,
            node_attention_mask: Any,
            node_role_ids: Any,
            node_mask: Any,
            edge_index: Any,
            edge_type_ids: Any,
            edge_mask: Any,
            spectral_positions: Any,
            anchor_mask: Any,
        ) -> dict[str, Any]:
            query_embeddings = self.embedding(query_input_ids)
            query_hidden = self.query_projection(self._token_mean(query_embeddings, query_attention_mask))

            node_embeddings = self.embedding(node_input_ids)
            node_text = self._token_mean(node_embeddings, node_attention_mask)
            harmonic_correction, harmonic_regularization = self._harmonic_terms(
                node_text,
                spectral_positions,
                node_mask,
            )

            if spectral_k > 0:
                node_hidden = self.node_projection(torch.cat([node_text, spectral_positions], dim=-1))
            else:
                node_hidden = self.node_projection(node_text)
            node_hidden = torch.relu(node_hidden + harmonic_correction + self.node_role_embedding(node_role_ids))
            node_hidden = node_hidden * node_mask.unsqueeze(-1)

            edge_scores = torch.zeros(
                edge_mask.size(0),
                edge_mask.size(1),
                dtype=node_hidden.dtype,
                device=node_hidden.device,
            )
            for layer in self.layers:
                layer_hidden, layer_scores = layer(node_hidden, edge_index, edge_type_ids, edge_mask, node_mask)
                node_hidden = (node_hidden + self.dropout(layer_hidden)) * node_mask.unsqueeze(-1)
                edge_scores = edge_scores + layer_scores

            graph_pool = self._masked_mean(node_hidden, node_mask)
            anchor_weights = torch.where(
                anchor_mask.sum(dim=1, keepdim=True) > 0,
                anchor_mask,
                node_mask,
            )
            anchor_pool = self._masked_mean(node_hidden, anchor_weights)
            fusion_input = torch.cat([query_hidden, graph_pool, anchor_pool], dim=1)
            graph_context = 0.5 * (graph_pool + anchor_pool)
            gate = torch.sigmoid(self.fusion_gate(fusion_input))
            fused_hidden = gate * query_hidden + (1.0 - gate) * graph_context + torch.tanh(
                self.fusion_projection(fusion_input)
            )
            fused_hidden = self.dropout(fused_hidden)
            return {
                "hidden": fused_hidden,
                "node_hidden": node_hidden,
                "edge_scores": edge_scores,
                "harmonic_regularization": harmonic_regularization,
            }

        def encode(
            self,
            query_input_ids: Any,
            query_attention_mask: Any,
            node_input_ids: Any,
            node_attention_mask: Any,
            node_role_ids: Any,
            node_mask: Any,
            edge_index: Any,
            edge_type_ids: Any,
            edge_mask: Any,
            spectral_positions: Any,
            anchor_mask: Any,
        ) -> Any:
            return self.encode_graph(
                query_input_ids,
                query_attention_mask,
                node_input_ids,
                node_attention_mask,
                node_role_ids,
                node_mask,
                edge_index,
                edge_type_ids,
                edge_mask,
                spectral_positions,
                anchor_mask,
            )["hidden"]

        def forward(self, *args: Any) -> Any:
            hidden = self.encode(*args)
            logits = self.output(hidden)
            return _apply_memory_logits(hidden, logits, self.label_memory, self.label_memory_counts)

        def forward_with_aux(self, *args: Any, return_debug: bool = False) -> tuple[Any, Any, dict[str, Any]]:
            encoded = self.encode_graph(*args)
            hidden = encoded["hidden"]
            logits = self.output(hidden)
            logits = _apply_memory_logits(hidden, logits, self.label_memory, self.label_memory_counts)
            debug = {"edge_scores": encoded["edge_scores"]} if return_debug else {}
            return logits, encoded["harmonic_regularization"], debug

        def self_supervised_loss(
            self,
            query_input_ids: Any,
            query_attention_mask: Any,
            node_input_ids: Any,
            node_attention_mask: Any,
            node_role_ids: Any,
            node_mask: Any,
            edge_index: Any,
            edge_type_ids: Any,
            edge_mask: Any,
            spectral_positions: Any,
            anchor_mask: Any,
            *,
            tasks: tuple[str, ...],
        ) -> Any:
            encoded = self.encode_graph(
                query_input_ids,
                query_attention_mask,
                node_input_ids,
                node_attention_mask,
                node_role_ids,
                node_mask,
                edge_index,
                edge_type_ids,
                edge_mask,
                spectral_positions,
                anchor_mask,
            )
            node_hidden = encoded["node_hidden"]
            losses: list[Any] = []
            for batch_index in range(node_hidden.size(0)):
                valid_edges = edge_mask[batch_index].bool()
                valid_node_count = int(node_mask[batch_index].sum().item())
                if not bool(valid_edges.any()) or valid_node_count <= 0:
                    continue
                batch_edges = edge_index[batch_index, valid_edges].long()
                batch_relations = edge_type_ids[batch_index, valid_edges].long()
                source_hidden = node_hidden[batch_index, batch_edges[:, 0]]
                target_hidden = node_hidden[batch_index, batch_edges[:, 1]]
                if "relation_prediction" in tasks:
                    relation_features = torch.cat(
                        [source_hidden, target_hidden, source_hidden * target_hidden, (source_hidden - target_hidden).abs()],
                        dim=1,
                    )
                    relation_logits = self.relation_decoder(relation_features)
                    losses.append(torch.nn.functional.cross_entropy(relation_logits, batch_relations))
                if "edge_denoising" in tasks and valid_node_count > 1:
                    negative_targets = (batch_edges[:, 1] + 1) % valid_node_count
                    negative_hidden = node_hidden[batch_index, negative_targets]
                    relation_vectors = self.relation_embedding(batch_relations)
                    positive_features = torch.cat(
                        [source_hidden, target_hidden, source_hidden * target_hidden, relation_vectors],
                        dim=1,
                    )
                    negative_features = torch.cat(
                        [source_hidden, negative_hidden, source_hidden * negative_hidden, relation_vectors],
                        dim=1,
                    )
                    positive_logits = self.edge_decoder(positive_features).squeeze(-1)
                    negative_logits = self.edge_decoder(negative_features).squeeze(-1)
                    losses.append(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            positive_logits,
                            torch.ones_like(positive_logits),
                        )
                    )
                    losses.append(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            negative_logits,
                            torch.zeros_like(negative_logits),
                        )
                    )
            if not losses:
                return encoded["harmonic_regularization"]
            return torch.stack(losses).mean() + encoded["harmonic_regularization"]

    return ZRIAGraphClassifier()


def _graph_batch_inputs(batch: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(batch[field_name] for field_name in _GRAPH_BATCH_FIELDS)


def _clone_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _apply_memory_logits(hidden: Any, logits: Any, label_memory: Any, label_memory_counts: Any) -> Any:
    torch, _, _ = _require_torch()
    if label_memory.numel() == 0 or float(label_memory_counts.sum().item()) <= 0.0:
        return logits
    normalized_hidden = torch.nn.functional.normalize(hidden, dim=1, eps=1e-6)
    normalized_memory = torch.nn.functional.normalize(label_memory, dim=1, eps=1e-6)
    memory_logits = normalized_hidden @ normalized_memory.transpose(0, 1)
    active_labels = (label_memory_counts > 0).to(logits.dtype).unsqueeze(0)
    return logits + (memory_logits * active_labels)


def _memory_payload(model: Any) -> tuple[list[list[float]], list[float]]:
    return (
        model.label_memory.detach().cpu().tolist(),
        model.label_memory_counts.detach().cpu().tolist(),
    )


def _load_memory_artifact(path: str | Path) -> ZRIALearnedArtifact:
    artifact_path = Path(path)
    if artifact_path.is_dir():
        artifact_path = artifact_path / "artifact.json"
    with artifact_path.open("r", encoding="utf-8") as handle:
        return ZRIALearnedArtifact.model_validate(json.load(handle))


def _restore_memory_from_artifact(
    model: Any,
    artifact: ZRIALearnedArtifact,
    labels: list[EvaluationTarget],
    hidden_dim: int,
) -> int:
    torch, _, _ = _require_torch()
    if not artifact.memory_enabled or not artifact.memory_vectors:
        return 0
    if len(artifact.memory_vectors) != len(artifact.labels):
        raise ValueError("Persisted ZRIA memory vectors do not align with artifact labels.")
    label_to_memory = {
        _label_key(label): vector
        for label, vector in zip(artifact.labels, artifact.memory_vectors, strict=False)
    }
    label_to_count = {
        _label_key(label): artifact.memory_counts[index] if index < len(artifact.memory_counts) else 0.0
        for index, label in enumerate(artifact.labels)
    }

    restored_vectors: list[list[float]] = []
    restored_counts: list[float] = []
    restored_labels = 0
    for label in labels:
        vector = label_to_memory.get(_label_key(label))
        if vector is None:
            restored_vectors.append([0.0] * hidden_dim)
            restored_counts.append(0.0)
            continue
        if len(vector) != hidden_dim:
            raise ValueError("Persisted ZRIA memory vector width does not match the current hidden_dim.")
        restored_vectors.append([float(value) for value in vector])
        restored_counts.append(float(label_to_count.get(_label_key(label), 0.0)))
        restored_labels += 1

    memory_tensor = torch.tensor(restored_vectors, dtype=model.label_memory.dtype, device=model.label_memory.device)
    count_tensor = torch.tensor(
        restored_counts,
        dtype=model.label_memory_counts.dtype,
        device=model.label_memory_counts.device,
    )
    model.label_memory.copy_(memory_tensor)
    model.label_memory_counts.copy_(count_tensor)
    return restored_labels


def _update_memory_bank(
    model: Any,
    dataloader: Any,
    num_labels: int,
    momentum: float,
    *,
    backend_type: Literal["learned", "learned_graph"] = "learned",
) -> int:
    torch, _, _ = _require_torch()
    was_training = model.training
    model.eval()
    hidden_chunks: list[Any] = []
    label_chunks: list[Any] = []
    with torch.no_grad():
        for batch in dataloader:
            if backend_type == "learned_graph":
                hidden = model.encode(*_graph_batch_inputs(batch))
            else:
                hidden = model.encode(batch["input_ids"], batch["attention_mask"])
            hidden_chunks.append(hidden.detach().cpu())
            label_chunks.append(batch["label"].detach().cpu())
    if was_training:
        model.train()
    if not hidden_chunks:
        return 0

    hidden_tensor = torch.cat(hidden_chunks, dim=0)
    label_tensor = torch.cat(label_chunks, dim=0)
    current_memory = model.label_memory.detach().cpu()
    current_counts = model.label_memory_counts.detach().cpu()
    updated_memory = current_memory.clone()
    updated_counts = current_counts.clone()
    updated_labels = 0
    for label_idx in range(num_labels):
        label_mask = label_tensor == label_idx
        if not bool(label_mask.any()):
            continue
        centroid = hidden_tensor[label_mask].mean(dim=0)
        if float(current_counts[label_idx].item()) > 0.0:
            centroid = (momentum * current_memory[label_idx]) + ((1.0 - momentum) * centroid)
        updated_memory[label_idx] = centroid
        updated_counts[label_idx] = current_counts[label_idx] + float(label_mask.sum().item())
        updated_labels += 1

    model.label_memory.copy_(updated_memory.to(model.label_memory.device))
    model.label_memory_counts.copy_(updated_counts.to(model.label_memory_counts.device))
    return updated_labels


def _evaluate_classifier(
    model: Any,
    examples: list[ZRIAExample],
    vocab: dict[str, int],
    labels: list[EvaluationTarget],
    max_length: int,
    batch_size: int,
    *,
    backend_type: Literal["learned", "learned_graph"] = "learned",
    spectral_k: int = 0,
    relation_vocab: dict[str, int] | None = None,
    max_graph_nodes: int = 0,
    max_graph_edges: int = 0,
    node_token_length: int = _GRAPH_NODE_TOKEN_LENGTH,
) -> tuple[float, float]:
    torch, _, DataLoader = _require_torch()
    if backend_type == "learned_graph":
        dataset = _ZRIAGraphDataset(
            examples,
            vocab,
            relation_vocab or {_REL_PAD: 0},
            labels,
            max_length,
            spectral_k,
            max_graph_nodes=max_graph_nodes,
            max_graph_edges=max_graph_edges,
            node_token_length=node_token_length,
        )
    else:
        dataset = _ZRIADataset(examples, vocab, labels, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0
    correct = 0
    score_sum = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if backend_type == "learned_graph":
                logits, _, _ = model.forward_with_aux(*_graph_batch_inputs(batch))
            else:
                logits = model(batch["input_ids"], batch["attention_mask"])
            predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == batch["label"]).sum().item())
            total += int(batch["label"].numel())
            for expected_idx, predicted_idx in zip(batch["label"].tolist(), predictions.tolist(), strict=False):
                score_sum += evaluate_case(
                    EvaluationCase(
                        case_id=f"eval-{total}",
                        task="evaluate",
                        expected=labels[int(expected_idx)],
                        predicted=labels[int(predicted_idx)],
                    )
                ).score
    if total == 0:
        return 0.0, 0.0
    return correct / total, score_sum / total


def _collect_classifier_logits(
    model: Any,
    examples: list[ZRIAExample],
    vocab: dict[str, int],
    labels: list[EvaluationTarget],
    max_length: int,
    batch_size: int,
    *,
    backend_type: Literal["learned", "learned_graph"] = "learned",
    spectral_k: int = 0,
    relation_vocab: dict[str, int] | None = None,
    max_graph_nodes: int = 0,
    max_graph_edges: int = 0,
    node_token_length: int = _GRAPH_NODE_TOKEN_LENGTH,
) -> tuple[list[Any], list[Any]]:
    torch, _, DataLoader = _require_torch()
    if backend_type == "learned_graph":
        dataset = _ZRIAGraphDataset(
            examples,
            vocab,
            relation_vocab or {_REL_PAD: 0},
            labels,
            max_length,
            spectral_k,
            max_graph_nodes=max_graph_nodes,
            max_graph_edges=max_graph_edges,
            node_token_length=node_token_length,
        )
    else:
        dataset = _ZRIADataset(examples, vocab, labels, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks: list[Any] = []
    label_chunks: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if backend_type == "learned_graph":
                logits, _, _ = model.forward_with_aux(*_graph_batch_inputs(batch))
            else:
                logits = model(batch["input_ids"], batch["attention_mask"])
            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(batch["label"].detach().cpu())
    return logits_chunks, label_chunks


def _fit_temperature(logits_chunks: list[Any], label_chunks: list[Any]) -> float:
    torch, _, _ = _require_torch()
    if not logits_chunks or not label_chunks:
        return 1.0
    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    if logits.numel() == 0 or labels.numel() == 0:
        return 1.0

    best_temperature = 1.0
    best_loss: float | None = None
    for step in range(1, 41):
        temperature = 0.25 + (step * 0.125)
        loss = float(torch.nn.functional.cross_entropy(logits / temperature, labels).item())
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _graph_explanation(
    forward_edges: list[Triple],
    edge_scores: list[float],
    *,
    limit: int = 5,
) -> list[ZRIAGraphEvidence]:
    if not forward_edges or not edge_scores:
        return []
    paired = list(zip(forward_edges, edge_scores[: len(forward_edges)], strict=False))
    paired.sort(key=lambda item: item[1], reverse=True)
    max_score = max(score for _, score in paired[:limit]) if paired else 1.0
    if max_score <= 0:
        max_score = 1.0
    return [
        ZRIAGraphEvidence(
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            score=float(score / max_score),
        )
        for triple, score in paired[:limit]
    ]


def _run_graph_self_supervision(
    model: Any,
    dataloader: Any,
    *,
    epochs: int,
    learning_rate: float,
    tasks: tuple[str, ...],
) -> dict[str, Any]:
    torch, _, _ = _require_torch()
    if epochs <= 0 or not tasks:
        return {"pretrain_epochs": 0, "pretrain_tasks": list(tasks), "pretrain_loss": 0.0}

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            loss = model.self_supervised_loss(*_graph_batch_inputs(batch), tasks=tasks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1
        last_loss = epoch_loss / max(1, batch_count)
    return {
        "pretrain_epochs": epochs,
        "pretrain_tasks": list(tasks),
        "pretrain_loss": last_loss,
    }


def train_model(
    train_examples: list[ZRIAExample],
    output_dir: str | Path,
    *,
    eval_examples: list[ZRIAExample] | None = None,
    epochs: int = 80,
    learning_rate: float = 1e-2,
    max_length: int = 64,
    embedding_dim: int = 32,
    spectral_k: int = 0,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    batch_size: int = 4,
    seed: int = 42,
    patience: int | None = None,
    memory_momentum: float = 0.9,
    resume_memory_from: str | Path | None = None,
) -> dict[str, Any]:
    torch, nn, DataLoader = _require_torch()
    torch.manual_seed(seed)
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(train_examples, include_neighborhood=False)
    labels = build_label_set(train_examples)
    model = create_model(vocab, labels, embedding_dim, spectral_k, hidden_dim, dropout=dropout)
    restored_memory_labels = 0
    if resume_memory_from is not None:
        restored_memory_labels = _restore_memory_from_artifact(
            model,
            _load_memory_artifact(resume_memory_from),
            labels,
            hidden_dim,
        )
    dataset = _ZRIADataset(train_examples, vocab, labels, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_state = _clone_state_dict(model)
    best_epoch = 0
    best_eval_accuracy = 0.0
    best_average_score = 0.0
    best_monitor: float | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    memory_labels_persisted = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        batch_count = 0
        for batch in dataloader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().cpu().item())
            batch_count += 1
        epochs_trained = epoch + 1
        memory_labels_persisted = _update_memory_bank(
            model,
            dataloader,
            len(labels),
            memory_momentum,
            backend_type="learned",
        )

        average_loss = epoch_loss_sum / max(1, batch_count)
        if eval_examples:
            eval_accuracy, average_score = _evaluate_classifier(
                model,
                eval_examples,
                vocab,
                labels,
                max_length,
                batch_size,
                backend_type="learned",
                spectral_k=spectral_k,
            )
            monitor = eval_accuracy
        else:
            eval_accuracy = 0.0
            average_score = 0.0
            monitor = -average_loss

        if best_monitor is None or monitor > best_monitor + 1e-6:
            best_monitor = monitor
            best_epoch = epoch + 1
            best_eval_accuracy = eval_accuracy
            best_average_score = average_score
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    calibration_temperature = 1.0
    calibration_source: Literal["none", "train", "eval"] = "none"
    calibration_examples = eval_examples if eval_examples else train_examples
    if calibration_examples:
        logits_chunks, label_chunks = _collect_classifier_logits(
            model,
            calibration_examples,
            vocab,
            labels,
            max_length,
            batch_size,
            backend_type="learned",
            spectral_k=spectral_k,
        )
        calibration_temperature = _fit_temperature(logits_chunks, label_chunks)
        calibration_source = "eval" if eval_examples else "train"

    state_dict_path = output_path / "model.pt"
    torch.save(model.state_dict(), state_dict_path)
    memory_vectors, memory_counts = _memory_payload(model)
    artifact = ZRIALearnedArtifact(
        backend_type="learned",
        vocab=vocab,
        labels=labels,
        max_length=max_length,
        embedding_dim=embedding_dim,
        spectral_k=spectral_k,
        hidden_dim=hidden_dim,
        dropout=dropout,
        graph_feature_dim=0,
        calibration_temperature=calibration_temperature,
        calibration_source=calibration_source,
        memory_enabled=True,
        memory_momentum=memory_momentum,
        memory_vectors=memory_vectors,
        memory_counts=memory_counts,
        state_dict_path=str(state_dict_path),
    )
    with (output_path / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact.model_dump(mode="json"), handle, indent=2)

    metrics = {
        "train_examples": len(train_examples),
        "labels": len(labels),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "early_stopped": patience is not None and patience > 0 and epochs_trained < epochs,
        "spectral_k": spectral_k,
        "dropout": dropout,
        "backend_type": "learned",
        "calibration_temperature": calibration_temperature,
        "calibration_source": calibration_source,
        "memory_momentum": memory_momentum,
        "memory_labels_persisted": memory_labels_persisted,
        "memory_labels_restored": restored_memory_labels,
    }
    if eval_examples:
        metrics["eval_accuracy"] = best_eval_accuracy
        metrics["eval_average_score"] = best_average_score
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def train_graph_model(
    train_examples: list[ZRIAExample],
    output_dir: str | Path,
    *,
    eval_examples: list[ZRIAExample] | None = None,
    epochs: int = 80,
    learning_rate: float = 1e-2,
    max_length: int = 64,
    embedding_dim: int = 32,
    spectral_k: int = 16,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    batch_size: int = 4,
    seed: int = 42,
    patience: int | None = None,
    memory_momentum: float = 0.9,
    resume_memory_from: str | Path | None = None,
    graph_num_layers: int = 2,
    harmonic_reg_weight: float = 5e-4,
    pretrain_epochs: int = 0,
    pretrain_learning_rate: float | None = None,
    pretraining_tasks: tuple[str, ...] = ("relation_prediction", "edge_denoising"),
) -> dict[str, Any]:
    torch, nn, DataLoader = _require_torch()
    torch.manual_seed(seed)
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(train_examples, include_neighborhood=True)
    relation_vocab = build_relation_vocab(train_examples)
    max_graph_nodes, max_graph_edges = _graph_shape_limits(train_examples)
    labels = build_label_set(train_examples)
    model = create_graph_model(
        vocab,
        relation_vocab,
        labels,
        embedding_dim,
        spectral_k,
        hidden_dim,
        dropout=dropout,
        graph_num_layers=graph_num_layers,
        harmonic_reg_weight=harmonic_reg_weight,
    )
    restored_memory_labels = 0
    if resume_memory_from is not None:
        restored_memory_labels = _restore_memory_from_artifact(
            model,
            _load_memory_artifact(resume_memory_from),
            labels,
            hidden_dim,
        )
    dataset = _ZRIAGraphDataset(
        train_examples,
        vocab,
        relation_vocab,
        labels,
        max_length,
        spectral_k,
        max_graph_nodes=max_graph_nodes,
        max_graph_edges=max_graph_edges,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    pretraining_metrics = _run_graph_self_supervision(
        model,
        dataloader,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate or learning_rate,
        tasks=pretraining_tasks,
    )

    best_state = _clone_state_dict(model)
    best_epoch = 0
    best_eval_accuracy = 0.0
    best_average_score = 0.0
    best_monitor: float | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    memory_labels_persisted = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        batch_count = 0
        for batch in dataloader:
            logits, harmonic_loss, _ = model.forward_with_aux(*_graph_batch_inputs(batch))
            loss = criterion(logits, batch["label"]) + harmonic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().cpu().item())
            batch_count += 1
        epochs_trained = epoch + 1
        memory_labels_persisted = _update_memory_bank(
            model,
            dataloader,
            len(labels),
            memory_momentum,
            backend_type="learned_graph",
        )

        average_loss = epoch_loss_sum / max(1, batch_count)
        if eval_examples:
            eval_accuracy, average_score = _evaluate_classifier(
                model,
                eval_examples,
                vocab,
                labels,
                max_length,
                batch_size,
                backend_type="learned_graph",
                spectral_k=spectral_k,
                relation_vocab=relation_vocab,
                max_graph_nodes=max_graph_nodes,
                max_graph_edges=max_graph_edges,
            )
            monitor = eval_accuracy
        else:
            eval_accuracy = 0.0
            average_score = 0.0
            monitor = -average_loss

        if best_monitor is None or monitor > best_monitor + 1e-6:
            best_monitor = monitor
            best_epoch = epoch + 1
            best_eval_accuracy = eval_accuracy
            best_average_score = average_score
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    calibration_temperature = 1.0
    calibration_source: Literal["none", "train", "eval"] = "none"
    calibration_examples = eval_examples if eval_examples else train_examples
    if calibration_examples:
        logits_chunks, label_chunks = _collect_classifier_logits(
            model,
            calibration_examples,
            vocab,
            labels,
            max_length,
            batch_size,
            backend_type="learned_graph",
            spectral_k=spectral_k,
            relation_vocab=relation_vocab,
            max_graph_nodes=max_graph_nodes,
            max_graph_edges=max_graph_edges,
        )
        calibration_temperature = _fit_temperature(logits_chunks, label_chunks)
        calibration_source = "eval" if eval_examples else "train"

    state_dict_path = output_path / "model.pt"
    torch.save(model.state_dict(), state_dict_path)
    memory_vectors, memory_counts = _memory_payload(model)
    artifact = ZRIALearnedArtifact(
        backend_type="learned_graph",
        vocab=vocab,
        labels=labels,
        max_length=max_length,
        embedding_dim=embedding_dim,
        spectral_k=spectral_k,
        hidden_dim=hidden_dim,
        dropout=dropout,
        graph_feature_dim=graph_feature_dim(spectral_k),
        relation_vocab=relation_vocab,
        graph_node_token_length=_GRAPH_NODE_TOKEN_LENGTH,
        graph_max_nodes=max_graph_nodes,
        graph_max_edges=max_graph_edges,
        graph_num_layers=graph_num_layers,
        harmonic_reg_weight=harmonic_reg_weight,
        calibration_temperature=calibration_temperature,
        calibration_source=calibration_source,
        self_supervised_tasks=list(pretraining_tasks if pretrain_epochs > 0 else ()),
        memory_enabled=True,
        memory_momentum=memory_momentum,
        memory_vectors=memory_vectors,
        memory_counts=memory_counts,
        state_dict_path=str(state_dict_path),
    )
    with (output_path / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact.model_dump(mode="json"), handle, indent=2)

    metrics = {
        "train_examples": len(train_examples),
        "labels": len(labels),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "early_stopped": patience is not None and patience > 0 and epochs_trained < epochs,
        "spectral_k": spectral_k,
        "dropout": dropout,
        "backend_type": "learned_graph",
        "graph_num_layers": graph_num_layers,
        "harmonic_reg_weight": harmonic_reg_weight,
        "calibration_temperature": calibration_temperature,
        "calibration_source": calibration_source,
        "memory_momentum": memory_momentum,
        "memory_labels_persisted": memory_labels_persisted,
        "memory_labels_restored": restored_memory_labels,
    }
    metrics.update(pretraining_metrics)
    if eval_examples:
        metrics["eval_accuracy"] = best_eval_accuracy
        metrics["eval_average_score"] = best_average_score
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def load_artifact(path: str | Path) -> tuple[ZRIALearnedArtifact, Any]:
    torch, _, _ = _require_torch()
    base_path = Path(path)
    artifact_path = base_path if base_path.name.endswith(".json") else base_path / "artifact.json"
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = ZRIALearnedArtifact.model_validate(json.load(handle))
    state_dict_path = Path(artifact.state_dict_path)
    if not state_dict_path.is_absolute():
        state_dict_path = artifact_path.parent / state_dict_path.name
    if artifact.backend_type == "learned_graph":
        model = create_graph_model(
            artifact.vocab,
            artifact.relation_vocab,
            artifact.labels,
            artifact.embedding_dim,
            artifact.spectral_k,
            artifact.hidden_dim,
            dropout=artifact.dropout,
            graph_num_layers=artifact.graph_num_layers,
            harmonic_reg_weight=artifact.harmonic_reg_weight,
        )
    else:
        model = create_model(
            artifact.vocab,
            artifact.labels,
            artifact.embedding_dim,
            artifact.spectral_k,
            artifact.hidden_dim,
            dropout=artifact.dropout,
        )
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"), strict=False)
    _restore_memory_from_artifact(model, artifact, artifact.labels, artifact.hidden_dim)
    model.eval()
    return artifact, model


def predict_with_model(
    artifact: ZRIALearnedArtifact,
    model: Any,
    query: str,
    parsed: ParseTarget,
    kg_scope: KGScope | None = None,
    neighborhood: list[Triple] | None = None,
    *,
    return_debug: bool = False,
) -> tuple[EvaluationTarget, float] | tuple[EvaluationTarget, float, dict[str, Any]]:
    torch, _, _ = _require_torch()
    if artifact.backend_type == "learned_graph":
        feature_text = render_feature_text(query, parsed, kg_scope)
    else:
        feature_text = render_feature_text(query, parsed, kg_scope)
    input_ids = torch.tensor([encode_text(feature_text, artifact.vocab, artifact.max_length)], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1 if token_id != artifact.vocab[_PAD] else 0 for token_id in input_ids[0].tolist()]],
        dtype=torch.float32,
    )
    debug_payload: dict[str, Any] = {}
    with torch.no_grad():
        if artifact.backend_type == "learned_graph":
            graph_features = build_graph_features(
                neighborhood or [],
                parsed,
                artifact.vocab,
                artifact.relation_vocab or {_REL_PAD: 0},
                artifact.spectral_k,
                max(artifact.graph_max_nodes, 2),
                max(artifact.graph_max_edges, 2),
                artifact.graph_node_token_length,
                include_metadata=return_debug,
            )
            graph_batch = {
                "query_input_ids": input_ids,
                "query_attention_mask": attention_mask,
                "node_input_ids": torch.tensor([graph_features["node_input_ids"]], dtype=torch.long),
                "node_attention_mask": torch.tensor([graph_features["node_attention_mask"]], dtype=torch.float32),
                "node_role_ids": torch.tensor([graph_features["node_role_ids"]], dtype=torch.long),
                "node_mask": torch.tensor([graph_features["node_mask"]], dtype=torch.float32),
                "edge_index": torch.tensor([graph_features["edge_index"]], dtype=torch.long),
                "edge_type_ids": torch.tensor([graph_features["edge_type_ids"]], dtype=torch.long),
                "edge_mask": torch.tensor([graph_features["edge_mask"]], dtype=torch.float32),
                "spectral_positions": torch.tensor([graph_features["spectral_positions"]], dtype=torch.float32),
                "anchor_mask": torch.tensor([graph_features["anchor_mask"]], dtype=torch.float32),
            }
            logits, _, graph_debug = model.forward_with_aux(*_graph_batch_inputs(graph_batch), return_debug=return_debug)
            if return_debug:
                edge_scores = graph_debug.get("edge_scores")
                explanation_scores = [] if edge_scores is None else edge_scores[0].detach().cpu().tolist()
                debug_payload["graph_explanation"] = _graph_explanation(
                    graph_features.get("forward_edges", []),
                    explanation_scores[: graph_features["forward_edge_count"]],
                )
        else:
            logits = model(input_ids, attention_mask)
        raw_probabilities = torch.softmax(logits, dim=-1)
        calibrated_probabilities = torch.softmax(
            logits / max(artifact.calibration_temperature, 1e-3),
            dim=-1,
        )
    confidence, label_idx = torch.max(calibrated_probabilities, dim=-1)
    raw_confidence, _ = torch.max(raw_probabilities, dim=-1)
    debug_payload["raw_confidence"] = float(raw_confidence.item())
    debug_payload["temperature"] = float(artifact.calibration_temperature)
    prediction = artifact.labels[int(label_idx.item())]
    if return_debug:
        return prediction, float(confidence.item()), debug_payload
    return prediction, float(confidence.item())


def evaluate_learned_backend(path: str | Path, examples: list[ZRIAExample]) -> ZRIAComparisonReport:
    artifact, model = load_artifact(path)
    details: list[ZRIAComparisonDetail] = []
    for example in examples:
        predicted, confidence = predict_with_model(
            artifact,
            model,
            example.query,
            example.parsed,
            example.kg_scope,
            neighborhood=example.neighborhood,
        )
        evaluation = evaluate_case(
            EvaluationCase(
                case_id=example.example_id,
                task="evaluate",
                expected=example.target,
                predicted=predicted,
            )
        )
        details.append(
            ZRIAComparisonDetail(
                case_id=example.example_id,
                backend=artifact.backend_type,
                confidence=confidence,
                predicted=predicted,
                evaluation=evaluation,
            )
        )
    passed_cases = sum(1 for detail in details if detail.evaluation.passed)
    average_score = 0.0 if not details else sum(detail.evaluation.score for detail in details) / len(details)
    return ZRIAComparisonReport(
        summaries=[
            ZRIAComparisonSummary(
                backend=artifact.backend_type,
                total_cases=len(details),
                passed_cases=passed_cases,
                accuracy=0.0 if not details else passed_cases / len(details),
                average_score=average_score,
            )
        ],
        details=details,
    )


def compare_backends(
    *,
    learned_model_path: str | Path,
    rules_path: str | Path,
    examples: list[ZRIAExample],
    confidence_threshold: float = 0.6,
) -> ZRIAComparisonReport:
    from msa_zria.zria_backend import RuleBasedZRIABackend

    learned_artifact, learned_model = load_artifact(learned_model_path)
    rules_backend = RuleBasedZRIABackend.from_path(rules_path)
    details: list[ZRIAComparisonDetail] = []
    for example in examples:
        rules_prediction = rules_backend.predict(example.query, parsed=example.parsed, kg_scope=example.kg_scope)
        details.append(
            ZRIAComparisonDetail(
                case_id=example.example_id,
                backend="rules",
                confidence=None,
                predicted=rules_prediction,
                evaluation=evaluate_case(
                    EvaluationCase(
                        case_id=example.example_id,
                        task="evaluate",
                        expected=example.target,
                        predicted=rules_prediction,
                    )
                ),
            )
        )
        learned_prediction, confidence = predict_with_model(
            learned_artifact,
            learned_model,
            example.query,
            example.parsed,
            example.kg_scope,
            neighborhood=example.neighborhood,
        )
        details.append(
            ZRIAComparisonDetail(
                case_id=example.example_id,
                backend=learned_artifact.backend_type,
                confidence=confidence,
                predicted=learned_prediction,
                evaluation=evaluate_case(
                    EvaluationCase(
                        case_id=example.example_id,
                        task="evaluate",
                        expected=example.target,
                        predicted=learned_prediction,
                    )
                ),
            )
        )

    summaries: list[ZRIAComparisonSummary] = []
    for backend_name in ["rules", learned_artifact.backend_type]:
        backend_details = [detail for detail in details if detail.backend == backend_name]
        passed_cases = sum(1 for detail in backend_details if detail.evaluation.passed)
        average_score = 0.0 if not backend_details else sum(
            detail.evaluation.score for detail in backend_details
        ) / len(backend_details)
        summaries.append(
            ZRIAComparisonSummary(
                backend=backend_name,
                total_cases=len(backend_details),
                passed_cases=passed_cases,
                accuracy=0.0 if not backend_details else passed_cases / len(backend_details),
                average_score=average_score,
            )
        )
    return ZRIAComparisonReport(summaries=summaries, details=details)


def run_sweep(
    *,
    train_examples: list[ZRIAExample],
    eval_examples: list[ZRIAExample],
    output_dir: str | Path,
    backend_type: Literal["learned", "learned_graph"] = "learned",
    spectral_k_values: list[int],
    dropout_values: list[float],
    epochs: int = 80,
    learning_rate: float = 1e-2,
    max_length: int = 64,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    batch_size: int = 4,
    seed: int = 42,
    patience: int | None = None,
) -> ZRIASweepReport:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[ZRIASweepRecord] = []
    for spectral_k in spectral_k_values:
        for dropout in dropout_values:
            run_dir = output_path / f"spectral_k_{spectral_k}_dropout_{str(dropout).replace('.', '_')}"
            if backend_type == "learned_graph":
                metrics = train_graph_model(
                    train_examples,
                    run_dir,
                    eval_examples=eval_examples,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    max_length=max_length,
                    embedding_dim=embedding_dim,
                    spectral_k=spectral_k,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    batch_size=batch_size,
                    seed=seed,
                    patience=patience,
                )
            else:
                metrics = train_model(
                    train_examples,
                    run_dir,
                    eval_examples=eval_examples,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    max_length=max_length,
                    embedding_dim=embedding_dim,
                    spectral_k=spectral_k,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    batch_size=batch_size,
                    seed=seed,
                    patience=patience,
                )
            records.append(
                ZRIASweepRecord(
                    spectral_k=spectral_k,
                    dropout=dropout,
                    artifact_path=str(run_dir),
                    eval_accuracy=float(metrics.get("eval_accuracy", 0.0)),
                    average_score=float(metrics.get("eval_average_score", 0.0)),
                    epochs_trained=int(metrics["epochs_trained"]),
                    best_epoch=int(metrics["best_epoch"]),
                    early_stopped=bool(metrics["early_stopped"]),
                )
            )

    if not records:
        raise ValueError("At least one spectral_k and one dropout value are required for a sweep.")

    best_record = max(records, key=lambda record: (record.eval_accuracy, record.average_score))
    return ZRIASweepReport(records=records, best_record=best_record)


def write_comparison_report(report: ZRIAComparisonReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)


def write_sweep_report(report: ZRIASweepReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the learned local ZRIA backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train", required=True, help="Path to ZRIA training examples JSONL.")
    train_parser.add_argument("--output", required=True, help="Directory for learned ZRIA artifacts.")
    train_parser.add_argument("--eval", help="Optional eval-set path.")
    train_parser.add_argument("--backend-type", choices=["learned", "learned_graph"], default="learned")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=1e-2)
    train_parser.add_argument("--max-length", type=int, default=64)
    train_parser.add_argument("--embedding-dim", type=int, default=32)
    train_parser.add_argument("--spectral-k", type=int, default=0)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--memory-momentum", type=float, default=0.9)
    train_parser.add_argument("--graph-layers", type=int, default=2)
    train_parser.add_argument("--harmonic-reg-weight", type=float, default=5e-4)
    train_parser.add_argument("--pretrain-epochs", type=int, default=0)
    train_parser.add_argument("--pretrain-learning-rate", type=float)
    train_parser.add_argument(
        "--resume-memory-from",
        help="Optional artifact directory or artifact.json path to warm-start persisted ZRIA memory.",
    )
    train_parser.add_argument("--audit-path", help="Optional audit JSONL output path.")
    train_parser.add_argument("--audit-wwkg", action="store_true")
    train_parser.add_argument("--confidence-threshold", type=float, default=0.6)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", required=True, help="Artifact directory or artifact.json path.")
    eval_parser.add_argument("--input", required=True, help="Path to ZRIA eval examples JSONL.")
    eval_parser.add_argument("--output", help="Optional JSON report path.")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--model", required=True, help="Learned ZRIA artifact directory or artifact.json path.")
    compare_parser.add_argument("--rules", required=True, help="Path to rules backend JSON.")
    compare_parser.add_argument("--input", required=True, help="Path to ZRIA eval examples JSONL.")
    compare_parser.add_argument("--output", help="Optional JSON report path.")
    compare_parser.add_argument("--confidence-threshold", type=float, default=0.6)
    compare_parser.add_argument("--audit-path", help="Optional audit JSONL output path.")
    compare_parser.add_argument("--audit-wwkg", action="store_true")

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--train", required=True, help="Path to ZRIA training examples JSONL.")
    sweep_parser.add_argument("--eval", required=True, help="Path to ZRIA eval examples JSONL.")
    sweep_parser.add_argument("--output-dir", required=True, help="Directory to store learned artifacts for each run.")
    sweep_parser.add_argument("--output", help="Optional JSON report path.")
    sweep_parser.add_argument("--backend-type", choices=["learned", "learned_graph"], default="learned")
    sweep_parser.add_argument("--spectral-k", type=int, action="append", dest="spectral_k_values")
    sweep_parser.add_argument("--dropout", type=float, action="append", dest="dropout_values")
    sweep_parser.add_argument("--epochs", type=int, default=80)
    sweep_parser.add_argument("--learning-rate", type=float, default=1e-2)
    sweep_parser.add_argument("--max-length", type=int, default=64)
    sweep_parser.add_argument("--embedding-dim", type=int, default=32)
    sweep_parser.add_argument("--hidden-dim", type=int, default=64)
    sweep_parser.add_argument("--batch-size", type=int, default=4)
    sweep_parser.add_argument("--seed", type=int, default=42)
    sweep_parser.add_argument("--patience", type=int, default=10)
    sweep_parser.add_argument("--audit-path", help="Optional audit JSONL output path.")
    sweep_parser.add_argument("--audit-wwkg", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        train_examples = load_zria_examples(args.train)
        eval_examples = load_zria_examples(args.eval) if args.eval else None
        spectral_k = args.spectral_k
        if args.backend_type == "learned_graph" and spectral_k == 0:
            spectral_k = 16
        train_fn = train_graph_model if args.backend_type == "learned_graph" else train_model
        train_kwargs = {
            "eval_examples": eval_examples,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "spectral_k": spectral_k,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "patience": args.patience,
            "memory_momentum": args.memory_momentum,
            "resume_memory_from": args.resume_memory_from,
        }
        if args.backend_type == "learned_graph":
            train_kwargs.update(
                {
                    "graph_num_layers": args.graph_layers,
                    "harmonic_reg_weight": args.harmonic_reg_weight,
                    "pretrain_epochs": args.pretrain_epochs,
                    "pretrain_learning_rate": args.pretrain_learning_rate,
                }
            )
        metrics = train_fn(
            train_examples,
            args.output,
            **train_kwargs,
        )
        if args.audit_path:
            recorder = AuditRecorder.from_output_path(args.audit_path, wwkg_enabled=args.audit_wwkg)
            recorder.record_model_lineage(
                experiment_config_hash=sha256_json(
                    {
                        "command": "zria_train",
                        "backend_type": args.backend_type,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "max_length": args.max_length,
                        "embedding_dim": args.embedding_dim,
                        "spectral_k": spectral_k,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                        "patience": args.patience,
                        "memory_momentum": args.memory_momentum,
                        "resume_memory_from": args.resume_memory_from,
                        "graph_layers": args.graph_layers,
                        "harmonic_reg_weight": args.harmonic_reg_weight,
                        "pretrain_epochs": args.pretrain_epochs,
                        "pretrain_learning_rate": args.pretrain_learning_rate,
                    }
                ),
                training_dataset_version=stable_dataset_version(
                    [args.train, args.eval] if args.eval else [args.train]
                ),
                model_artifact_path=args.output,
                model_artifact_hash=sha256_directory(args.output),
                backend_type=args.backend_type,
                fallback_setting=True,
                confidence_threshold=args.confidence_threshold,
            )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "evaluate":
        report = evaluate_learned_backend(args.model, load_zria_examples(args.input))
        if args.output:
            write_comparison_report(report, args.output)
        print(report.model_dump_json(indent=2))
        return

    if args.command == "compare":
        report = compare_backends(
            learned_model_path=args.model,
            rules_path=args.rules,
            examples=load_zria_examples(args.input),
            confidence_threshold=args.confidence_threshold,
        )
        if args.output:
            write_comparison_report(report, args.output)
        if args.audit_path:
            recorder = AuditRecorder.from_output_path(args.audit_path, wwkg_enabled=args.audit_wwkg)
            recorder.record_validation_evidence(
                report_id=event_report_id(report.model_dump(mode="json")),
                evidence_type="learned_vs_rules_comparison",
                per_backend_scores={
                    summary.backend: {
                        "total_cases": summary.total_cases,
                        "passed_cases": summary.passed_cases,
                        "accuracy": summary.accuracy,
                        "average_score": summary.average_score,
                    }
                    for summary in report.summaries
                },
                artifact_path=args.output,
                learned_vs_rules_report=report.model_dump(mode="json"),
            )
        print(report.model_dump_json(indent=2))
        return

    if args.command == "sweep":
        report = run_sweep(
            train_examples=load_zria_examples(args.train),
            eval_examples=load_zria_examples(args.eval),
            output_dir=args.output_dir,
            backend_type=args.backend_type,
            spectral_k_values=args.spectral_k_values or [16, 32, 64, 128],
            dropout_values=args.dropout_values or [0.2, 0.5],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            seed=args.seed,
            patience=args.patience,
        )
        if args.output:
            write_sweep_report(report, args.output)
        if args.audit_path:
            recorder = AuditRecorder.from_output_path(args.audit_path, wwkg_enabled=args.audit_wwkg)
            recorder.record_validation_evidence(
                report_id=event_report_id(report.model_dump(mode="json")),
                evidence_type="learned_backend_sweep",
                per_backend_scores={
                    f"spectral_k={record.spectral_k},dropout={record.dropout}": {
                        "accuracy": record.eval_accuracy,
                        "average_score": record.average_score,
                        "epochs_trained": record.epochs_trained,
                        "best_epoch": record.best_epoch,
                    }
                    for record in report.records
                },
                artifact_path=args.output,
                learned_vs_rules_report=report.model_dump(mode="json"),
            )
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
