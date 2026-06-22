from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder, event_report_id, sha256_directory, sha256_json, stable_dataset_version
from msa_zria.config import KGScope
from msa_zria.data import EvaluationCase, EvaluationResult, EvaluationTarget, ParseTarget, evaluate_case

if TYPE_CHECKING:
    import torch

_PAD = "<pad>"
_UNK = "<unk>"


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


class ZRIALearnedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    vocab: dict[str, int]
    labels: list[EvaluationTarget]
    max_length: int
    embedding_dim: int
    hidden_dim: int
    state_dict_path: str


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_:/.-]+", text.lower())


def build_vocab(examples: list[ZRIAExample]) -> dict[str, int]:
    vocab = {_PAD: 0, _UNK: 1}
    for example in examples:
        for token in tokenize(render_feature_text(example.query, example.parsed, example.kg_scope)):
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
    serialized = json.dumps(target.model_dump(), sort_keys=True)
    for index, label in enumerate(labels):
        if serialized == json.dumps(label.model_dump(), sort_keys=True):
            return index
    raise ValueError("Encountered target that is not present in the label set.")


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


def create_model(
    vocab: dict[str, int],
    labels: list[EvaluationTarget],
    embedding_dim: int,
    hidden_dim: int,
) -> Any:
    _, nn, _ = _require_torch()

    class ZRIALearnedClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(labels)),
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            embeddings = self.embedding(input_ids)
            masked = embeddings * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.classifier(pooled)

    return ZRIALearnedClassifier()


def train_model(
    train_examples: list[ZRIAExample],
    output_dir: str | Path,
    *,
    eval_examples: list[ZRIAExample] | None = None,
    epochs: int = 80,
    learning_rate: float = 1e-2,
    max_length: int = 64,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    torch, nn, DataLoader = _require_torch()
    torch.manual_seed(seed)
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(train_examples)
    labels = build_label_set(train_examples)
    model = create_model(vocab, labels, embedding_dim, hidden_dim)
    dataset = _ZRIADataset(train_examples, vocab, labels, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    state_dict_path = output_path / "model.pt"
    torch.save(model.state_dict(), state_dict_path)
    artifact = ZRIALearnedArtifact(
        vocab=vocab,
        labels=labels,
        max_length=max_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        state_dict_path=str(state_dict_path),
    )
    with (output_path / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact.model_dump(mode="json"), handle, indent=2)

    metrics = {
        "train_examples": len(train_examples),
        "labels": len(labels),
    }
    if eval_examples:
        report = evaluate_learned_backend(output_path, eval_examples)
        metrics["eval_accuracy"] = report.summaries[0].accuracy
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
    model = create_model(artifact.vocab, artifact.labels, artifact.embedding_dim, artifact.hidden_dim)
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.eval()
    return artifact, model


def predict_with_model(
    artifact: ZRIALearnedArtifact,
    model: Any,
    query: str,
    parsed: ParseTarget,
    kg_scope: KGScope | None = None,
) -> tuple[EvaluationTarget, float]:
    torch, _, _ = _require_torch()
    feature_text = render_feature_text(query, parsed, kg_scope)
    input_ids = torch.tensor([encode_text(feature_text, artifact.vocab, artifact.max_length)], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1 if token_id != artifact.vocab[_PAD] else 0 for token_id in input_ids[0].tolist()]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
    confidence, label_idx = torch.max(probabilities, dim=-1)
    return artifact.labels[int(label_idx.item())], float(confidence.item())


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
                backend="learned",
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
                backend="learned",
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
    from msa_zria.zria_backend import LearnedZRIABackend, RuleBasedZRIABackend

    learned_artifact, learned_model = load_artifact(learned_model_path)
    backends = [
        (
            "rules",
            RuleBasedZRIABackend.from_path(rules_path),
            None,
        ),
        (
            "learned",
            LearnedZRIABackend.from_path(
                learned_model_path,
                confidence_threshold=confidence_threshold,
                fallback_backend=None,
            ),
            (learned_artifact, learned_model),
        ),
    ]
    details: list[ZRIAComparisonDetail] = []
    for backend_name, backend, learned_bundle in backends:
        for example in examples:
            predicted = backend.predict(example.query, parsed=example.parsed, kg_scope=example.kg_scope)
            confidence = None
            if backend_name == "learned":
                artifact, model = learned_bundle
                _, confidence = predict_with_model(artifact, model, example.query, example.parsed, example.kg_scope)
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
                    backend=backend_name,
                    confidence=confidence,
                    predicted=predicted,
                    evaluation=evaluation,
                )
            )

    summaries: list[ZRIAComparisonSummary] = []
    for backend_name in ["rules", "learned"]:
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


def write_comparison_report(report: ZRIAComparisonReport, output_path: str | Path) -> None:
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
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=1e-2)
    train_parser.add_argument("--max-length", type=int, default=64)
    train_parser.add_argument("--embedding-dim", type=int, default=32)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
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

    args = parser.parse_args()

    if args.command == "train":
        train_examples = load_zria_examples(args.train)
        eval_examples = load_zria_examples(args.eval) if args.eval else None
        metrics = train_model(
            train_examples,
            args.output,
            eval_examples=eval_examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.audit_path:
            recorder = AuditRecorder.from_output_path(args.audit_path, wwkg_enabled=args.audit_wwkg)
            recorder.record_model_lineage(
                experiment_config_hash=sha256_json(
                    {
                        "command": "zria_train",
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "max_length": args.max_length,
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                    }
                ),
                training_dataset_version=stable_dataset_version(
                    [args.train, args.eval] if args.eval else [args.train]
                ),
                model_artifact_path=args.output,
                model_artifact_hash=sha256_directory(args.output),
                backend_type="learned",
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


if __name__ == "__main__":
    main()
