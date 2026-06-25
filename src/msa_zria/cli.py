from __future__ import annotations

import argparse
import json
import sys

from msa_zria.config import KGScope
from msa_zria.runtime import (
    BranchConfigurationError,
    InferenceExecutionError,
    InferenceModuleNotConfiguredError,
    RuntimeDependencies,
    UnsupportedModeError,
    available_reasoning_branches,
    build_runtime_dependencies,
    infer,
)
from msa_zria.thinking import ingest_thinking_cases


def _kg_scope_from_args(args: argparse.Namespace) -> KGScope | None:
    if not any([args.kg_workspace, args.kg_branch, args.kg_commit, args.kg_as_of]):
        return None
    return KGScope(
        workspace=args.kg_workspace,
        branch=args.kg_branch,
        commit=args.kg_commit,
        as_of=args.kg_as_of,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for selecting msa_zria reasoning branches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    branches_parser = subparsers.add_parser("branches", help="List configured reasoning branches.")
    branches_parser.set_defaults(handler=_run_branches)

    infer_parser = subparsers.add_parser("infer", help="Run one inference request.")
    infer_parser.add_argument("--query", required=True, help="Input query text.")
    infer_parser.add_argument(
        "--mode",
        required=True,
        choices=["pyro", "zria", "hybrid"],
        help="Inference mode to run.",
    )
    infer_parser.add_argument(
        "--reasoning-branch",
        default="non_thinking",
        choices=["non_thinking", "thinking"],
        help="Which reasoning branch to use.",
    )
    infer_parser.add_argument("--kg-workspace", help="Optional KG workspace.")
    infer_parser.add_argument("--kg-branch", help="Optional KG branch.")
    infer_parser.add_argument("--kg-commit", help="Optional KG commit.")
    infer_parser.add_argument("--kg-as-of", help="Optional KG as-of timestamp.")
    infer_parser.set_defaults(handler=_run_infer)

    thinking_ingest_parser = subparsers.add_parser(
        "thinking-ingest",
        help="Build canonical training records for the specialist thinking branch.",
    )
    thinking_ingest_parser.add_argument("--input", required=True, help="Path to specialist thinking source cases.")
    thinking_ingest_parser.add_argument("--output", required=True, help="Path to output specialist training records.")
    thinking_ingest_parser.add_argument(
        "--input-mode",
        action="append",
        choices=["triples", "text", "hybrid"],
        dest="input_modes",
        help="One or more input modes to emit. Defaults to hybrid.",
    )
    thinking_ingest_parser.add_argument("--kg-workspace", help="Optional KG workspace override.")
    thinking_ingest_parser.add_argument("--kg-branch", help="Optional KG branch override.")
    thinking_ingest_parser.add_argument("--kg-commit", help="Optional KG commit override.")
    thinking_ingest_parser.add_argument("--kg-as-of", help="Optional KG as-of override.")
    thinking_ingest_parser.set_defaults(handler=_run_thinking_ingest)

    return parser


def _run_branches(_: argparse.Namespace, runtime: RuntimeDependencies | None = None) -> int:
    runtime = runtime or build_runtime_dependencies()
    payload = {
        "available": available_reasoning_branches(runtime),
        "default": "non_thinking",
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


def _run_infer(args: argparse.Namespace, runtime: RuntimeDependencies | None = None) -> int:
    runtime = runtime or build_runtime_dependencies()
    payload = infer(
        runtime,
        args.query,
        mode=args.mode,
        kg_scope=_kg_scope_from_args(args),
        reasoning_branch=args.reasoning_branch,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


def _run_thinking_ingest(args: argparse.Namespace, runtime: RuntimeDependencies | None = None) -> int:
    del runtime
    count = ingest_thinking_cases(
        args.input,
        args.output,
        input_modes=args.input_modes,
        kg_scope=_kg_scope_from_args(args),
    )
    print(json.dumps({"output": args.output, "records_written": count}, ensure_ascii=True, sort_keys=True))
    return 0


def main(argv: list[str] | None = None, runtime: RuntimeDependencies | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args, runtime=runtime)
    except (
        BranchConfigurationError,
        InferenceExecutionError,
        InferenceModuleNotConfiguredError,
        UnsupportedModeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
