from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_ALLOWED_IMPORTS = {"math", "pyro", "pyro.distributions", "torch"}
_ALLOWED_AST_NODES = (
    ast.Module,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.Return,
    ast.Expr,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Attribute,
    ast.Constant,
    ast.If,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.keyword,
)
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class PyroExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    answer: Any = None
    stdout: str = ""
    error: str | None = None
    timed_out: bool = False
    control_events: list[dict[str, Any]] = Field(default_factory=list)


def _normalize_answer(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalize_answer(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_answer(item) for item in value]
    return value


def _validate_program(program: str, entrypoint: str) -> ast.AST:
    tree = ast.parse(program, mode="exec")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            exc = ValueError(f"Disallowed syntax in generated program: {type(node).__name__}")
            exc.audit_code = "disallowed_ast_syntax"
            raise exc
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    exc = ValueError(f"Disallowed import in generated program: {alias.name}")
                    exc.audit_code = "disallowed_import"
                    raise exc
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in _ALLOWED_IMPORTS:
                exc = ValueError(f"Disallowed import in generated program: {module}")
                exc.audit_code = "disallowed_import"
                raise exc
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            exc = ValueError("Dunder attribute access is not allowed in generated programs.")
            exc.audit_code = "disallowed_dunder_access"
            raise exc

    if not any(isinstance(node, ast.FunctionDef) and node.name == entrypoint for node in tree.body):
        exc = ValueError(f"Expected program to define '{entrypoint}'.")
        exc.audit_code = "missing_entrypoint"
        raise exc
    return tree


def _execution_context() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "math": math}
    try:
        import torch

        namespace["torch"] = torch
    except ModuleNotFoundError:
        pass
    try:
        import pyro
        import pyro.distributions as dist

        namespace["pyro"] = pyro
        namespace["dist"] = dist
    except ModuleNotFoundError:
        pass
    return namespace


def _apply_resource_limits(timeout_seconds: float, memory_limit_mb: int | None) -> None:
    if os.name != "posix":
        return
    try:
        import resource
    except ModuleNotFoundError:
        return

    cpu_limit = max(1, int(math.ceil(timeout_seconds)))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    if memory_limit_mb is not None:
        memory_limit_bytes = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def _execute_in_worker(
    program: str,
    entrypoint: str,
    *,
    timeout_seconds: float,
    memory_limit_mb: int | None,
) -> dict[str, Any]:
    _apply_resource_limits(timeout_seconds, memory_limit_mb)
    stdout = io.StringIO()
    try:
        tree = _validate_program(program, entrypoint)
        namespace = _execution_context()
        compiled = compile(tree, "<generated-pyro>", "exec")
        with redirect_stdout(stdout):
            exec(compiled, namespace, namespace)
            answer = namespace[entrypoint]()
        return {
            "success": True,
            "answer": _normalize_answer(answer),
            "stdout": stdout.getvalue(),
            "error": None,
            "timed_out": False,
            "control_events": [],
        }
    except Exception as exc:
        return {
            "success": False,
            "answer": None,
            "stdout": stdout.getvalue(),
            "error": str(exc),
            "timed_out": False,
            "control_events": [
                {
                    "control_type": "pyro_worker_error",
                    "backend_type": "pyro",
                    "details": {"entrypoint": entrypoint, "error": str(exc)},
                }
            ],
        }


def _write_worker_result(output_path: str, result: dict[str, Any]) -> None:
    Path(output_path).write_text(json.dumps(result), encoding="utf-8")


def _worker_main(output_path: str) -> int:
    payload = json.loads(sys.stdin.read())
    result = _execute_in_worker(
        payload["program"],
        payload["entrypoint"],
        timeout_seconds=payload["timeout_seconds"],
        memory_limit_mb=payload.get("memory_limit_mb"),
    )
    _write_worker_result(output_path, result)
    return 0


def execute_pyro_program(
    program: str,
    entrypoint: str = "run_inference",
    timeout_seconds: float = 5.0,
    memory_limit_mb: int | None = 1024,
) -> PyroExecutionResult:
    try:
        _validate_program(program, entrypoint)
    except Exception as exc:
        return PyroExecutionResult(
            success=False,
            error=str(exc),
            control_events=[
                {
                    "control_type": getattr(exc, "audit_code", "validation_error"),
                    "backend_type": "pyro",
                    "details": {"entrypoint": entrypoint, "error": str(exc)},
                }
            ],
        )

    with tempfile.NamedTemporaryFile(prefix="msa_zria_pyro_", suffix=".json", delete=False) as handle:
        output_path = handle.name

    worker_payload = json.dumps(
        {
            "program": program,
            "entrypoint": entrypoint,
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
        }
    )
    command = [sys.executable, "-m", "msa_zria.pyro_runtime", "--worker-output", output_path]
    try:
        completed = subprocess.run(
            command,
            input=worker_payload,
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 1.0,
            env=dict(os.environ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        _cleanup_output_file(output_path)
        return PyroExecutionResult(
            success=False,
            error=f"Generated program exceeded timeout of {timeout_seconds} seconds.",
            timed_out=True,
            control_events=[
                {
                    "control_type": "pyro_timeout",
                    "backend_type": "pyro",
                    "details": {"timeout_seconds": timeout_seconds, "entrypoint": entrypoint},
                }
            ],
        )

    if not Path(output_path).exists() or not Path(output_path).read_text(encoding="utf-8").strip():
        _cleanup_output_file(output_path)
        return PyroExecutionResult(
            success=False,
            error=(
                "Generated program worker exited without returning a result."
                if completed.returncode == 0
                else f"Generated program worker failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            ),
            control_events=[
                {
                    "control_type": "pyro_subprocess_failure",
                    "backend_type": "pyro",
                    "details": {
                        "entrypoint": entrypoint,
                        "returncode": completed.returncode,
                        "stderr": completed.stderr.strip(),
                    },
                }
            ],
        )

    try:
        result_payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    finally:
        _cleanup_output_file(output_path)

    return PyroExecutionResult.model_validate(result_payload)


def _cleanup_output_file(output_path: str) -> None:
    try:
        Path(output_path).unlink(missing_ok=True)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Pyro runtime worker.")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker_output:
        raise SystemExit(_worker_main(args.worker_output))
    parser.error("This module is intended to be used via execute_pyro_program().")


if __name__ == "__main__":
    main()
