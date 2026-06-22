from __future__ import annotations

import ast
import io
import math
import multiprocessing as mp
from contextlib import redirect_stdout
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


def _worker(program: str, entrypoint: str, queue: Any) -> None:
    stdout = io.StringIO()
    try:
        tree = _validate_program(program, entrypoint)
        namespace = _execution_context()
        compiled = compile(tree, "<generated-pyro>", "exec")
        with redirect_stdout(stdout):
            exec(compiled, namespace, namespace)
            answer = namespace[entrypoint]()
        queue.put(
            {
                "success": True,
                "answer": _normalize_answer(answer),
                "stdout": stdout.getvalue(),
                "error": None,
                "timed_out": False,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "success": False,
                "answer": None,
                "stdout": stdout.getvalue(),
                "error": str(exc),
                "timed_out": False,
            }
        )


def _context() -> mp.context.BaseContext:
    for method in ("fork", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()


def execute_pyro_program(
    program: str,
    entrypoint: str = "run_inference",
    timeout_seconds: float = 5.0,
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

    ctx = _context()
    queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(program, entrypoint, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
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

    if queue.empty():
        return PyroExecutionResult(
            success=False,
            error="Generated program exited without returning a result.",
            control_events=[
                {
                    "control_type": "pyro_no_result",
                    "backend_type": "pyro",
                    "details": {"entrypoint": entrypoint},
                }
            ],
        )

    return PyroExecutionResult.model_validate(queue.get())
