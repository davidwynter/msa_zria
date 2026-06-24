from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, parse, request

from msa_zria.config import KGConfig, KGScope
from msa_zria.data import ParseTarget, Triple


def load_triples(config: KGConfig) -> list[Triple]:
    if config.backend == "oxigraph":
        return _load_triples_from_oxigraph(config)
    if config.backend == "wwkg":
        return _load_triples_from_wwkg(config)
    raise ValueError(f"Unsupported KG backend '{config.backend}'.")


def kg_context_metadata(scope: KGScope | KGConfig | None) -> dict[str, str]:
    if scope is None:
        return {}
    return scope.to_metadata()


def retrieve_neighborhood(
    config: KGConfig,
    query: str,
    parsed: ParseTarget | None,
    *,
    limit: int = 64,
) -> list[Triple]:
    terms = _candidate_terms(query, parsed)
    if not terms:
        return []
    if config.backend == "oxigraph":
        return _retrieve_neighborhood_from_oxigraph(config, terms, limit=limit)
    if config.backend == "wwkg":
        return _retrieve_neighborhood_from_wwkg(config, terms, limit=limit)
    raise ValueError(f"Unsupported KG backend '{config.backend}'.")


def _load_triples_from_oxigraph(config: KGConfig) -> list[Triple]:
    if not config.graph_path:
        raise ValueError("The oxigraph backend requires kg.graph_path.")

    try:
        from oxigraph import Graph, MemoryStore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The oxigraph backend requires the optional 'oxigraph' dependency."
        ) from exc

    store = MemoryStore()
    graph = Graph(store)
    graph.load_file(config.graph_path, format=config.graph_format)
    return [
        Triple(subject=str(subject), predicate=str(predicate), object=str(obj))
        for subject, predicate, obj in graph.quads()
    ]


def _load_triples_from_wwkg(config: KGConfig) -> list[Triple]:
    payload = _wwkg_query(config, config.resolved_sparql_query())
    if not isinstance(payload, dict):
        raise RuntimeError("WWKG returned a non-JSON payload for a SPARQL query.")

    bindings = payload.get("results", {}).get("bindings", [])
    triples: list[Triple] = []
    for row in bindings:
        triples.append(
            Triple(
                subject=_binding_value(row, "subject"),
                predicate=_binding_value(row, "predicate"),
                object=_binding_value(row, "object"),
            )
        )
    return triples


def _retrieve_neighborhood_from_oxigraph(
    config: KGConfig,
    terms: list[str],
    *,
    limit: int,
) -> list[Triple]:
    triples = load_triples(config)
    matches: list[Triple] = []
    lowered_terms = [term.lower() for term in terms]
    for triple in triples:
        haystack = f"{triple.subject} {triple.predicate} {triple.object}".lower()
        if any(term in haystack for term in lowered_terms):
            matches.append(triple)
            if len(matches) >= limit:
                break
    return matches


def _retrieve_neighborhood_from_wwkg(
    config: KGConfig,
    terms: list[str],
    *,
    limit: int,
) -> list[Triple]:
    payload = _wwkg_query(config, _neighborhood_query(config, terms, limit=limit))
    if not isinstance(payload, dict):
        raise RuntimeError("WWKG returned a non-JSON payload for a neighborhood query.")

    bindings = payload.get("results", {}).get("bindings", [])
    return [
        Triple(
            subject=_binding_value(row, "subject"),
            predicate=_binding_value(row, "predicate"),
            object=_binding_value(row, "object"),
        )
        for row in bindings
    ]


def _candidate_terms(query: str, parsed: ParseTarget | None) -> list[str]:
    candidates: list[str] = []
    if parsed is not None:
        candidates.extend([parsed.device, parsed.issue])
        if parsed.cause:
            candidates.append(parsed.cause)
        if parsed.severity:
            candidates.append(parsed.severity)
    candidates.extend(query.split())

    terms: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        term = raw.strip().lower()
        if len(term) < 3:
            continue
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 8:
            break
    return terms


def _neighborhood_query(config: KGConfig, terms: list[str], *, limit: int) -> str:
    filters = [
        (
            f"CONTAINS(LCASE(STR(?subject)), {_sparql_string(term)}) || "
            f"CONTAINS(LCASE(STR(?predicate)), {_sparql_string(term)}) || "
            f"CONTAINS(LCASE(STR(?object)), {_sparql_string(term)})"
        )
        for term in terms
    ]
    graph_clause_open = ""
    graph_clause_close = ""
    if config.graph_iri:
        graph_clause_open = f"GRAPH <{config.graph_iri}> {{ "
        graph_clause_close = " }"
    return (
        "SELECT ?subject ?predicate ?object WHERE { "
        f"{graph_clause_open}?subject ?predicate ?object . "
        f"FILTER ({' || '.join(filters)})"
        f"{graph_clause_close} "
        f"}} LIMIT {int(limit)}"
    )


def _sparql_string(value: str) -> str:
    return json.dumps(value)


def _binding_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key, {})
    if not isinstance(value, dict) or "value" not in value:
        raise RuntimeError(f"WWKG SPARQL row is missing binding '{key}'.")
    return str(value["value"])


def _wwkg_query(config: KGConfig, sparql: str) -> Any:
    url = _build_url(config.base_url, "/sparql")
    headers = _wwkg_headers(config)
    headers.update(
        {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json, application/json, text/plain",
        }
    )
    body = sparql.encode("utf-8")

    for attempt in range(config.retry_attempts + 1):
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                raw_body = response.read()
                return _decode_payload(raw_body, response.headers.get("Content-Type", ""))
        except error.HTTPError as exc:
            if attempt < config.retry_attempts and exc.code in {408, 429, 500, 502, 503, 504}:
                _sleep_before_retry(config, attempt)
                continue
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"WWKG request failed with HTTP {exc.code}: {body_text}") from exc
        except error.URLError as exc:
            if attempt < config.retry_attempts:
                _sleep_before_retry(config, attempt)
                continue
            raise RuntimeError(f"WWKG request failed: {exc.reason}") from exc


def _wwkg_headers(config: KGConfig) -> dict[str, str]:
    headers = {"User-Agent": config.user_agent}
    if config.api_key:
        headers["X-WWKG-API-Key"] = config.api_key
    if config.workspace:
        headers["X-WWKG-Workspace"] = config.workspace
    if config.branch:
        headers["X-WWKG-Branch"] = config.branch
    if config.commit:
        headers["X-WWKG-Commit"] = config.commit
    if config.as_of:
        headers["X-WWKG-AsOf"] = config.as_of
    return headers


def _build_url(base_url: str, path: str) -> str:
    parsed = parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("kg.base_url must be an absolute http(s) URL for the wwkg backend.")
    return f"{base_url.rstrip('/')}{path}"


def _decode_payload(raw_body: bytes, content_type: str) -> Any:
    text = raw_body.decode("utf-8", errors="replace")
    if "json" in content_type.lower():
        if not text.strip():
            return {}
        return json.loads(text)
    return text


def _sleep_before_retry(config: KGConfig, attempt: int) -> None:
    delay = config.retry_backoff_seconds * (2**attempt)
    if delay > 0:
        time.sleep(delay)
