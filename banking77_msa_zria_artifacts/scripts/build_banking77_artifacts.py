#!/usr/bin/env python3
"""Build msa_zria artifacts from BANKING77-like source rows.

This file has two modes:
1. smoke: uses a small embedded, non-BANKING77 synthetic smoke corpus based on the BANKING77 intent taxonomy.
2. hf: downloads the real BANKING77 dataset through Hugging Face `datasets` and converts it.

The smoke corpus is only for schema/runtime testing. Use `--mode hf` for benchmark experiments.
"""
from __future__ import annotations

import argparse, hashlib, json, random, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SYSTEM = "You are a customer support reasoning assistant. Use the provided facts and return only valid JSON that matches the requested schema."

SMOKE_CASES = [
    ("cash_withdrawal", "The ATM took my card and cash did not come out. What should I do?", "debit_card", "cash withdrawal failed", "atm retained card", "high"),
    ("cash_withdrawal_card", "I tried withdrawing money but the machine kept my card.", "debit_card", "cash withdrawal card retained", "atm retained card", "high"),
    ("cash_withdrawal_cash", "The ATM says transaction complete but I received no cash.", "account", "cash withdrawal failed", "atm cash not dispensed", "high"),
    ("cash_withdrawal_charge", "I was charged for a cash withdrawal that failed.", "account", "cash withdrawal charge dispute", "failed withdrawal charge", "medium"),
    ("cash_withdrawal_pending", "My ATM withdrawal is still pending even though I got the cash.", "account", "pending cash withdrawal", "settlement delay", "medium"),
    ("card_acceptance", "Why was my card declined at a shop yesterday?", "card", "card declined", "merchant/card acceptance", "medium"),
    ("cash_withdrawal_limit", "Can I increase my ATM withdrawal limit for today?", "account", "withdrawal limit change", "limit request", "medium"),
    ("cash_withdrawal_verify", "How can I verify a suspicious cash withdrawal?", "account", "suspicious cash withdrawal", "possible fraud", "critical"),
    ("beneficiary_not_verified", "I added a new payee but they are not verified yet.", "transfer", "beneficiary not verified", "verification pending", "low"),
    ("card_arrival", "When will my new bank card arrive?", "card", "card arrival status", "delivery tracking", "low"),
    ("cash_withdrawal_statement", "Where can I find ATM withdrawals on my statement?", "statement", "statement lookup", "transaction search", "low"),
    ("beneficiary_activation", "How long does it take to activate a new beneficiary?", "transfer", "beneficiary activation", "activation window", "low"),
]

ESCALATE_PATTERNS = ["fraud", "suspicious", "stolen", "retained", "not dispensed", "charged", "dispute", "declined"]
INSUFFICIENT_PATTERNS = ["how can", "where can", "when will", "how long", "can i"]


def norm_label(label: Any, label_names: Optional[List[str]] = None) -> str:
    if isinstance(label, int) and label_names:
        return label_names[label]
    return str(label)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def decision_for(intent: str, issue: str, severity: str) -> Dict[str, Any]:
    text = f"{intent} {issue}".lower()
    if severity == "critical" or any(p in text for p in ESCALATE_PATTERNS):
        return {"task":"evaluate","verdict":"escalate","resolved":False,"should_escalate":True,"explanation":f"{intent} is treated as a risk or exception case and should be escalated for controlled handling."}
    if any(p in text for p in INSUFFICIENT_PATTERNS):
        return {"task":"evaluate","verdict":"insufficient_information","resolved":False,"should_escalate":False,"explanation":f"{intent} requires additional account-specific or policy-specific context before a final answer is safe."}
    return {"task":"evaluate","verdict":"resolved","resolved":True,"should_escalate":False,"explanation":f"{intent} can be handled through the standard banking support workflow."}


def parse_from(intent: str, text: str, explicit: Optional[Tuple[str,str,str,str]]=None) -> Dict[str, Any]:
    if explicit:
        device, issue, cause, severity = explicit
    else:
        device = "banking_service"
        issue = intent.replace("_", " ")
        lower = text.lower()
        cause = None
        if "atm" in lower: cause = "atm transaction context"
        elif "card" in lower: cause = "card service context"
        elif "transfer" in lower or "beneficiary" in lower or "payee" in lower: cause = "transfer workflow context"
        severity = "high" if any(p in lower for p in ["fraud", "stolen", "suspicious", "kept my card", "took my card", "no cash", "charged"]) else "medium"
        if lower.startswith(("when", "where", "how long")): severity = "low"
    return {"task":"parse","device":device,"issue":issue,"cause":cause,"severity":severity}


def code_target(parsed: Dict[str,Any], evaluation: Dict[str,Any]) -> Dict[str,Any]:
    var = "escalate" if evaluation["should_escalate"] else "resolved"
    value = "True" if (evaluation["should_escalate"] or evaluation["resolved"]) else "False"
    program = f"def run_inference():\n    {var} = {value}\n    return {var}"
    return {"task":"code","language":"python","framework":"pyro","entrypoint":"run_inference","query_variable":var,"required_statements":["return", var],"program":program}


def triples_for(intent: str, parsed: Dict[str, Any], evaluation: Dict[str, Any]) -> List[Dict[str,str]]:
    issue_entity = slug(parsed["issue"]).replace("-", "_")
    return [
        {"subject": parsed["device"], "predicate":"hasIntent", "object": intent},
        {"subject": parsed["device"], "predicate":"hasIssue", "object": issue_entity},
        {"subject": intent, "predicate":"recommendedVerdict", "object": evaluation["verdict"]},
        {"subject": intent, "predicate":"shouldEscalate", "object": str(evaluation["should_escalate"]).lower()},
    ]


def build_case(case_id: str, text: str, intent: str, split: str, explicit: Optional[Tuple[str,str,str,str]]=None) -> Dict[str,Any]:
    parsed = parse_from(intent, text, explicit)
    evaluation = decision_for(intent, parsed["issue"], parsed["severity"])
    code = code_target(parsed, evaluation)
    triples = triples_for(intent, parsed, evaluation)
    candidate_answer = "Route the case according to the banking intent policy and apply controlled escalation when risk signals are present."
    return {
        "case_id": case_id,
        "customer_message": text,
        "candidate_answer": candidate_answer,
        "triples": triples,
        "context": [f"BANKING77 intent: {intent}", f"Expected benchmark decision: {evaluation['verdict']}"],
        "parse_target": parsed,
        "code_target": code,
        "evaluation_target": evaluation,
        "metadata": {"split": split, "domain":"banking_intent", "source_dataset":"BANKING77", "intent": intent},
    }


def record_messages(case: Dict[str,Any], task: str) -> List[Dict[str,str]]:
    facts = "\n".join(f"{t['subject']} | {t['predicate']} | {t['object']}" for t in case["triples"])
    context = "\n".join([case["customer_message"], *case.get("context", [])])
    if task == "parse":
        prompt = f"Extract the device, issue, cause, and severity from the customer message. Return only valid JSON for the parse contract.\n\nFacts:\n{facts}\n\nContext:\n{context}"
        target = case["parse_target"]
    elif task == "code":
        prompt = f"Using the message facts and the parsed case state below, generate a Pyro reasoning program. Return only valid JSON for the code contract.\n\nParsed state: {case['parse_target']}\n\nFacts:\n{facts}\n\nContext:\n{context}"
        target = case["code_target"]
    else:
        prompt = f"Evaluate whether the candidate answer resolves the customer support case. Return only valid JSON for the evaluation contract.\n\nCustomer message: {case['customer_message']}\nCandidate answer: {case['candidate_answer']}\nParsed state: {case['parse_target']}\nFacts:\n{facts}\n\nContext:\n{context}"
        target = case["evaluation_target"]
    return [{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":json.dumps(target, sort_keys=True)}]


def dataset_records(cases: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    rows=[]
    for c in cases:
        for task, target_key in [("parse","parse_target"),("code","code_target"),("evaluate","evaluation_target")]:
            metadata = {**c.get("metadata",{}), "case_id": c["case_id"], "task": task, "input_mode":"hybrid"}
            kg_scope = c.get("kg_scope") or {}
            if kg_scope.get("workspace"):
                metadata["kg_workspace"] = kg_scope["workspace"]
            if kg_scope.get("branch"):
                metadata["kg_branch"] = kg_scope["branch"]
            rows.append({
                "example_id": f"{c['case_id']}-{task}-hybrid",
                "task": task,
                "input_mode": "hybrid",
                "messages": record_messages(c, task),
                "target": c[target_key],
                "metadata": metadata,
            })
    return rows


def zria_examples(cases: List[Dict[str,Any]], prefix: str) -> List[Dict[str,Any]]:
    out=[]
    for i,c in enumerate(cases,1):
        row = {
            "example_id": f"zria-{prefix}-{i:05d}",
            "query": c["customer_message"],
            "parsed": c["parse_target"],
            "neighborhood": c["triples"],
            "target": c["evaluation_target"],
            "metadata": {"case_id": c["case_id"], "intent": c["metadata"]["intent"], "source_dataset":"BANKING77"},
        }
        if c.get("kg_scope"):
            row["kg_scope"] = c["kg_scope"]
        out.append(row)
    return out


def ablation_cases(cases: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    rows = []
    for c in cases:
        row = {"case_id":c["case_id"],"query":c["customer_message"],"expected":c["evaluation_target"]}
        if c.get("kg_scope"):
            row["kg_scope"] = c["kg_scope"]
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str,Any]]) -> int:
    count=0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n")
            count += 1
    return count


def load_smoke_cases() -> List[Dict[str,Any]]:
    cases=[]
    for i,(intent,text,device,issue,cause,severity) in enumerate(SMOKE_CASES):
        split = "eval" if i % 5 == 0 else "train"
        cases.append(build_case(f"banking77-smoke-{i+1:04d}-{slug(intent)}", text, intent, split, (device, issue, cause, severity)))
    return cases


def load_hf_cases(max_train: int, max_eval: int, seed: int) -> List[Dict[str,Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("Install datasets first: pip install datasets") from e
    last = None
    ds = None
    for name in ["PolyAI/banking77", "banking77"]:
        try:
            ds = load_dataset(name)
            break
        except Exception as e:
            last = e
    if ds is None:
        raise SystemExit(f"Could not load BANKING77 from Hugging Face: {last}")
    label_names = None
    try:
        label_names = ds["train"].features["label"].names
    except Exception:
        pass
    rows=[]
    rnd = random.Random(seed)
    for split_name, limit in [("train", max_train), ("test", max_eval)]:
        split = ds[split_name]
        idxs = list(range(len(split)))
        rnd.shuffle(idxs)
        for j in idxs[:limit]:
            item = split[j]
            text = item.get("text") or item.get("utterance") or item.get("query") or item.get("sentence")
            intent = norm_label(item.get("label", item.get("intent")), label_names)
            rows.append(build_case(f"banking77-{split_name}-{j:05d}-{slug(intent)}", text, intent, "eval" if split_name=="test" else "train"))
    return rows


def rules_for(cases: List[Dict[str,Any]]) -> Dict[str,Any]:
    intents = {}
    for c in cases:
        intent = c["metadata"]["intent"]
        if intent not in intents:
            intents[intent]=c["evaluation_target"]
    rules=[]
    for intent,outcome in sorted(intents.items()):
        rules.append({"rule_id": f"banking77-{slug(intent)}", "branch": BRANCH, "keywords": intent.replace("_", " ").split(), "outcome": outcome})
    return {"version":"1", "rules":rules, "default_outcome":{"task":"evaluate","verdict":"insufficient_information","resolved":False,"should_escalate":False,"explanation":"No BANKING77 intent rule matched this case."}}


def sha256(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke","hf"], default="smoke")
    ap.add_argument("--out", type=Path, default=Path("examples"))
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--max-eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = load_smoke_cases() if args.mode == "smoke" else load_hf_cases(args.max_train, args.max_eval, args.seed)
    train_cases=[c for c in cases if c["metadata"]["split"]=="train"]
    eval_cases=[c for c in cases if c["metadata"]["split"]=="eval"]
    files={}
    files["customer_support_cases"] = write_jsonl(args.out/"banking77_customer_support_cases.jsonl", cases)
    files["customer_support_records_train"] = write_jsonl(args.out/"banking77_customer_support_records_train.jsonl", dataset_records(train_cases))
    files["customer_support_records_eval"] = write_jsonl(args.out/"banking77_customer_support_records_eval.jsonl", dataset_records(eval_cases))
    files["zria_examples_train"] = write_jsonl(args.out/"banking77_zria_examples_train.jsonl", zria_examples(train_cases, "train"))
    files["zria_examples_eval"] = write_jsonl(args.out/"banking77_zria_examples_eval.jsonl", zria_examples(eval_cases, "eval"))
    files["ablation_cases"] = write_jsonl(args.out/"banking77_ablation_cases.jsonl", ablation_cases(eval_cases or cases[:10]))
    (args.out/"banking77_zria_rules.json").write_text(json.dumps(rules_for(cases), indent=2, sort_keys=True), encoding="utf-8")
    files["zria_rules"] = 1
    manifest={"dataset":"BANKING77", "mode":args.mode, "workspace":WORKSPACE, "branch":BRANCH, "counts":files, "files":{p.name: sha256(p) for p in sorted(args.out.glob("banking77_*"))}}
    (args.out/"banking77_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
