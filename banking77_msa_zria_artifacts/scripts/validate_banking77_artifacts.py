#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

REQUIRED = {
    "cases": "banking77_customer_support_cases.jsonl",
    "records_train": "banking77_customer_support_records_train.jsonl",
    "records_eval": "banking77_customer_support_records_eval.jsonl",
    "zria_train": "banking77_zria_examples_train.jsonl",
    "zria_eval": "banking77_zria_examples_eval.jsonl",
    "ablation": "banking77_ablation_cases.jsonl",
    "rules": "banking77_zria_rules.json",
}

def load_jsonl(path: Path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except Exception as e: raise AssertionError(f"{path}:{i} invalid JSONL: {e}")
    return rows

def require(cond, msg):
    if not cond: raise AssertionError(msg)

def main(root: str="examples"):
    root=Path(root)
    for name,file in REQUIRED.items():
        require((root/file).exists(), f"missing {file}")
    cases=load_jsonl(root/REQUIRED["cases"])
    train=load_jsonl(root/REQUIRED["records_train"])
    eval_=load_jsonl(root/REQUIRED["records_eval"])
    ztrain=load_jsonl(root/REQUIRED["zria_train"])
    zeval=load_jsonl(root/REQUIRED["zria_eval"])
    ablation=load_jsonl(root/REQUIRED["ablation"])
    rules=json.loads((root/REQUIRED["rules"]).read_text(encoding="utf-8"))
    for c in cases:
        for k in ["case_id","customer_message","triples","kg_scope","parse_target","code_target","evaluation_target"]:
            require(k in c, f"case missing {k}: {c.get('case_id')}")
    for r in train+eval_:
        require(r.get("task") in {"parse","code","evaluate"}, f"bad task {r.get('task')}")
        require(len(r.get("messages",[])) == 3, f"bad message count {r.get('example_id')}")
        require(r["messages"][-1]["role"] == "assistant", f"assistant target missing {r.get('example_id')}")
        json.loads(r["messages"][-1]["content"])
    for z in ztrain+zeval:
        require("neighborhood" in z and z["neighborhood"], f"zria neighborhood missing {z.get('example_id')}")
        require(z["target"].get("task") == "evaluate", f"zria target bad {z.get('example_id')}")
    require(rules.get("rules"), "rules empty")
    print(json.dumps({"ok": True, "case_count": len(cases), "train_records": len(train), "eval_records": len(eval_), "zria_train": len(ztrain), "zria_eval": len(zeval), "ablation_cases": len(ablation), "rules": len(rules['rules'])}, indent=2))

if __name__ == "__main__": main(sys.argv[1] if len(sys.argv)>1 else "examples")
