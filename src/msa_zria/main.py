import os
import json
from msa_zria.dspy_modules import CodeGenModule, EvalModule, ParseModule, pyro_interpreter
import dspy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import ray

# KG imports
from oxigraph import Graph, MemoryStore

# Pyro for probabilistic reasoning
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, EmpiricalMarginal

# Ray initialization
ray.init(ignore_reinit_error=True)

app = FastAPI(title="MSA Reasoning Service")

# ------------------------ Pydantic Schemas ------------------------
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

class DatasetConfig(BaseModel):
    graph_path: str
    output_path: str
    format: str  # 'triples', 'nl', or 'hybrid'

class FineTuneConfig(BaseModel):
    dataset_path: str
    output_dir: str
    epochs: int = 3
    lora_r: int = 16
    learning_rate: float = 2e-4
    batch_size: int = 1
    seed: int = 42

class InferenceRequest(BaseModel):
    query: str
    mode: str  # 'zria', 'pyro', 'hybrid'

class ParseRequest(BaseModel):
    text: str

class CodeGenRequest(BaseModel):
    parsed: Dict[str, Any]

class EvalRequest(BaseModel):
    query: str
    answer: Any

# ------------------------ Utility Functions ------------------------
def safe_run_module(module, *args):
    try:
        return module(*args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module execution error: {e}")
    
# ------------------------ Oxigraph Dataset Production ------------------------
@app.post("/produce_dataset")
def produce_dataset(cfg: DatasetConfig):
    store = MemoryStore()
    graph = Graph(store)
    graph.load_file(cfg.graph_path, format="nquads")
    triples = list(graph.quads())
    os.makedirs(cfg.output_path, exist_ok=True)
    # Write triples
    if cfg.format in ['triples', 'hybrid']:
        with open(os.path.join(cfg.output_path, 'triples.jsonl'), 'w') as f:
            for s, p, o in triples:
                f.write(json.dumps({
                    'subject': str(s), 'predicate': str(p), 'object': str(o)
                }) + '\n')
    # Write NL sentences
    if cfg.format in ['nl', 'hybrid']:
        with open(os.path.join(cfg.output_path, 'nl.jsonl'), 'w') as f:
            for s, p, o in triples:
                sentence = f"{s} {p} {o}."
                f.write(json.dumps({'text': sentence}) + '\n')
    return {'status': 'dataset_produced', 'count': len(triples)}


# ------------------------ Inference and MSA Endpoints ------------------------
# Load fine-tuned LM for inference in DSPy modules
LM_PATH = os.getenv('LM_PATH', 'outputs/llama2-finetune')
llm = dspy.LM(model=LM_PATH)

# Instantiate DSPy modules with the loaded LM
parse_module = ParseModule(llm=llm)
code_module  = CodeGenModule(llm=llm)
eval_module  = EvalModule(llm=llm)

@app.post("/parse")
def parse(req: ParseRequest):
    result = parse_module(req.text)
    return {'parsed': result['parsed_result']}

@app.post("/code_synthesis")
def code_synthesis(req: CodeGenRequest):
    result = code_module(req.parsed)
    return {'code': result['code_str']}

@app.post("/run_pyro")
def run_pyro(req: CodeGenRequest):
    # Execute the generated Pyro code string
    code_str = code_module(req.parsed)['code_str']
    output = pyro_interpreter.run(code_str + "\nprint(run_inference())")
    return {'pyro_result': output}

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    result = eval_module(req.query, req.answer)
    return {'evaluation': result['evaluation']}

@app.post("/infer")
def inference(req: InferenceRequest):
    if req.mode == 'pyro':
        # Example direct Pyro inference
        return run_pyro(CodeGenRequest(parsed=parse_module(req.query)['parsed_result']))
    elif req.mode == 'zria':
        answer = zria_predict(req.query)
        return {'answer': answer}
    elif req.mode == 'hybrid':
        z = zria_predict(req.query)
        pyro_out = run_pyro(CodeGenRequest(parsed=parse_module(req.query)['parsed_result']))['pyro_result']
        # Simple merge: prefer Pyro if boolean, else ZRIA
        hybrid_ans = pyro_out if isinstance(pyro_out, bool) else z
        return {'answer': hybrid_ans}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

@app.post("/fine_tune")
def fine_tune(cfg: FineTuneConfig):
    return fine_tune(cfg)

# Placeholder ZRIA function
def zria_predict(query: str) -> bool:
    # TODO: replace with actual ZRIA model invocation
    return True

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
