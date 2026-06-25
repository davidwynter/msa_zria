.PHONY: validate smoke hf train zria ablation

validate:
	python scripts/validate_banking77_artifacts.py examples

smoke:
	python scripts/build_banking77_artifacts.py --mode smoke --out examples
	python scripts/validate_banking77_artifacts.py examples

hf:
	python scripts/build_banking77_artifacts.py --mode hf --out examples --max-train 3000 --max-eval 500
	python scripts/validate_banking77_artifacts.py examples

train:
	python -m msa_zria.train --config configs/gemma4_12b_banking77.yaml

zria:
	python -m msa_zria.zria train --train examples/banking77_zria_examples_train.jsonl --eval examples/banking77_zria_examples_eval.jsonl --backend-type learned_graph --spectral-k 16 --dropout 0.2 --graph-layers 2 --harmonic-reg-weight 5e-4 --pretrain-epochs 10 --memory-momentum 0.9 --patience 10 --output outputs/zria_graph_banking77

ablation:
	python -m msa_zria.ablation --config configs/gemma4_12b_banking77.yaml
