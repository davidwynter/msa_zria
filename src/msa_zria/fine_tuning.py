from msa_zria.main import FineTuneConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from msa_zria.config import ModelConfig
from msa_zria.training import (
    detect_accelerator,
    load_model_and_processor,
    preferred_optimizer,
    preferred_torch_dtype,
)

# Hugging Face + PEFT imports
import torch
from transformers import (
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ------------------------ Fine-Tuning with Ray Tune ------------------------
def train_lora(config, checkpoint_dir=None):
    accelerator = detect_accelerator(config["accelerator"])
    torch_dtype = preferred_torch_dtype(accelerator)
    model_config = ModelConfig(
        base_model_id=config["model_name"],
        processor_id=config["processor_name"],
        quantization_bits=config["quantization_bits"],
        quantization_backend=config["quantization_backend"],
    )
    model, processor = load_model_and_processor(model_config, accelerator=accelerator)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=config['lora_r'], lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, lora_cfg)

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset('json', data_files=config['dataset_path'], split='train')
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)
    tokenized = ds.map(tokenize, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=16,
        learning_rate=config['learning_rate'],
        optim=preferred_optimizer(accelerator),
        fp16=accelerator != "cpu" and torch_dtype == torch.float16,
        bf16=accelerator != "cpu" and torch_dtype == torch.bfloat16,
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=50,
        save_strategy="no",
        seed=config['seed']
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized, data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(config['output_dir'])
    processor.save_pretrained(config['output_dir'])

def fine_tune(cfg: FineTuneConfig):
    ray_cfg = {
        'model_name': 'google/gemma-4-12B',
        'processor_name': 'google/gemma-4-12B-it',
        'accelerator': cfg.accelerator,
        'quantization_bits': cfg.quantization_bits,
        'quantization_backend': cfg.quantization_backend,
        'gradient_checkpointing': cfg.gradient_checkpointing,
        'dataset_path': cfg.dataset_path,
        'output_dir': cfg.output_dir,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.learning_rate,
        'lora_r': cfg.lora_r,
        'seed': cfg.seed
    }
    scheduler = ASHAScheduler(metric="training_iteration", mode="max")
    reporter = CLIReporter(metric_columns=["training_iteration"])
    analysis = tune.run(
        train_lora,
        config=ray_cfg,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter
    )
    best = analysis.get_best_trial(metric="training_iteration", mode="max")
    return {'best_checkpoint': best.checkpoint.value}
