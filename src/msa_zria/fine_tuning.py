from msa_zria.main import FineTuneConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

# Hugging Face + PEFT imports
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------ Fine-Tuning with Ray Tune ------------------------
def train_lora(config, checkpoint_dir=None):
    # Model & tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=config['lora_r'], lora_alpha=32,
        target_modules=["q_proj","v_proj"], lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
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
        optim="paged_adamw_8bit",
        fp16=True,
        gradient_checkpointing=True,
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
    tokenizer.save_pretrained(config['output_dir'])

def fine_tune(cfg: FineTuneConfig):
    ray_cfg = {
        'model_name': 'meta-llama/Llama-2-14b-hf',
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