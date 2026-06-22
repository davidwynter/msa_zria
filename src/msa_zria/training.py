from __future__ import annotations

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig

from msa_zria.config import ModelConfig, TrainingConfig


def preferred_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(model_config: ModelConfig) -> BitsAndBytesConfig | None:
    if not model_config.load_in_4bit:
        return None

    dtype = preferred_torch_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=model_config.use_double_quant,
        bnb_4bit_quant_type=model_config.quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=dtype,
    )


def load_model_and_processor(
    model_config: ModelConfig,
) -> tuple[AutoModelForMultimodalLM, AutoProcessor]:
    dtype = preferred_torch_dtype()
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "dtype": dtype,
    }

    quantization_config = build_quantization_config(model_config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForMultimodalLM.from_pretrained(
        model_config.base_model_id,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_config.processor_id)

    if model_config.prepare_for_kbit_training and quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def build_lora_config(training_config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=training_config.modules_to_save,
        ensure_weight_tying=True,
    )


def build_sft_config(training_config: TrainingConfig) -> SFTConfig:
    dtype = preferred_torch_dtype()

    return SFTConfig(
        output_dir=training_config.output_dir,
        max_length=training_config.max_length,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        eval_strategy=training_config.eval_strategy,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        report_to=training_config.report_to,
        dataset_kwargs={"skip_prepare_dataset": training_config.skip_prepare_dataset},
        remove_unused_columns=training_config.remove_unused_columns,
    )
