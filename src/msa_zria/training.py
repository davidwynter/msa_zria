from __future__ import annotations

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig, HqqConfig
from trl import SFTConfig

from msa_zria.config import ModelConfig, TrainingConfig


def xpu_is_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def detect_accelerator(requested: str = "auto") -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if xpu_is_available():
            return "xpu"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no NVIDIA CUDA device is available.")
        return "cuda"

    if requested == "xpu":
        if not xpu_is_available():
            raise RuntimeError("XPU was requested but no Intel XPU device is available.")
        return "xpu"

    if requested == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported accelerator '{requested}'. Expected auto, cuda, xpu, or cpu.")


def preferred_torch_dtype(accelerator: str = "auto") -> torch.dtype:
    resolved = detect_accelerator(accelerator)
    if resolved == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if resolved == "xpu":
        return torch.float16
    return torch.float32


def resolve_device_map(accelerator: str) -> str | dict[str, str]:
    if accelerator == "cuda":
        return "auto"
    if accelerator == "xpu":
        return {"": "xpu:0"}
    return {"": "cpu"}


def preferred_optimizer(accelerator: str) -> str:
    if accelerator == "cuda":
        return "paged_adamw_8bit"
    return "adamw_torch"


def build_quantization_config(
    model_config: ModelConfig,
    accelerator: str = "auto",
) -> BitsAndBytesConfig | HqqConfig | None:
    if model_config.quantization_bits is None:
        return None

    resolved = detect_accelerator(accelerator)
    bits = model_config.quantization_bits
    backend = model_config.quantization_backend
    if backend == "auto":
        backend = "bitsandbytes" if bits == 4 else "hqq"

    if backend == "bitsandbytes":
        if bits != 4:
            raise RuntimeError(
                f"BitsAndBytes quantization only supports the current 4-bit path in msa_zria; received {bits}-bit."
            )
        if resolved not in {"cuda", "xpu"}:
            raise RuntimeError("4-bit BitsAndBytes quantization is only enabled for CUDA or XPU accelerators.")

        dtype = preferred_torch_dtype(resolved)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_config.use_double_quant,
            bnb_4bit_quant_type=model_config.quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )

    if backend == "hqq":
        try:
            return HqqConfig(
                nbits=bits,
                skip_modules=["lm_head"],
            )
        except ImportError as exc:
            raise RuntimeError(
                f"{bits}-bit quantization requires the optional HQQ backend to be installed."
            ) from exc

    raise ValueError(
        f"Unsupported quantization backend '{model_config.quantization_backend}'."
    )


def load_model_and_processor(
    model_config: ModelConfig,
    accelerator: str = "auto",
) -> tuple[AutoModelForMultimodalLM, AutoProcessor]:
    resolved = detect_accelerator(accelerator)
    dtype = preferred_torch_dtype(resolved)
    model_kwargs: dict[str, object] = {
        "device_map": resolve_device_map(resolved),
        "dtype": dtype,
    }

    quantization_config = build_quantization_config(model_config, resolved)
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
    accelerator = detect_accelerator(training_config.accelerator)
    dtype = preferred_torch_dtype(accelerator)

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
        optim=preferred_optimizer(accelerator),
        fp16=accelerator != "cpu" and dtype == torch.float16,
        bf16=accelerator != "cpu" and dtype == torch.bfloat16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        report_to=training_config.report_to,
        dataset_kwargs={"skip_prepare_dataset": training_config.skip_prepare_dataset},
        remove_unused_columns=training_config.remove_unused_columns,
    )
