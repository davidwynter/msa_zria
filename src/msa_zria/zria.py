from huggingface_hub import login
from google.colab import userdata

# Retrieve your secret API key
HF_TOKEN = userdata.get('Hugging')

# Log in to Hugging Face
login(token=HF_TOKEN)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import random
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. Core ZRIA Architecture (As Provided)
# =============================================================================

class FractalAttentionalResonance(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim))
        self.bias_generator = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, self.num_heads * self.head_dim))

    def forward(self, x):
        B, T, D = x.shape
        global_context = x.mean(dim=1)
        dynamic_bias = self.bias_generator(global_context).view(B, self.num_heads, self.head_dim)
        Q, K, V = (p(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        bias = dynamic_bias.unsqueeze(-1)
        fractal_resonance = torch.matmul(Q, bias)
        scores = scores + fractal_resonance
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)

class FAR_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = FractalAttentionalResonance(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output

# This is the P-FAF implementation
class FractalEmbeddingLayer(nn.Module):
    def __init__(self, dim, num_fractals=4):
        super().__init__()
        self.num_fractals, self.dim = num_fractals, dim
        self.dims = nn.Parameter(torch.rand(num_fractals) * 2 + 1)
        self.weight_generator = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, num_fractals))
        self.fractal_functions = [lambda x: torch.sin(x*2*math.pi), lambda x: x-torch.floor(x), lambda x: 4*x*(1-x), lambda x: torch.sigmoid(5*(x-0.5))]

    def forward(self, x):
        # The equation P-FAF(x) = ∑(p_i * f_i(x^(1/d_i))) is implemented here.
        x_safe = torch.sigmoid(x) # Ensure input is in a stable range for fractal functions
        # p_i: Probabilities are generated dynamically from the input
        p_logits = self.weight_generator(x.mean(dim=1))
        p_weights = F.softmax(p_logits, dim=-1).unsqueeze(1).unsqueeze(-1)
        # f_i(x^(1/d_i)): Apply each fractal function to the scaled input
        fractal_outputs = [f(torch.pow(x_safe, 1.0/d)).unsqueeze(-2) for d, f in zip(self.dims, self.fractal_functions)]
        fractal_stack = torch.cat(fractal_outputs, dim=-2)
        # ∑(...): The weighted sum is calculated
        return x + torch.sum(p_weights * fractal_stack, dim=-2)

class ZRIA_for_Reasoning(nn.Module):
    """
    The complete ZRIA model, adapted for the generative reasoning task.
    """
    def __init__(self, dim, vocab_size, max_seq_len=256):
        super().__init__()
        # Foundational Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.fractal_embedding = FractalEmbeddingLayer(dim) # P-FAF Layer

        # Core Encoders
        far_encoder_layer = FAR_TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*2)
        self.encoder = CustomTransformerEncoder(far_encoder_layer, num_layers=2)

        # ADJUSTMENT: A single generative head replaces the old classifier/regressor heads
        self.decoder_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.positional_embedding[:, :T, :]
        pfaf_x = self.fractal_embedding(x) # Apply P-FAF
        encoded_repr = self.encoder(pfaf_x)
        logits = self.decoder_head(encoded_repr)
        return logits

# =============================================================================
# 2. Data Handling & Task Generation
# =============================================================================

class ReasoningTokenizer:
    """A word-level tokenizer for the reasoning task."""
    def __init__(self, corpus):
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        all_words = set(word for sent in corpus for word in sent.lower().split())
        self.vocab = self.special_tokens + sorted(list(all_words))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_idx, self.sos_idx, self.eos_idx = 0, 2, 3

    def encode(self, sentence, add_special_tokens=True):
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence.lower().split()]
        if add_special_tokens:
            return [self.sos_idx] + tokens + [self.eos_idx]
        return tokens

    def decode(self, token_ids):
        return ' '.join([self.idx_to_word.get(idx, '<UNK>') for idx in token_ids if idx not in (self.pad_idx, self.sos_idx, self.eos_idx)])

class ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data, self.tokenizer, self.max_len = data, tokenizer, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_tokens = self.tokenizer.encode(item['prompt'], add_special_tokens=True)
        answer_tokens = self.tokenizer.encode(item['action'] + " " + item['justification'], add_special_tokens=True)

        enc_input = prompt_tokens[:self.max_len]
        enc_input += [self.tokenizer.pad_idx] * (self.max_len - len(enc_input))

        target = answer_tokens[:self.max_len]
        target += [self.tokenizer.pad_idx] * (self.max_len - len(target))

        return {
            "prompt_text": item['prompt'],
            "answer_text": item['action'] + " " + item['justification'],
            "encoder_input": torch.tensor(enc_input, dtype=torch.long),
            "target_output": torch.tensor(target, dtype=torch.long),
            "phase": item['phase'] # ✅ Add this line
        }

def generate_test_case(phase):
    # This function remains the same as before, generating varied test cases.
    objects, colors, actions = [("A", "B"), ("C", "D")], [("red", "blue"), ("green", "yellow")], [("left", "right"), ("up", "down")]
    obj1, obj2 = random.choice(objects)
    color1, color2 = random.choice(colors)
    action1, action2 = random.choice(actions)
    if phase == "basic":
        prompt = f"Facts: Object {obj1} is {color1}. Object {obj2} is {color2}. Rule: If Object {obj1} is {color1}, move Object {obj2} {action1}. Question: What should you do?"
        action, justification = f"Move Object {obj2} {action1}.", f"The condition 'Object {obj1} is {color1}' was met."
    elif phase == "contradiction":
        prompt = f"Facts: Object {obj1} is {color1}. If Object {obj1} is {color1}, ignore the primary rule. Primary Rule: If Object {obj1} is {color1}, move Object {obj2} {action1}. Question: What should you do?"
        action, justification = "Do nothing.", "The instruction to ignore the primary rule overrides the action rule."
    elif phase == "recursive":
        prompt = f"Facts: Object {obj1} is {color1}. Object {obj2} is {color2}. It is not raining. Rule: If Object {obj1} is {color1} and Object {obj2} is {color2}, and if it is not raining, then move Object {obj1} {action1} and {obj2} {action2}. Question: What should you do?"
        action, justification = f"Move Object {obj1} {action1} and {obj2} {action2}.", "All conditions in the nested rule were met."
    elif phase == "memory":
        filler = "Intermediate Log: The sky is cloudy. Inventory logs were updated yesterday. System diagnostics show normal parameters."
        prompt = f"Initial Facts: Object {obj1} is {color1}. If Object {obj1} is {color1}, move Object {obj2} {action1}. {filler} Question: Based on the initial facts, what action should be taken regarding Object {obj2}?"
        action, justification = f"Move Object {obj2} {action1}.", f"The initial fact stated 'Object {obj1} is {color1}', triggering the rule."
    return {"prompt": prompt, "action": action, "justification": justification, "phase": phase}


# =============================================================================
# 3. Model Training & Inference
# =============================================================================

def train_zria_model(model, dataloader, epochs=15, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=dataloader.dataset.tokenizer.pad_idx)

    print("=== Starting ZRIA Training on Reasoning Task ===")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # For this encoder-only architecture, we use the prompt as input and
            # the full answer sequence as the target for every position.
            encoder_input = batch['encoder_input'].to(device)
            target_output = batch['target_output'].to(device) # Shape: [B, T_ans]

            # The model predicts a distribution over the vocab for each input token position.
            logits = model(encoder_input) # Shape: [B, T_prompt, Vocab]

            # We'll align the target to the prompt length for the loss calculation.
            # This is a common strategy for encoder-only generative training.
            T_prompt = logits.shape[1]
            T_ans = target_output.shape[1]
            aligned_target = F.pad(target_output, (0, max(0, T_prompt - T_ans)), value=dataloader.dataset.tokenizer.pad_idx)[:, :T_prompt]

            loss = criterion(logits.view(-1, logits.size(-1)), aligned_target.view(-1))
            if torch.isnan(loss): continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    print("=== Training Complete ===")

def zria_generate_answer(model, prompt, tokenizer, max_len=50, device='cpu'):
    model.eval()
    model.to(device)
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(prompt_tokens)
        # For an encoder-only model, we take the argmax at each output position.
        # This is not auto-regressive but a direct generation based on the input states.
        predicted_token_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    return tokenizer.decode(predicted_token_ids)

class Gemma_InferenceShell:
    """Wrapper for Gemma that includes few-shot prompting."""
    def __init__(self, model_name="google/gemma-4-12B-it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=self.torch_dtype,
        )
        print("Gemma Model Loaded.")
        self.few_shot_prompt_template = self._build_few_shot_template()

    def _build_few_shot_template(self):
        return """You are a precise, logical reasoning engine. Analyze the facts and rules to determine the correct action and provide a justification.

**Example 1:**
Facts: Object X is purple. If Object X is purple, ignore the primary rule. Primary Rule: If Object X is purple, move Object Y up. Question: What should you do?
Answer:
Do nothing. The instruction to ignore the primary rule overrides the action rule.

**Example 2:**
Facts: Object C is green. Object D is yellow. Rule: If Object C is green, move Object D forward. Question: What should you do?
Answer:
Move Object D forward. The condition 'Object C is green' was met.

---

**Current Problem:**
{problem}
Answer:
"""

    def forward(self, prompt_text):
        full_prompt = self.few_shot_prompt_template.format(problem=prompt_text)
        messages = [{"role": "user", "content": full_prompt}]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=60,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        prompt_length = model_inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# =============================================================================
# 4. Main Execution & Benchmark
# =============================================================================

def run_benchmark(zria_model, gemma_shell, test_set, zria_tokenizer):
    print("\n" + "="*50)
    print("🔬 RUNNING BENCHMARK...")
    print("="*50)

    zria_results = []
    gemma_results = []

    for case in tqdm(test_set, desc="Benchmarking Models"):
        # ZRIA Evaluation
        zria_answer = zria_generate_answer(zria_model, case['prompt_text'], zria_tokenizer, device=zria_model.token_embedding.weight.device)
        zria_correct = case['answer_text'].lower() in zria_answer.lower()
        zria_results.append({"phase": case['phase'], "correct": zria_correct})

        # Gemma Evaluation
        gemma_answer = gemma_shell.forward(case['prompt_text'])
        gemma_correct = case['answer_text'].lower() in gemma_answer.lower()
        gemma_results.append({"phase": case['phase'], "correct": gemma_correct})

    # Summarize
    def summarize(results, model_name):
        print(f"\n--- {model_name} Summary ---")
        summary = {}
        phases = sorted(list(set(r['phase'] for r in results)))
        for phase in phases:
            phase_results = [r['correct'] for r in results if r['phase'] == phase]
            accuracy = sum(phase_results) / len(phase_results) if phase_results else 0
            print(f"  Phase: {phase.title():<15} | Accuracy: {accuracy:.2%}")

    summarize(zria_results, "ZRIA with P-FAF")
    summarize(gemma_results, "Gemma-4-12B-IT")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Generate Datasets
    phases = ["basic", "recursive", "contradiction", "memory"]
    train_data = [generate_test_case(p) for p in phases for _ in range(50)]
    test_data = [generate_test_case(p) for p in phases for _ in range(10)]
    corpus = [d['prompt'] + " " + d['action'] + " " + d['justification'] for d in train_data]

    # 2. Setup ZRIA
    zria_tokenizer = ReasoningTokenizer(corpus)
    zria_dataset = ReasoningDataset(train_data, zria_tokenizer)
    zria_dataloader = DataLoader(zria_dataset, batch_size=8, shuffle=True)
    zria_model = ZRIA_for_Reasoning(dim=128, vocab_size=zria_tokenizer.vocab_size, max_seq_len=128)

    # 3. Train ZRIA
    train_zria_model(zria_model, zria_dataloader, epochs=20, device=device)

    # 4. Initialize Gemma
    gemma_shell = Gemma_InferenceShell()

    # 5. Run Benchmark
    test_dataset = ReasoningDataset(test_data, zria_tokenizer)
    run_benchmark(zria_model, gemma_shell, test_dataset, zria_tokenizer)
