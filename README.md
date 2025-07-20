# msa_zria
Combining the MSA research with the ZRIA reasoning model.

Thank you for the clarification. I will now begin a comprehensive research task to address the following:

1. Optimal stack for training LLaMA-2 14B on a single GPU, supporting both knowledge graph triples and natural language representations for fine-tuning.
2. Code generation for dataset preparation and fine-tuning of the LLaMA-2 model using both triple-form and NL-form inputs.
3. Code for parsing, code synthesis, and LM-based evaluations tailored to the customer service domain, orchestrated with DSPy and pydantic-ai.
4. Code for conducting the ablation study comparing ZRIA-only, Bayesian-only, and hybrid ZRIA+Pyro setups.
5. Recommendations and code for full end-to-end testing and ablation execution pipelines.
6. Recommendations for integrating pydantic-ai and DSPy for managing MSA workflows including prompt construction, task routing, and orchestration.

I will return with detailed findings, code snippets, and tooling guidance shortly.


# Architecture for LLaMA-2 Fine-Tuning with Knowledge Graphs and MSA

## 1. Dataset Generation from Knowledge Graph Triples

To fine-tune LLaMA-2 (14B) on our domain tasks, we first prepare **multiple versions of the training dataset** using knowledge graph (KG) triples:

* **Direct Triples** – Each sample presents triples in a structured format (e.g. `Subject | Predicate | Object` lines) as input. The model learns to interpret triples directly as knowledge. For example, a triple like `(Router123, hasIssue, Overheating)` could be given as:

  ```text
  Router123 | hasIssue | Overheating
  ```

  This format is easily read by LLMs when framed properly. We may wrap the triples with special tokens or instructions indicating their format. For instance, using a prompt template: *“Following are knowledge graph triples delimited by backticks. Use only this information to answer.”* then listing the triples. This teaches the model to handle structured triple input directly.

* **Natural Language (NL) Transformations** – Here we convert each triple (or set of triples) into natural language sentences. For example, the triple above might become: *“Router123 has an issue: overheating.”* This “lexicalization” process leverages the model’s strength in understanding text. We can use simple templates or even an LLM to generate such sentences for each triple. For instance:

  ```python
  subj, pred, obj = "Router123", "hasIssue", "Overheating"
  nl_sentence = f"{subj} has an issue: {obj.lower()}."
  ```

  We might also include context to mimic customer service dialogues, e.g. *“Customer reports that Router123 is overheating.”* The goal is to embed triples in fluent language so the model can learn from textual knowledge as it would from documents.

* **Hybrid (Combined)** – In some training examples, we provide both formats: the raw triples alongside a natural language description. This can be done by concatenating them (e.g. triples first, then a sentence summary) or using a structured prompt that includes both. For example: *“**Facts:** `Router123 | hasIssue | Overheating`. **In words:** Router123 has an overheating issue.”* By exposing the model to both, we allow it to align structured and unstructured knowledge. This is useful for ablation: we can later evaluate how the model performs when given only triples vs. only text vs. both.

**Dataset construction:** If a domain-specific KG is available (e.g. a product troubleshooting knowledge base), we extract triples from it. We then create parallel datasets:

1. **Triple-only dataset:** each entry has triples as input and the desired output (e.g. an answer or parsed result).
2. **Text-only dataset:** each entry has the NL version of those triples as input (same outputs).
3. **Hybrid dataset:** entries include both forms.

We can use existing KG-to-text datasets like WebNLG as a template, which provide pairs of triples and their textual descriptions. For our **customer service domain**, triples might represent facts like *Product features, known issues, resolutions*. For example, a set of triples:

```
Printer_X | issue | PaperJam  
Printer_X | resolution | ClearPaperPath  
```

could be turned into a training prompt:

```text
```

Printer\_X | issue | PaperJam
Printer\_X | resolution | ClearPaperPath

```

Explain the issue and resolution in one sentence.
```

And the target output: *“The printer Printer\_X has a paper jam issue, and the recommended resolution is to clear the paper path.”*.

By generating such data, we teach LLaMA-2 to incorporate factual triples into natural-language outputs. We ensure the dataset covers parsing tasks (e.g. turning a customer utterance into triples), code synthesis tasks (if any – see below), and evaluation tasks, all in our domain context.

**Ablation Preparation:** With triple-only and text-only subsets, we will be able to compare their impact. This supports experiments on whether the model learns better from structured knowledge or from natural language, or if combining them yields improvements.

## 2. Efficient Single-GPU Training Stack (DSPy-Compatible)

Fine-tuning a 14B parameter model is resource-intensive. We recommend using **QLoRA (Quantized Low-Rank Adaptation)** for an efficient single-GPU training stack. QLoRA quantizes the model weights to 4-bit precision and adds small trainable LoRA adapters, dramatically reducing memory usage while preserving model quality. This approach was shown to enable fine-tuning LLaMA-2 on a single 24GB GPU with minimal loss in performance.

Key components of the training stack:

* **Hugging Face Transformers + PEFT:** We use the HuggingFace Transformers library for model and tokenizer, and the PEFT (Parameter-Efficient Fine-Tuning) library to apply LoRA. These are compatible with DSPy since DSPy can load models via HuggingFace interfaces or through local inference servers. Using standard libraries ensures we can later integrate the fine-tuned model into DSPy’s workflow.

* **BitsAndBytes 4-bit Quantization:** We load LLaMA-2 in 4-bit mode using `BitsAndBytesConfig`. This drastically lowers VRAM usage. For example:

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  model_name = "meta-llama/Llama-2-14b-hf"
  bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                  bnb_4bit_use_double_quant=True, 
                                  bnb_4bit_quant_type="nf4", 
                                  bnb_4bit_compute_dtype=torch.float16)
  model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  ```

  Here we quantize to 4-bit NF4 format and let `device_map="auto"` to spread layers on the GPU (with CPU offload if needed). This is memory-efficient and **compatible with single-GPU setups**.

* **Apply LoRA adapters:** We freeze the base model weights and inject LoRA layers for fine-tuning:

  ```python
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  model = prepare_model_for_kbit_training(model)  # Prepares model for mixed-INT8/4bit training (e.g., handles layernorms)
  lora_config = LoraConfig(
      r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
  )
  model = get_peft_model(model, lora_config)
  print("Trainable params:", model.print_trainable_parameters())
  ```

  We target key projection layers of the transformer for LoRA (e.g., query/key/value projections). This drastically reduces trainable parameter count. The above config prints the number of trainable parameters to verify the reduction (should be only a few million parameters now, instead of 14B).

* **Training Configuration:** We use the Hugging Face `Trainer` or `Accelerate` for training. We enable **gradient checkpointing** and **8-bit optimizers** to further save memory. For example:

  ```python
  from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
  training_args = TrainingArguments(
      output_dir="outputs/llama2-finetune", 
      per_device_train_batch_size=1,
      gradient_accumulation_steps=16,  # accumulate to effectively larger batch
      num_train_epochs=3,
      learning_rate=2e-4,
      fp16=True,
      optim="paged_adamw_8bit",       # 8-bit Adam optimizer
      gradient_checkpointing=True,
      logging_steps=50,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      report_to="none",  # (or "wandb" if logging)
      ddp_find_unused_parameters=False
  )
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
  trainer = Trainer(model=model, args=training_args, 
                    train_dataset=train_dataset, eval_dataset=val_dataset, 
                    data_collator=data_collator)
  trainer.train()
  ```

  This configuration assumes a single GPU. We set a small batch and accumulate gradients to simulate a larger batch size. Mixed precision (`fp16`) and 8-bit optimizer reduce memory. **DeepSpeed** or **FSDP** are not strictly needed thanks to QLoRA, but if a GPU has limited RAM (e.g. <24GB), integrating DeepSpeed ZeRO stage-3 for CPU offloading could help. However, QLoRA already enabled single-GPU fine-tuning in practice.

* **DSPy Compatibility:** After fine-tuning, the model can be loaded via its HuggingFace name or local path. DSPy 3.0 can interface with local models by either direct HuggingFace integration or via backend like Ollama/LangChain. For example, we could load the model in DSPy with:

  ```python
  import dspy
  llama_path = "outputs/llama2-finetune"  # directory with trained model
  lm = dspy.LM(model=llama_path)  # DSPy should detect it as a local HuggingFace model
  dspy.configure(lm=lm)
  ```

  This ensures the fine-tuned model is ready for use in the modular DSPy pipeline described later. In summary, using HF Transformers + PEFT (QLoRA) gives us an **efficient, reproducible training stack** that fits on one GPU and remains compatible with DSPy’s tooling.

## 3. Fine-Tuning LLaMA-2 on Parsing, Code Synthesis, and Evaluation Tasks

We fine-tune LLaMA-2 on a **multi-task dataset** encompassing: (a) **parsing** of inputs into structured forms, (b) **code synthesis** for reasoning or actions, and (c) **LM-based evaluation** of outcomes. All tasks are grounded in the customer service domain, meaning the content involves support scenarios (user issues, troubleshooting steps, responses, etc.).

**Task Formulation:**

* **Parsing Task:** The model takes a customer utterance or dialogue and produces a structured representation (such as a semantic parse, slots, or KG triples). For instance, given input: *“User: My internet keeps disconnecting after the storm.”*, the model might output a JSON or triple like:

  ```json
  {"device": "internet_connection", "issue": "keeps disconnecting", "cause": "after storm"}
  ```

  We include many such examples in fine-tuning, possibly derived from annotated support logs or simulated via our KG. This teaches the model to extract key facts. We use an instruction style prompt to indicate the task, for example:

  **Prompt:** *"\[PARSE] Extract structured info from the user message.* `User: My printer is jammed.` *Provide output as JSON."*

  **Target:** `{"device":"printer","issue":"paper jam"}`.

* **Code Synthesis Task:** Here the model generates code (or pseudo-code) that solves or simulates a reasoning problem based on an input scenario. In an MSA-style system, this “code” could be a **probabilistic program** (e.g., Pyro code or a domain-specific script) that represents the problem logically. For example, the input might describe a scenario with certain conditions, and the output is a small Python/Pyro function modeling the scenario:

  **Prompt (MSA-style):** *"\[CODE] Given the facts below, write a Pyro probabilistic program that models the situation and answers the query.*
  Facts: If a device overheats, it will shut down 80% of the time; otherwise 0%. The device overheated. Query: will it shut down?"\*

  **Target (Pyro-like code):**

  ```python
  def model():
      overheat = True  # observed fact
      shut_down = pyro.sample("shut_down", pyro.distributions.Bernoulli(0.8 if overheat else 0.0))
      return shut_down
  ```

  This is an example where the model **learns to output executable code or a logical form** capturing the scenario. During fine-tuning, we likely provide simpler pseudo-code examples if Pyro syntax is too complex. The key is to train the model in translating natural language descriptions into structured *procedures*. In customer support, code generation might also mean producing a SQL query, a DSL snippet, or step-by-step pseudo-code to resolve an issue.

* **LM-Based Evaluation Task:** The model is asked to assess or evaluate a response/conversation. This could mean determining if a solution solved the user’s problem, rating the quality of service, or verifying compliance with guidelines. For example:

  **Prompt:** *"\[EVALUATE] Conversation: Agent says they reset the router; User says the internet is back. Did the agent resolve the issue?"*

  **Target:** *"Yes – the agent’s action resolved the user’s connectivity issue."*

  Another example: the model might be given a proposed solution and a problem description, and must output an evaluation like *"Likely correct"* or *"Missing steps X and Y"*. These tasks train the model to perform reasoning *about* outputs (a sort of meta task using its own LM capabilities to judge correctness or completeness).

**Unified Fine-Tuning Approach:** We combine these tasks in one training run by formulating prompts that identify the task type. This can be done with special tokens or bracketed keywords (`[PARSE]`, `[CODE]`, `[EVALUATE]` as above) or via natural instructions. The model then learns to map each prompt to the correct kind of output (structured parse, code, or analysis).

We ensure the training data covers varied scenarios in the customer service domain – different products, issues, and conversation styles – to make the model robust. By fine-tuning on these tasks, the model acquires the ability to (1) interpret and structure knowledge (parsing), (2) **synthesize procedures or reasoning code** (like constructing an internal simulation of a problem), and (3) reflect on outcomes (evaluation). These capabilities set the stage for implementing an **MSA-style reasoning pipeline** at inference.

**Example Training Data Entries (pseudo-code):**

```python
train_data = [
  # Parsing example
  {
    "input": "[PARSE] User: My ACME router is overheating and keeps rebooting. Provide a JSON of issue and device.",
    "output": '{"device": "ACME router", "issue": "overheating causing reboots"}'
  },
  # Code synthesis example
  {
    "input": "[CODE] Fact: If overheating, then reboot occurs with 90% probability. The device is overheating. Query: reboot?",
    "output": "def model():\n    overheat = True\n    reboot = pyro.sample('reboot', pyro.distributions.Bernoulli(0.9))\n    return reboot"
  },
  # Evaluation example
  {
    "input": "[EVALUATE] User issue: slow internet. Agent response: restarted router. Outcome: issue resolved. Was the response effective?",
    "output": "Yes. Restarting the router resolved the slow internet issue."
  }
]
```

During fine-tuning, we use a causal language modeling objective (next token prediction) on these prompt→response pairs. Over epochs, the model learns to produce well-formatted JSON for parsing, syntactically correct code for reasoning, and coherent judgments for evaluation.

Notably, including **code in training** will teach the model the style and syntax required (the model already has some knowledge from pre-training, but domain-specific fine-tuning helps, especially if using Pyro or a DSL it may not have seen). We might augment the code synthesis data by generating synthetic examples or using small scripts relevant to troubleshooting.

Finally, by mixing all tasks, the model can potentially **handle an end-to-end scenario**: e.g. parse a query, generate reasoning code, then evaluate an outcome – or at least each step individually when prompted appropriately. This sets up the model to be a core component in an MSA pipeline where it works alongside other components like a probabilistic program executor.

## 4. MSA-Style Parsing and Code Generation Logic (Inference Pipeline)

After fine-tuning, we deploy the model in an **MSA-style reasoning pipeline** inspired by the Model Synthesis Architecture. In this paradigm, the language model (LLaMA-2) is used to **parse problems and synthesize a probabilistic model**, while an external reasoner (Pyro) executes that model to derive answers. The fine-tuned LLaMA-2 now plays the role of the “global relevance and model construction” system, and Pyro provides the “coherent inference” system.

**Inference Workflow:**

1. **Parsing with LLM:** When a new query or scenario comes in (e.g., a complex customer issue), we first prompt LLaMA-2 to **parse** relevant information. This could involve extracting key facts or converting the problem description into a structured form. For example:

   ```python
   query = "Whenever I print more than 10 pages, the printer overheats and fails. It works fine for small jobs."
   parse_prompt = "[PARSE] " + query + " -> Extract conditions and outcomes as JSON."
   parsed = model.generate(parse_prompt)
   # parsed might be: {"condition": "print > 10 pages", "issue": "overheating failure", "normal_condition": "print <= 10 pages"}
   ```

   The structured output (here, JSON) identifies the key variables and conditions (e.g. large print jobs lead to overheating failure). This structured info will guide code generation.

2. **Code/Model Synthesis with LLM:** Next, we prompt LLaMA-2 to generate a **code representation** of the scenario – effectively a custom probabilistic model that can be executed. We use an instruction for code generation, possibly feeding in the structured info from the previous step. For example:

   ```python
   facts = parsed  # from step 1, e.g. a dict
   code_prompt = "[CODE] Use Pyro to model this scenario:\n"
   code_prompt += f"Condition: {facts['condition']} causes issue: {facts['issue']}; otherwise no issue.\n"
   code_prompt += "Query: Probability of failure under given condition."
   pyro_code = model.generate(code_prompt)
   print(pyro_code)
   ```

   The model might output a Pyro program like:

   ```python
   def model():
       failure_prob = 1.0 if condition_met else 0.0
       pyro.sample("failure", pyro.distributions.Bernoulli(failure_prob))
   condition_met = True  # more than 10 pages
   ```

   (This is a simplified example; a real case might include priors or uncertainty if appropriate.) The key is the **LM has turned the natural description into formal code**. This code uses domain knowledge (learned from fine-tuning) and any relevant triple facts provided in the prompt.

3. **Probabilistic Inference with Pyro:** We take the generated code and execute it with Pyro (or another PPL). Pyro allows us to perform inference on the model – e.g., compute the probability of the printer failing, or sample possible outcomes. Continuing the example:

   ```python
   import pyro, torch
   # assume pyro_code defines a function `model()`
   conditioned_model = pyro.condition(model, data={})  # if we have observed data to condition on
   posterior = pyro.infer.Importance(conditioned_model, num_samples=1000).run()
   marginal = pyro.infer.EmpiricalMarginal(posterior, "failure")
   failure_prob_est = float(marginal.mean)
   print("Estimated failure probability:", failure_prob_est)
   ```

   This step uses Pyro’s inference algorithms (here Importance sampling for simplicity) to answer the query posed. **Pyro serves as the “Bayesian reasoner,”** ensuring that the reasoning is coherent and probabilistically sound. In an MSA, this corresponds to the bespoke mental model inference.

4. **LM-Based Evaluation (optional):** After obtaining results from Pyro, we can loop the LLM back in to **explain or evaluate** the outcome (if needed). For instance, we could prompt LLaMA-2 with an `[EVALUATE]` instruction, providing the Pyro result and asking for a natural language explanation or a quality check. For example:

   ```python
   eval_prompt = f"[EVALUATE] The model predicts failure_prob ~ {failure_prob_est:.2f} when printing >10 pages. Explain what this means for the user."
   assessment = model.generate(eval_prompt)
   print(assessment)
   ```

   The model might output: *"There is about an 80% chance the printer will fail when printing more than 10 pages, which means for large jobs the printer is very likely to overheat and stop."* – translating the raw probability into a user-friendly explanation or further evaluation.

This pipeline marries LLM flexibility with probabilistic rigor. The **LLM (LLaMA-2)** provides global knowledge (it can recall relevant facts from training or the prompt, including KG triples) and constructs a **problem-specific program**, while **Pyro** executes that program to yield coherent results under uncertainty. This design reflects the findings that such a hybrid approach captures reasoning better than LLMs alone.

Under the hood, our fine-tuned LLaMA-2 is crucial for steps 1, 2, and 4. Its parsing and code generation abilities (honed on domain data) enable the MSA loop. We also benefit from the knowledge graph integration: the model was trained on triples and text, so we can *inject relevant triples into the prompt at inference time* (a form of retrieval-augmented generation). For example, if a user asks about a specific product issue, we can retrieve the product’s triple data from the KG and include it before the `[CODE]` prompt as additional context. The model will then utilize those facts when generating the program or answer, ensuring domain-specific accuracy.

In summary, the **MSA-style logic** is implemented by a sequence of LM calls (parse → code → evaluate) interleaved with a probabilistic reasoning step (Pyro execution). This achieves what the research envisioned: *“using language models to implement global relevance and model synthesis, and probabilistic programs to implement coherent world models”*. The result is an interpretable reasoning chain where each part can be inspected (parsed facts, generated code, inference result, and final evaluation).

## 5. Ablation Study: ZRIA vs. Pyro vs. Hybrid Execution

We now set up an **ablation study** to compare three approaches on reasoning tasks in our domain:

* **ZRIA model only:** Use the **ZRIA (Zero-Resonance Intelligence Architecture)** on its own. ZRIA is a specialized small neural model (from user files, \~1M parameters) designed for symbolic reasoning with mechanisms like fractal resonance and explicit memory. It excels at tasks requiring stepwise logic and agent-like behavior. We assume we have a trained ZRIA model (for example, one trained on the same parsing/reasoning tasks). This model will take the input (e.g. a structured problem or user query) and directly output an answer. It does *not* use LLaMA-2 or Pyro; it's a standalone reasoning engine. Prior results show ZRIA can significantly outperform larger LLMs on certain logical tasks – e.g., in a benchmark (ROCG) it achieved high accuracy where a 2B LLM got 0% – due to its architectural optimizations for reasoning over facts and rules.

* **Pyro-based Bayesian reasoner only:** Use a pure probabilistic approach with no LLM. In practice, this means we (the system designers) must pre-define a Bayesian model or set of rules for the task, and then use Pyro (or another PPL) to infer answers. For example, if the task is troubleshooting a device, a Pyro-only approach might involve a manually crafted Bayesian network of possible causes and effects. Given evidence, we run inference to find the most likely cause or outcome. This approach ensures **coherence and correct uncertainty handling** by design, but it relies on the model being correctly specified. It may struggle with open-world knowledge (since it can only use what we encoded). In evaluation, we feed each problem into a Pyro reasoning function and get an answer (e.g., the most probable diagnosis).

* **Hybrid (MSA) execution:** This uses **both LLaMA-2 (or ZRIA) and Pyro in parallel or sequence**, as described in the MSA pipeline. There are two flavors:

  * *Parallel hybrid:* Run the LLM-based solver and the Pyro solver independently on the same query, and then reconcile their answers. For instance, the LLM might answer using its learned knowledge, and the Pyro model might answer using the probabilistic model; if they agree, we have high confidence, if they differ, we might defer to one or have a rule to decide.
  * *Integrated hybrid:* Use the LLM to set up the Pyro model (as in the MSA style above), effectively chaining them. This is more serial than parallel, but it’s the true synergy of both. Since this was covered in Section 4, for the ablation we consider the **Parallel scenario**: LLM (or ZRIA) and Pyro each produce an answer independently.

**Accuracy Evaluation:** We use a test set of reasoning problems (e.g., a set of customer issues with known correct outcomes or a benchmark from the domain). We measure simple accuracy: does the model’s answer match the ground truth? We compute accuracy for:

* ZRIA alone,
* Pyro alone,
* Hybrid.

If available, we could also measure other metrics (like confidence calibration or reasoning consistency), but accuracy (or exact match) is our primary metric.

**Implementation Outline:** Pseudocode for running the ablation:

```python
# Assume we have functions or models:
# zria_model (callable on input), llama_model (if using LLM for hybrid), pyro_reasoner (function)
correct = {"ZRIA": 0, "Pyro": 0, "Hybrid": 0}
for case in test_cases:
    question, true_answer = case["question"], case["answer"]
    # ZRIA only
    zria_ans = zria_model.predict(question)  # e.g., returns an answer string or structured answer
    if zria_ans == true_answer:
        correct["ZRIA"] += 1
    # Pyro only
    pyro_ans = pyro_reasoner(question)  # runs a predefined probabilistic model for this question
    if pyro_ans == true_answer:
        correct["Pyro"] += 1
    # Hybrid (parallel execution and combination)
    llama_ans = llama_model.generate(question)  # LLaMA-2's direct answer (for comparison or integration)
    # For combination logic: here we choose a simple rule, e.g. trust Pyro for numeric/probabilistic answers, 
    # otherwise trust ZRIA/LLM.
    if isinstance(pyro_ans, (int,float)) or pyro_ans in ["Yes","No"]:  
        hybrid_ans = pyro_ans  # if Pyro yields a concrete evaluation
    else:
        # otherwise, if Pyro returns a distribution or complex output, use LLM answer or a merge strategy
        hybrid_ans = zria_ans  # (could also compare and choose majority if both are yes/no, etc.)
    if hybrid_ans == true_answer:
        correct["Hybrid"] += 1

# Calculate accuracies
for mode, count in correct.items():
    accuracy = count / len(test_cases) * 100
    print(f"{mode} Accuracy: {accuracy:.2f}%")
```

In practice, the combination logic for hybrid can be more sophisticated. For example, the **MSA integrated approach** would always produce one answer (so no conflict). In a *truly parallel* hybrid, one could ensemble the outputs or have the LLM evaluate which answer is more plausible.

**Expected Outcomes:** Based on prior research and our hypothesis:

* The Pyro-only model will excel when the problem is well-described by the fixed probabilistic rules it knows, but it may **fail on questions requiring external knowledge** or flexible understanding (long-tail issues not encoded in the Bayesian model).
* The ZRIA/LLM-only model will do well on questions requiring broad knowledge or pattern recognition (thanks to training), but might **make logical mistakes** or be inconsistent on complex multi-step reasoning (LLMs can hallucinate or confuse correlation vs causation).
* The Hybrid approach should achieve the **highest accuracy**, as it leverages strengths of both. For instance, the LLM can bring in background knowledge K, while Pyro ensures logically coherent inference on that knowledge. This mirrors the finding that MSA (hybrid) captured human-level reasoning better than either direct LLM or chain-of-thought LLM alone.

From the user-provided benchmark: ZRIA (a specialized reasoning model) **handled all aspects of a complex reasoning game with high fidelity, whereas a standard LLM failed across the board**. This suggests that on purely symbolic tasks, ZRIA > LLM. However, ZRIA is not a large model with encyclopedic knowledge; it might not know specific customer service facts unless explicitly programmed. That’s where the LLM’s pretraining helps. The hybrid strategy would use LLM to supply needed facts and ZRIA/Pyro to carry out exact reasoning – achieving the best of both.

In our ablation experiment, we will report the accuracy of each approach. We expect to see the hybrid come out on top. If, say, ZRIA got 85% of the questions right and Pyro got 75% (on different subsets of questions), the hybrid might solve \~95% by succeeding whenever either method could solve it. We will also analyze failure cases to see where each approach fails: e.g., ZRIA/LLM might misinterpret a question (failure of parsing) while Pyro might miss an edge case due to an incomplete model.

This ablation is valuable to justify the inclusion of both learning-based and rule-based components in the final system.

## 6. End-to-End Pipeline Completeness and Reproducibility

Ensuring the **completeness and reproducibility** of the entire pipeline is crucial for reliable experimentation and deployment. We address this by improvements in each stage and by adding reproducibility mechanisms:

* **Comprehensive Data Preparation:** We provide scripts to systematically generate the datasets described in Section 1. This includes:

  * **KG Triple Extraction:** Code that pulls triples from the knowledge graph (with versioning of the KG if it changes).
  * **Triple-to-Text Conversion:** Reproducible transformations (with fixed random seeds if using an LLM to generate text). For traceability, we might log which triples were converted to which sentences.
  * **Dataset Serialization:** We save the prepared datasets in a standard format (e.g. JSONL or CSV) and **include them in version control or data registry**. For example, using Hugging Face Datasets to load and split data ensures consistency in shuffling and batching across runs.

* **Training Process Logging:** The training code is instrumented to log all relevant details:

  * Hyperparameters (learning rate, batch size, seed, LoRA config) are saved (for example, dumped as a JSON or printed at start). We can use a configuration file or even a `Pydantic` model to define these, ensuring they are documented.
  * During training, metrics (loss, validation accuracy on tasks) are logged at each epoch. This can be done via `transformers.Trainer` callbacks or using an experiment tracking tool (like Weights & Biases or TensorBoard). Logging helps reproduce or debug if needed.
  * We set a **random seed** for training for determinism: e.g., `set_seed(42)` from Transformers ensures data shuffling and weight initialization are the same each run. Additionally, we can set `torch.backends.cudnn.deterministic = True` if exact reproducibility is needed (at some cost to performance).

  Example:

  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42); torch.manual_seed(42)
  ```

* **Model Checkpointing and Versioning:** We save intermediate checkpoints and especially the final fine-tuned model (including tokenizer and LoRA adapters). Each checkpoint is tagged with a version. Using the Hugging Face Hub or a model registry can help keep track of versions. This allows us to reload a specific trained model exactly as it was. Also, we export the model in a standard format (PyTorch `.bin` or safetensors) to avoid any drift.

* **Inference Pipeline Integration:** We ensure our inference code (Section 4 workflow) is packaged in a clear, step-by-step manner (possibly a single script or notebook that loads the model, runs parsing → code gen → pyro inference). All external dependencies (like the Pyro model definitions) are contained or referenced. If the Pyro model depends on certain parameters (say, a probability for an event), those are documented or learned from data to avoid ad-hoc changes.

* **Evaluation and Benchmarking:** The evaluation scripts (for the ablation in Section 5 and any other metrics) are included in the repository. They should load the appropriate model (ZRIA, LLaMA, etc.) and dataset, then produce metrics in a consistent format. We use standard metrics libraries where possible (e.g., scikit-learn for accuracy/f1, or custom code for exact match). Reproducibility here means that given the same model and test set, the results should match published numbers. We seed any stochastic evaluation component (for example, if we sample from the model for evaluation, we fix the sampling seed to compare outputs exactly across runs).

* **Reproducibility Mechanisms:** Besides seeding and logging, we can utilize containerization or environment management:

  * Provide a `requirements.txt` or `environment.yml` specifying exact library versions (e.g. transformers 4.x, dspy 3.0, pyro 1.9.1, etc.). This ensures that anyone setting up the environment gets the same versions we tested with.
  * Optionally, use Docker to encapsulate the runtime environment. This includes CUDA version, drivers, etc., eliminating "it works on my machine" issues.
  * Use **Pydantic for config management**: We define Pydantic models for configuration (for example, a `TrainingConfig` class with fields for all hyperparams). This not only validates the config values but can easily serialize to JSON/YAML. This config, when saved alongside results, allows exact reruns. For instance:

    ```python
    from pydantic import BaseModel
    class TrainingConfig(BaseModel):
        model_name: str
        epochs: int
        batch_size: int
        lora_r: int
        seed: int
    config = TrainingConfig(model_name="LLaMA-2-14B", epochs=3, batch_size=1, lora_r=16, seed=42)
    config_json = config.model_dump_json()
    ```

    We save `config_json` with the model. Later, we can load it to instantiate the same config for retraining or analysis.

* **End-to-End Testing:** We construct a few end-to-end scenarios (small-scale integration tests). For example, take a sample user question, run it through the entire orchestrated pipeline (parse → code → reason → answer) and check that the output is sensible and consistent. These test cases (and their expected outputs) can be part of the repo, ensuring that changes to any component can be quickly validated against known correct behavior.

By implementing the above, we attain a pipeline that is **complete** (covers data prep to evaluation) and reproducible. If someone else follows the documented steps and uses the provided code, they should be able to regenerate the fine-tuned model and see the same evaluation results. Reproducibility is further aided by the modular design: each component (data generation, training, inference, evaluation) is encapsulated and can be run independently if needed, which helps in isolating issues and verifying each stage. In research contexts, this thoroughness means our results can be trusted and built upon.

## 7. Orchestration with Pydantic-AI and DSPy

To manage the complex workflow, we use **pydantic-ai** for structured prompt I/O and **DSPy 3.0** for high-level orchestration of the modules. This combination allows us to define clear schemas and assemble modular workflows in a **declarative, extensible manner**.

**Schema Definitions with Pydantic:** We define Pydantic `BaseModel` classes to represent the data flowing between modules. This enforces a schema at each step:

* *Example:* Define a schema for the parse output:

  ```python
  from pydantic import BaseModel
  class ParseOutput(BaseModel):
      device: str
      issue: str
      cause: str | None = None
  ```

  This means the parsing step should produce a JSON with keys `device`, `issue`, and optionally `cause`. Similarly, we can define `CodeOutput` (if we want to structure the code as text or as a Python object), and `EvaluationOutput` for the evaluation step (could be as simple as a boolean or rating, or a text summary).

* We incorporate these schemas into prompting. With Pydantic-AI, we can obtain a JSON Schema or example from the model to guide the LLM. For instance:

  ```python
  from pydantic_ai import generate_schema_prompt
  schema_str = ParseOutput.model_json_schema()  # JSON schema as dict
  schema_prompt = generate_schema_prompt(schema_str)
  ```

  The `generate_schema_prompt` (hypothetical helper) might produce an instruction like: *“Provide output in JSON with fields: device (string), issue (string), cause (string or null).”* This is then appended to the LLM prompt. By doing so, we **steer the LLM to output structured data** that can be parsed into our Pydantic model. After generation, we do:

  ```python
  raw_output = lm_agent.chat(prompt)  # get LLM response text
  result = ParseOutput.model_validate_json(raw_output.content)
  ```

  The `model_validate_json` will parse the JSON (ignoring any extraneous text) and validate it against the schema, raising an error if something is missing or invalid. This catches issues early and can trigger a retry or fix if the LLM's output was malformatted.

* We use similar patterns for other steps. Pydantic not only validates but can coerce types (e.g., ensure numbers are int/float) and provide defaults. This ensures, for example, if the LLM returns `"cause": none` (as text), it can interpret it as Python `None` or throw a clear error.

**Modular Workflows with DSPy:** DSPy allows us to codify each step (parsing, code gen, reasoning, evaluation) as a **module** with well-defined interfaces, rather than writing one-off prompt strings for each. This makes the pipeline **maintainable and reusable**. We use DSPy’s paradigm of describing AI behaviors as code to create our pipeline:

* **Prompt Templates as Modules:** In DSPy, we can create classes that subclass `dspy.Module` to encapsulate a prompt. For example:

  ```python
  import dspy
  from dspy import Signature, InputField, OutputField

  class ParseModule(dspy.Module):
      # Define the input and output schema for this module using DSPy Signature
      signature = Signature(
          inputs=InputField(str, "customer_query"),
          outputs=OutputField(ParseOutput, "parsed_result")
      )
      # The prompt method defines how to turn input into a prompt
      def prompt(self, customer_query: str) -> str:
          return (f"Extract the device and issue from the following customer query.\n"
                  f"Return a JSON: {customer_query}")
  ```

  Here, `ParseOutput` is our Pydantic model from above. DSPy can utilize that to automatically validate the output (especially if using the `JSONAdapter` in DSPy, which ensures the LLM outputs JSON conforming to the OutputField schema). The module’s `prompt` function prepares a consistent prompt string. We could further refine by including few-shot examples inside the prompt if needed (DSPy supports things like `dspy.Example` objects in prompts to provide exemplars).

* **Tool Integration:** For steps that involve external reasoning (Pyro, or calling the ZRIA model), we can use DSPy Tools. For instance, DSPy has a `PythonInterpreter` tool. We could set up a tool that executes a given Python code string (carefully sandboxed) and returns the result. Our code generation module can then be followed by a tool invocation that runs the Pyro code. Concretely:

  ```python
  from dspy import Tool, PythonInterpreter

  pyro_tool = PythonInterpreter()  # a tool that runs python code
  # After code generation:
  code = code_generation_module(customer_query)
  pyro_result = pyro_tool.run(code + "\nprint(run_inference())")
  ```

  In the above pseudo-code, assume the generated code string contains a `run_inference()` function or similar that executes the model and returns an answer. We append a print to get output. The tool runs it and we capture the output (which might be the probability or answer we seek). This could then be fed into the evaluation module.

  Alternatively, we treat the Pyro reasoner as a black-box tool: define a `reason_about` function in Python that encapsulates Pyro inference for our domain. Then use `Tool(reason_about)` to let the LLM invoke it via function-calling style. DSPy (and Pydantic AI) can facilitate function calling by exposing a JSON schema for the tool’s input/output.

* **Parallel and Sequential Composition:** DSPy provides combinators like `ChainOfThought`, `Parallel`, etc.. We can compose our modules as needed:

  * The **parsing** and **code generation** are sequential (the output of parse feeds into code gen). We can wrap them in a `ChainOfThought` module, which will feed the output of one as input to the next automatically.

  * The **parallel** execution for ablation or for running ZRIA & Pyro concurrently can be done with `dspy.Parallel`. For example, we could have:

    ```python
    zria_module = Tool(zria_model.predict)  # wrap the ZRIA model's predict function as a tool
    hybrid_module = dspy.Parallel(zria_module, pyro_tool)
    ```

    This would call both and gather results. We might then add a small post-processing step to choose between results – possibly using a DSPy `Module` that takes both outputs and has a rule to pick one (this could even be another LM module that, given the two answers, decides which is more likely correct).

  * **Chain with Evaluation:** Finally, we chain the evaluation step. That might take the original query, the model’s answer (or the Pyro results), and produce an evaluation. We can design a module `EvaluationModule` similarly, with a prompt that says: *“Given the conversation and outcome, evaluate success.”* Its input could be a combination of initial query and model answer (DSPy’s Signature supports multiple inputs).

* **Using Pydantic with DSPy:** Notably, DSPy can integrate with Pydantic for I/O. By specifying `OutputField(ParseOutput, ...)` as above, DSPy knows the output should be parsed into that model. Internally it may use a `JSONAdapter` to format the prompt and parse the response automatically. This means the LLM’s raw string is converted to `ParseOutput` object without manual JSON loads – reducing error-prone parsing code. It’s part of making the AI programming less about string hacks and more about structured data. As the DSPy documentation notes, it lets us build AI logic in *“structured code, rather than brittle strings”*.

* **Knowledge Graph Integration:** We incorporate KG access as needed. If using *retrieval augmentation*, we can create a retrieval tool that given a query (or identified entity from parse) returns relevant triples. For example, a `KnowledgeGraphTool` that queries our KG (could be a simple lookup in a Python dict or a database query). In DSPy, this can be a subclass of `Tool` or just used via the `PythonInterpreter` with a function. The workflow might be:

  1. Parse user query -> get device/issue.
  2. Call KG tool with device/issue -> get stored triples about that issue (e.g., known resolution steps).
  3. Feed those triples into the code generation prompt (so the model has factual info to use).
     DSPy’s module design makes it easy to insert this extra step without rewriting the whole prompt logic.

* **Extensibility and Modularity:** If we want to add a new task or swap out a model, the modular design helps. For example, if a new version of ZRIA or a different reasoning engine comes out, we can just implement a new Tool for it and plug it into the Parallel module. Similarly, if we want to experiment with a different way of prompting the evaluation (say, using a different schema or more chain-of-thought), we can adjust the EvaluationModule without touching the parsing/code modules.

To illustrate, here’s a schematic orchestration using DSPy with the modules (simplified):

```python
# Define modules
parser = ParseModule()
code_gen = CodeGenModule()  # similar to ParseModule, outputs a CodeOutput (string with code)
evaluator = EvalModule()

# Compose workflow: Parse -> Code -> Run Pyro -> Evaluate
# Using a custom Orchestrator class to tie it together in code for clarity
class ReasoningPipeline:
    def __init__(self):
        self.parser = ParseModule()
        self.code_gen = CodeGenModule()
        self.pyro_tool = PythonInterpreter()
        self.evaluator = EvalModule()
    def __call__(self, customer_query):
        parsed = self.parser(customer_query)            # structured ParseOutput
        code = self.code_gen(parsed=parsed)             # generate code using parsed info
        pyro_result = self.pyro_tool.run(code.code_str) # execute code (assuming code.code_str is the string)
        evaluation = self.evaluator(query=customer_query, answer=pyro_result)
        return evaluation
pipeline = ReasoningPipeline()
output = pipeline("User: ... (some query)...")
print(output)
```

In an actual DSPy implementation, some of this orchestration might be handled by DSPy itself (since DSPy can wire modules together if we specify dependencies). The code above is a conceptual outline showing each piece used in turn.

**Advantages:** By leveraging Pydantic and DSPy:

* We **reduce errors**: The model outputs are validated at each step, catching JSON syntax errors or missing fields, which we can handle (e.g., by re-prompting or defaulting). This is much safer than relying on the LLM to always format perfectly.
* We **improve clarity**: The use of classes (`ParseModule`, etc.) makes the flow self-documenting. Future contributors can see the expected input/output of each part easily. The prompt templates are contained in one place (the module code) rather than scattered.
* We **enable experimentation**: Want to try a different prompt wording or a different LLM for one part? Just change that module or its configuration. DSPy’s declarative design supports swapping models or adding optimizers (it has facilities like `BootstrapFinetune` optimizer if we wanted to do dynamic prompt tuning, etc., though that’s optional).
* We maintain **consistency**: All prompts can follow a unified style (since they’re defined systematically) and we avoid ad-hoc string concatenation throughout the codebase.

In conclusion, **DSPy 3.0 and Pydantic-AI provide an orchestration framework** ideally suited to our needs. They ensure that prompt templates and schemas are centrally defined and enforced, that each component of the parsing → code → evaluation workflow is modular, and that integration with external reasoning (ZRIA, Pyro) is straightforward via tools and parallel modules. This high-level programming of the AI pipeline (as opposed to low-level prompt tinkering) mirrors the shift “from prompting to programming”, giving us a robust, extensible end-to-end system for domain-specific reasoning with LLaMA-2.

**Sources:**

* LLaMA-2 fine-tuning with 4-bit LoRA for single-GPU efficiency
* Knowledge graph triple to text prompting and dataset examples
* Model Synthesis Architecture (combining LMs with probabilistic programs)
* Pyro probabilistic programming for coherent Bayesian reasoning
* ZRIA architecture advantages and benchmark results over LLMs
* DSPy modular AI programming and structured prompting paradigm

