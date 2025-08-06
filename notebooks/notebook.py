# %% [markdown]
# ### Installation

# %%
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     !pip install --no-deps unsloth vllm==0.8.5.post1

# # %%
# #@title Colab Extra Install { display-mode: "form" }
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     !pip install --no-deps unsloth vllm==0.8.5.post1
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     # Skip restarting message in Colab
#     import sys, re, requests; modules = list(sys.modules.keys())
#     for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer

#     # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
#     f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
#     with open("vllm_requirements.txt", "wb") as file:
#         file.write(re.sub(rb"(transformers|numpy|xformers)[^\n]{1,}\n", b"", f))
#     !pip install -r vllm_requirements.txt

# %% [markdown]
# ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# %%
from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth
max_seq_length = 2048
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None, # None for auto detection by unsloth
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit dynamic quantization for superior accuracy and lower memory use
    full_finetuning = False,
    # token = "hf_...", # use one if using gated models
)

# %%
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

# Check model quantization status
print(f"4-bit loaded: {model.is_loaded_in_4bit}")
print(f"Quantized: {model.is_quantized}")
print(f"Method: {model.quantization_method}")

# %% [markdown]
# # Gemma 3N can process Text, Vision and Audio!
# 
# Let's first experience how Gemma 3N can handle multimodal inputs. We use Gemma 3N's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

# %%
from transformers import TextStreamer
# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

# %% [markdown]
# We now add LoRA adapters so we only need to update a small amount of parameters!

# %%
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!
    r = 16,           # Larger = higher accuracy, but no significant improvements. checkout [LoRA](https://arxiv.org/abs/2106.09685) paper
    lora_alpha = 32,  # Recommended alpha >= r
    lora_dropout = 0, # 0.1 provides moderate regularization without being too aggressive. Common range is 0.05-0.2 for LoRA fine-tuning. Since we are doing text only, 0.1 is a good default.
    bias = "none", # Bias terms are simple additive constants that shift neuron outputs. They're less critical for task adaptation because:
    # What bias does: If a neuron computes Wx + b, the bias b just shifts the entire output up/down by a constant.
    # Why freezing works: The main "intelligence" comes from the weight matrix W learning new patterns. The bias shifts are usually already well-calibrated from pretraining.
    random_state = 3407,
    use_rslora = True,
    use_gradient_checkpointing = "unsloth",
    loftq_config = {}
)
model.print_trainable_parameters()

# %% [markdown]
# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-3` format for conversation style finetunes.
# 
# ```
# <bos><start_of_turn>user
# Hello!<end_of_turn>
# <start_of_turn>model
# Hey there!<end_of_turn>
# ```
# 
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.

# %%
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# %%
from datasets import load_dataset

train_ds = load_dataset("muzzz/coedit-cot-reasoning", split="train")
val_ds = load_dataset("muzzz/coedit-cot-reasoning", split="validation")

# %%
print("Train Dataset Info:")
print(train_ds)
print("features: ", train_ds.features)
print("example: ", train_ds[0])
print("--------------------------------")
print("\nValidation Dataset Info:")
print(val_ds)
print("features: ", val_ds.features)
print("example: ", val_ds[0])

# %%
# Check how many rows have empty reasoning column
empty_reasoning_count_train = sum(1 for example in train_ds if not example["reasoning"] or example["reasoning"].strip() == "")
empty_reasoning_count_val = sum(1 for example in val_ds if not example["reasoning"] or example["reasoning"].strip() == "")

print(f"Train dataset - Empty reasoning rows: {empty_reasoning_count_train} out of {len(train_ds)} ({empty_reasoning_count_train/len(train_ds)*100:.2f}%)")
print(f"Validation dataset - Empty reasoning rows: {empty_reasoning_count_val} out of {len(val_ds)} ({empty_reasoning_count_val/len(val_ds)*100:.2f}%)")

print("\nExamples with empty reasoning from train dataset:")
empty_examples_train = [example for example in train_ds if not example["reasoning"] or example["reasoning"].strip() == ""]
for i, example in enumerate(empty_examples_train[:3]):
    print(f"Example {i+1}:")
    print(f"  src: {example['src'][:100]}...")
    print(f"  reasoning: '{example['reasoning']}'")
    print(f"  tgt: {example['tgt'][:100]}...")


# %% [markdown]
# ### the dataset being used [here](https://huggingface.co/datasets/muzzz/coedit-cot-reasoning)

# %%
reasoning_start_token = "<think>"
reasoning_end_token = "</think>"

# The detailed system prompt for SFT phases (1 and 3)
system_prompt_sft = f"""You are an expert text editor. First, think step-by-step about the user's instruction and the source text. Place your reasoning inside {reasoning_start_token} and {reasoning_end_token} tags. Then, provide the final, edited text immediately after the closing tag. Your reasoning must follow a logical structure: Instruction analysis, Sentence analysis, Identify error(s), Apply correction(s), and Synthesized correction."""

# The simpler system prompt for the GRPO phase (2)
system_prompt_grpo = f"""You are an expert text editor. First, think step-by-step about the user's instruction and the source text. Place your reasoning inside {reasoning_start_token} and {reasoning_end_token} tags. Then, provide the final, edited text immediately after the closing tag."""

# %%
def format_sft_dataset(example, tokenizer, system_prompt):
    # For gemma_chatml, include system prompt as part of the user message
    user_content = system_prompt + "\n\n" + example["src"]
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": (example["reasoning"] or "") + "\n" + example["tgt"]},
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if formatted_text.startswith(tokenizer.bos_token):
        formatted_text = formatted_text[len(tokenizer.bos_token):]
    return {"text": formatted_text}


def format_grpo_dataset(example, system_prompt):
    """
    Formats data for GRPO. The 'prompt' is a list of messages for generation,
    and 'answer' is the ground truth for the reward function.
    """
    # For gemma_chatml, include system prompt as part of the user message
    user_content = system_prompt + "\n\n" + example["src"]
    return {
        "prompt": [
            {"role": "user", "content": user_content},
        ],
        "answer": example["tgt"],
    }

# %%
# --- Testing Chat Template Formatting ---

# --- [Phase 1 & 3] SFT Template Test ---
print("--- [Phase 1 & 3] SFT Template Test ---")
print("This template should include the detailed system prompt and the full model response.")

# Reset tokenizer to use the base gemma-3 template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# Format the messages just like the SFTTrainer will (system prompt included in user message)
user_content_sft = system_prompt_sft + "\n\nWhat is the capital of France?"
sft_messages = [
    {"role": "user", "content": user_content_sft},
    {"role": "assistant", "content": "The capital of France is Paris."},
]
sft_formatted = tokenizer.apply_chat_template(sft_messages, tokenize=False, add_generation_prompt=False)
print(sft_formatted)

# --- [Phase 2] GRPO Template Test ---
print("\n--- [Phase 2] GRPO Template Test ---")
print("This template should include the simpler system prompt and add the generation prompt at the end.")

# Format the messages just like the GRPOTrainer will (system prompt included in user message)
user_content_grpo = system_prompt_grpo + "\n\nWhat is the capital of France?"
grpo_messages = [
    {"role": "user", "content": user_content_grpo},
]
grpo_formatted = tokenizer.apply_chat_template(grpo_messages, tokenize=False, add_generation_prompt=True)
print(grpo_formatted)

# %%
# GRPO REWARD FUNCS
import re

extraction_pattern = re.compile(rf"{re.escape(reasoning_end_token)}(.*)", flags=re.DOTALL)

def reward_reasoning_structure(completions, **kwargs):
    """
    Rewards completions for having the correct <think> tags and the expected
    step-by-step reasoning structure based on your data generation prompt.
    """
    structural_keywords = [
        re.compile(r"instruction.*analysis", re.IGNORECASE),
        re.compile(r"sentence.*analysis", re.IGNORECASE),
        re.compile(r"identify.*error", re.IGNORECASE),
        re.compile(r"apply.*correction", re.IGNORECASE),
        re.compile(r"synthesized.*correction", re.IGNORECASE),
    ]
    scores = []
    for completion in completions:
        score = 0
        response_text = completion[0]["content"]

        # Base check for the enclosing <think> tags. This is fundamental.
        if reasoning_start_token in response_text and reasoning_end_token in response_text:
            score += 1.0  # Base reward for correct tag usage
        else:
            scores.append(-4.0) # Penalize heavily if tags are missing
            continue

        # Additive reward for each structural keyword found.
        num_keywords_found = sum(1 for keyword_regex in structural_keywords if keyword_regex.search(response_text))

        # Scale the reward. Max of +3 points for a perfectly structured response.
        if num_keywords_found > 0:
            score += (num_keywords_found / len(structural_keywords)) * 3.0

        scores.append(score)
    return scores

def reward_target_match(completions, answer, **kwargs):
    """
    Heavily rewards completions where the final extracted text exactly matches the target.
    """
    scores = []
    ground_truth_tgts = answer # The 'answer' here is `example["tgt"]` from format_grpo_dataset

    for completion, true_tgt in zip(completions, ground_truth_tgts):
        score = 0
        response_text = completion[0]["content"]

        # Extract the model's generated final answer
        extracted_match = extraction_pattern.search(response_text)

        if extracted_match:
            generated_text = extracted_match.group(1).strip()
            if generated_text == true_tgt.strip():
                score += 5.0  # High reward for an exact match
            else:
                score -= 2.0  # Penalize if it generates something, but it's wrong
        else:
            score -= 4.0  # Penalize heavily if it fails to produce any final answer

        scores.append(score)
    return scores

reward_funcs = [
    reward_reasoning_structure,
    reward_target_match,
]

# %%
# PHASE 1
print("preparing phase 1")
# Reset tokenizer to use the base gemma-3 template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

sft_dataset_full = train_ds.map(lambda x: format_sft_dataset(x, tokenizer, system_prompt_sft), batched=False)
sft_val_dataset = val_ds.map(lambda x: format_sft_dataset(x, tokenizer, system_prompt_sft), batched=False)

train_ds_subset = sft_dataset_full.shuffle(seed=42).select(range(len(sft_dataset_full) // 10))
val_ds_subset = sft_val_dataset.shuffle(seed=42).select(range(len(sft_val_dataset) // 10))

print(f"Full SFT dataset size: {len(sft_dataset_full)}")
print(sft_dataset_full[0]['text'][:500])
print(f"SFT subset size: {len(train_ds_subset)}")

# %%
from trl import SFTTrainer, SFTConfig

trainer_phase1 = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds_subset,
    eval_dataset=sft_val_dataset,
    args=SFTConfig(
        output_dir="./phase1",
        dataset_text_field="text",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        save_total_limit=12,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=4e-5,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
    )
)

# %% [markdown]
# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# %%
from unsloth.chat_templates import train_on_responses_only
trainer_phase1 = train_on_responses_only(
    trainer_phase1,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# %%
print(trainer_phase1.train_dataset)

# %%
print(trainer_phase1.train_dataset[100]["text"])
print()
print(trainer_phase1.train_dataset[100]["input_ids"])
print()
print(trainer_phase1.train_dataset[100]["attention_mask"])
print()
print(trainer_phase1.train_dataset[100]["labels"])

# %%
total_training_tokens = sum(len(x) for x in trainer_phase1.train_dataset['input_ids'])
print("Total training tokens in CoEdIT dataset:", total_training_tokens)

# %%
import numpy as np

lengths = [len(x) for x in trainer_phase1.train_dataset['input_ids']]
print(f"Token lengths stats:")
print(f"Min: {np.min(lengths)}")
print(f"Max: {np.max(lengths)}")
print(f"Mean: {np.mean(lengths)}")
print(f"Median: {np.median(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95)}")
print(f"99th percentile: {np.percentile(lengths, 99)}")

# %% [markdown]
# Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!

# %%
tokenizer.decode(trainer_phase1.train_dataset[100]["input_ids"])

# %% [markdown]
# Now let's print the masked out example - you should see only the answer is present:

# %%
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer_phase1.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [markdown]
# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# %%
print("--- starting phase 1 (primer): initial SFT ---")
trainer_stats = trainer_phase1.train()
#resume_from_checkpoint=True

# %%

print('phase one training complete')

q = "fix grammar: she go to store yesterday"
message = [{
    "role": "user",
    "content": [{"type": "text", "text": q}]
}]
print(message)

# %%
inputs = tokenizer.apply_chat_template(
    message,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    temperature=1.0,
    max_new_tokens=528,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)


# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %%
# PHASE 2
print("preparing phase 2")
# Reset tokenizer to use the base gemma-3 template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

grpo_dataset_full = train_ds.map(lambda x: format_grpo_dataset(x, system_prompt_grpo), remove_columns=list(train_ds.features))

subset_size = len(train_ds) // 10
grpo_train_subset = grpo_dataset_full.shuffle(seed=42).select(range(subset_size))
grpo_val_subset = val_ds.map(lambda x: format_grpo_dataset(x, system_prompt_grpo), remove_columns=list(val_ds.features)).shuffle(seed=42).select(range(len(val_ds) // 10))

print(f"Full GRPO dataset size: {len(grpo_dataset_full)}")
print(grpo_dataset_full[0])
print(f"GRPO subset size: {len(grpo_train_subset)}")

# %%
grpo_train_subset

# %%
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    temperature=1.0,
    stop=[tokenizer.eos_token],
    max_tokens=max_seq_length,
)


# %%

# # Reset generation config to avoid HybridCache issues
# model.generation_config.cache_implementation = None
# if hasattr(model.generation_config, 'use_cache'):
#     model.generation_config.use_cache = True

# # Also clear any cached states in the model config
# if hasattr(model.config, 'cache_implementation'):
#     model.config.cache_implementation = None



# %%
from trl import GRPOTrainer, GRPOConfig

trainer_phase2_grpo = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=reward_funcs,
    train_dataset=grpo_train_subset,
    args=GRPOConfig(
        output_dir='./phase2',
        vllm_sampling_params=vllm_sampling_params,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=max_seq_length // 2,
        max_completion_length=max_seq_length // 2,
        max_steps=200,
        save_steps=50,
        logging_steps=5,
        learning_rate=4e-6,
        report_to="none",
    ),
    eval_dataset=grpo_val_subset,
)


# %%
print("starting phase 2")
trainer_stats = trainer_phase2_grpo.train()

# %%
# PHASE 3
from trl import SFTTrainer, SFTConfig

trainer_phase3 = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset_full,
    eval_dataset=sft_val_dataset,
    args=SFTConfig(
        output_dir="./phase3",
        dataset_text_field="text",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        save_total_limit=12,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
    )
)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %%
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 5,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.01,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer_phase3.add_callback(early_stopping_callback)

# %%
print('starting phase 3')
phase3_stats = trainer_phase3.train()


# %% [markdown]
# <a name="Inference"></a>
# ### Inference

# %%
print(trainer_phase3.state.best_model_checkpoint)
print(trainer_phase3.state.best_metric)

# %%
messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Fix grammar in this sentence: hello there the angle from my nightmare the shadow in teh background of the morgue",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = True,
    return_tensors = "pt",
    return_dict = True,
).to("cuda")


from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 3072,
    # Recommended Gemma-3 settings!
    temperature = 0.3, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %%
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Fix grammar in this sentence: If engineers do not come up with new ideas, they cannot find best solution for the problems.",
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
    enable_thinking=True,
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 3072, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)

# %% [markdown]
#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# %%
messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %% [markdown]
# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# %%
model.save_pretrained("gemma-3n")  # Local saving
tokenizer.save_pretrained("gemma-3n")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving

# %% [markdown]
# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# %%
if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3N?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %% [markdown]
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3N-finetune`. Set `if False` to `if True` to let it run!

# %%
if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3N-finetune", tokenizer)

# %% [markdown]
# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# %%
if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3N-finetune", tokenizer,
        token = "hf_..."
    )

# %% [markdown]
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# %%
if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

# %% [markdown]
# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# %%
if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-3N-finetune-gguf",
        token = "hf_...",
    )


