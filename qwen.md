# Idea to finetune Qwen 1.5




## 1. Install Dependencies

```bash
pip install torch transformers datasets accelerate peft trl
```


## 2. Prepare the Model and Tokenizer

Load Qwen 1.5 Instruct and add `<think>` and `</think>` to the tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen1.5-1.8B-Instruct" # or try Qwen/Qwen2-1.5B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add <think> and </think> tokens (if not already present)
new_tokens = ["<think>", "</think>",...]
num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})


# Load model and resize embeddings
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.resize_token_embeddings(len(tokenizer))
```

## 3. Load and Adapt Magpie-Reasoning

- Load the dataset and format it for Qwen’s chat template.
- **Notes**:
  - Magpie uses `<｜User｜>` and `<｜Assistant｜>` with `<think>` tags, but Qwen’s default template is simpler (`### Instruction: ... ### Response: ...`). We’ll adapt Magpie’s format to Qwen’s style while preserving `<think>` tags.
  - If you want multi-`<think>` responses, you can split single `<think>` blocks (optional).

```python
from datasets import load_dataset

# Load Magpie-Reasoning-V2
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", split="train")

# Format for Qwen
def format_instruction(example):
    # Magpie uses 'instruction' and 'response'; adapt to Qwen's template
    response = example["response"]
    # Ensure <think> tags are preserved; optionally split into multiple <think> tags
    return {"text": f"### Instruction: {example['instruction']}\n### Response: {response}"}

# Apply formatting
dataset = dataset.map(format_instruction)

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1)

# Optional: Subset for faster testing (e.g., 10k samples)
dataset["train"] = dataset["train"].select(range(10000))
dataset["test"] = dataset["test"].select(range(1000))
```

## 3.a Optional: Split into Multiple `<think>` Tags

- If you want multi-`<think>` responses (e.g., one per step), modify the formatting:

```python
def split_think_tags(example):
    response = example["response"]
    if "<think>" in response and "</think>" in response:
        think_content = response.split("<think>")[1].split("</think>")[0]
        steps = think_content.split(". ")
        new_response = "".join(f"<think>{step.strip()}.</think>" for step in steps if step.strip())
        new_response += response.split("</think>")[1]
    else:
        new_response = response
    return {"text": f"### Instruction: {example['instruction']}\n### Response: {new_response}"}

dataset = dataset.map(split_think_tags)
```

Example of the output after splitting:

- **Before**: `<think>Subtract 3: 2x = 4. Divide by 2: x = 2.</think>`
- **After**: `<think>Subtract 3: 2x = 4.</think><think>Divide by 2: x = 2.</think>`

## 4. Fine-Tune with SFT

- Use `SFTTrainer` with LoRA and Accelerate for multi-GPU efficiency.

```python
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import Accelerator

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# Training arguments; Adjust based on your hardware!
training_args = TrainingArguments(
    output_dir="./qwen1.5-1.8b-titan",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    dataloader_num_workers=4
)

# Initialize Accelerator
accelerator = Accelerator()

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=512
)

# Prepare for multi-GPU
model, trainer = accelerator.prepare(model, trainer)

# Train
trainer.train()

# Save model
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    trainer.save_model("./qwen1.5-1.8b-titan")
    tokenizer.save_pretrained("./qwen1.5-1.8b-titan")
```

- **Run Command**:
  ```bash
  accelerate launch --num_processes 4 train.py  # 4 GPUs
  ```

## 5. Test the Model

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./qwen1.5-1.8b-titan", tokenizer=tokenizer)
prompt = "### Instruction: Solve x + 5 = 12\n### Response: "
output = pipe(prompt, max_new_tokens=200)[0]["generated_text"]
print(output)


## 6. Kết quả mong đợi

```
- **Single `<think>`**:
  ```
  ### Instruction: Solve x + 5 = 12
  ### Response: <think>To solve x + 5 = 12, subtract 5 from both sides: x = 12 - 5. That gives x = 7.</think>
  ```
- **Multi-`<think>`**:
  ```
  ### Instruction: Solve x + 5 = 12
  ### Response: <think>To solve x + 5 = 12, subtract 5 from both sides.</think><think>Calculate 12 - 5 = 7.</think><think>So, x = 7.</think>
  ```


