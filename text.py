from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from dataset import disaster_headlines

# Load pre-trained GPT-2 medium model and tokenizer
model_name = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Fixed device assignment
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set end-of-sequence token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Curate seed data (real disaster news headlines)
seed_data = disaster_headlines

# Tokenize seed data
tokenized_seed_data = tokenizer(seed_data, truncation=True, padding=True, return_tensors="pt").to(device)

# Extract input_ids and attention_mask
input_ids = tokenized_seed_data["input_ids"]
attention_mask = tokenized_seed_data["attention_mask"]

# Combine input_ids and attention_mask into a list of dictionaries
train_data = [{"input_ids": input_ids[i], "attention_mask": attention_mask[i]} for i in range(len(seed_data))]

# Fine-tune the model on the seed data
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
)
trainer.train()

# Generate disaster-related headlines using the fine-tuned model
prompt = "Philippine Disaster-related headline prompt: "
# Tokenize the prompt
prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
# Generate headlines
# Tuning Hyperparameters
# Experiment with different hyperparameter values here
generated_headlines = model.generate(
    input_ids=prompt_tokens["input_ids"],
    attention_mask=prompt_tokens["attention_mask"],
    max_length=50,
    num_return_sequences=10,
    temperature=0.7,  # Experiment with different temperature values
    top_k=30,  # Experiment with different top-k values
    top_p=0.6,  # Experiment with different top-p values
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True
)
# Decode generated headlines
decoded_headlines = [tokenizer.decode(headline, skip_special_tokens=True) for headline in generated_headlines["sequences"]]

# Print generated headlines
for headline in decoded_headlines:
    print(headline)
