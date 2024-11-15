import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to eos_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("lambada", split="test")
sampled_dataset  = random.sample(list(dataset), 1000)

def evaluate_model(model, dataset, tokenizer, top_k=[1, 5, 10]):
    model.eval()
    correct_predictions = {k: 0 for k in top_k}
    total_time = {k: 0 for k in top_k}
    total_samples = len(dataset)

    for sample in dataset:
        input_text = sample["text"]
        words = input_text.strip().split()
        if len(words) < 2:
            continue
        context = " ".join(words[:-1])
        target_word = words[-1]

        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        for k in top_k:
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    num_beams=k,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            total_time[k] += (end_time - start_time)

            generated_words = [
                tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True).strip()
                for output in outputs
            ]

            if target_word in generated_words[:k]:
                correct_predictions[k] += 1

            print(f"Context: {context}")
            print(f"Target word: '{target_word}'")
            print(f"Generated words (Top-{k}): {generated_words[:k]}")

    accuracy = {k: correct_predictions[k] / total_samples for k in top_k}
    tokens_per_second = {k: total_samples / total_time[k] for k in top_k}

    results = {
        "accuracy": accuracy,
        "tokens_per_second": tokens_per_second
    }
    with open("evaluation_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    return accuracy, tokens_per_second

top_k_values = [1, 5, 10]
print("Evaluating the regular model with top-k accuracy...")
accuracy, tokens_per_second = evaluate_model(model, sampled_dataset, tokenizer, top_k=top_k_values)




