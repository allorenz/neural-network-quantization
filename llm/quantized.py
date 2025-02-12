import json
import random
import time
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

class CustomQuantizeConfig(BaseQuantizeConfig):
    def __init__(self, bits, group_size):
        super().__init__()
        self.bits = bits
        self.group_size = group_size

int8_quantize_config = CustomQuantizeConfig(bits=8, group_size=128)
int4_quantize_config = CustomQuantizeConfig(bits=4, group_size=128)
int2_quantize_config = CustomQuantizeConfig(bits=4, group_size=128)

dataset = load_dataset("lambada", split="test")
sampled_dataset = random.sample(list(dataset), 1000)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

    return accuracy, tokens_per_second

results = {}

print("Evaluating the int8 quantized model...")
model_int8 = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config=int8_quantize_config)
accuracy, tokens_per_second = evaluate_model(model_int8, sampled_dataset, tokenizer)
results["int8_quantized_model"] = {
    "accuracy": accuracy,
    "tokens_per_second": tokens_per_second
}
del model_int8  # Free memory

print("Evaluating the int4 quantized model...")
model_int4 = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config=int4_quantize_config)
accuracy, tokens_per_second = evaluate_model(model_int4, sampled_dataset, tokenizer)
results["int4_quantized_model"] = {
    "accuracy": accuracy,
    "tokens_per_second": tokens_per_second
}
del model_int4  # Free memory

print("Evaluating the int2 quantized model...")
model_int2 = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config=int2_quantize_config)
accuracy, tokens_per_second = evaluate_model(model_int2, sampled_dataset, tokenizer)
results["int4_quantized_model"] = {
    "accuracy": accuracy,
    "tokens_per_second": tokens_per_second
}
del model_int2  # Free memory

# Save all results to a JSON file
with open("quantized_evaluation_results_autogptq.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Quantization results have been saved to 'quantized_evaluation_results_cpu.json'")
