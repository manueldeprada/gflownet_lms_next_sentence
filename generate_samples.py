import argparse
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from functools import partial
from utils import (
    generate_and_return_termination_logprob,
    FrozenModelSentenceGivenPrompt,
)


def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config) 
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False), strict=False)
    model.eval()
    return model

def generate_samples(model, tokenizer, prompt, n_samples=4, min_sentence_len=1, max_sentence_len=20):
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device).squeeze(0)
    end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(".")
    prompt_batch = prompt_ids.unsqueeze(0).expand(n_samples, -1)
    # Create an empty illegal token mask (no constraints by default)
    illegal_token_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    
    generated_text, log_pf, _, _, _ = generate_and_return_termination_logprob(
        model,
        prompt_batch,
        termination_token_id=end_of_sentence_token_id,
        reward_fn=None,
        vocab_naughty_mask=illegal_token_mask,
        min_len=min_sentence_len,
        max_len=max_sentence_len,
        temperature=1.0,
        skip_rewards=True,
    )
    
    # generated_sentences, _ = model(prompt_batch, n_samples=n_samples)
    
    generated_sentences = tokenizer.batch_decode(generated_text[:, len(prompt_ids):])
    # retain only first occurence of "." in each sentence
    generated_sentences = [sent.replace(".", "").strip() for sent in generated_sentences]
    
    return generated_sentences, log_pf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate samples from.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--min_len", type=int, default=1, help="Minimum sentence length.")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum sentence length.")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

    samples, log_pf = generate_samples(
        model, 
        tokenizer, 
        args.prompt, 
        n_samples=args.n_samples,
        min_sentence_len=args.min_len,
        max_sentence_len=args.max_len
    )
    
    print(f"Prompt: {args.prompt}")
    for i, sample in enumerate(samples):
        print(f"Sample {i + 1} : {sample}")
    print(f"Log probabilities: {log_pf}")