import numpy as np
import torch
from generate_samples import load_model, generate_samples, compute_transition_scores_nonbeam
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers.util import cos_sim
import pandas as pd
from pathlib import Path

models_dir = "runs/lora_checkpoints_new"
diversity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def compute_diversity(samples):
    embeddings = diversity_model.encode(samples, show_progress_bar=False)
    sim = cos_sim(embeddings, embeddings)
    indices = torch.triu_indices(len(samples), len(samples), offset=1)
    diversity = 1 - sim[indices[0], indices[1]].mean().item()
    return diversity, np.linalg.det(np.array(embeddings) @ np.array(embeddings).T)

def get_prompts():
    with open("data/openwebtext/prompts.txt", "r") as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts][950:1000]
    return prompts

def compute_gflownet(file):
    model = load_model(file)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompts = get_prompts()
    results = np.zeros((len(prompts), 6))
    for i, prompt in enumerate(prompts):
        samples, log_pf, own_log_pf = generate_samples(
            model,
            tokenizer,
            prompt,
            n_samples=10,
            min_sentence_len=1,
            max_sentence_len=30
        )
        print(f"Prompt {i + 1}, mean_own_liks: {own_log_pf.sum(dim=-1).mean().item()}")
        samples_diversity, samples_det = compute_diversity(samples)
        samples_likelihood = log_pf.sum(dim=-1)
        avg_likelihood = samples_likelihood.mean().item()
        max_likelihood = samples_likelihood.max().item()
        results[i] = np.array([samples_diversity, samples_det, avg_likelihood, max_likelihood, own_log_pf.sum(dim=-1).mean().item(), own_log_pf.sum(dim=-1).max().item()])
    final_results = np.mean(results, axis=0)
    return final_results

def compute_huggingface(temp=1.0, do_sample=True, n_beams=1, num_beam_groups=1, top_p=1.0, num_return_sequences=10, diversity_penalty=0.0, length_penalty=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompts = get_prompts()
    results = np.zeros((len(prompts), 6))
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i + 1}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
            num_beams=n_beams,
            num_beam_groups=num_beam_groups,
            top_k=0,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
            max_new_tokens=30,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            pad_token_id=tokenizer.convert_tokens_to_ids("."),
            eos_token_id=tokenizer.convert_tokens_to_ids("."),
            forced_eos_token_id=tokenizer.convert_tokens_to_ids("."),
        )
        samples_diversity, samples_det = compute_diversity(tokenizer.batch_decode(output.sequences[:, prompt_len:]))
        transition_scores = compute_transition_scores_nonbeam(
            output.sequences[:, prompt_len:],
            output.logits,
            normalize_logits=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("."),
        )
        transition_scores_own = compute_transition_scores_nonbeam(
            output.sequences[:, prompt_len:],
            output.scores,
            normalize_logits=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("."),
        )
        samples_likelihood = transition_scores.sum(dim=1) if n_beams == 1 else output.sequences_scores #to do
        avg_likelihood = samples_likelihood.mean().item()
        max_likelihood = samples_likelihood.max().item()
        results[i] = np.array([samples_diversity, samples_det, avg_likelihood, max_likelihood, transition_scores_own.mean().item(), transition_scores_own.max().item()])
    final_results = np.mean(results, axis=0)
    print(f"results: {final_results}")
    return final_results


def gflownet():
    results = []
    all_dirs = Path(models_dir).glob(f"peft_lora_temp_*")
    for file in sorted(all_dirs):
        temp = float(file.name.split("_")[-1])
        print(f"Temperature: {temp}")
        out = compute_gflownet(file)
        results.append([
            "gflownet",
            temp,
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5]
        ])
    df_results = pd.DataFrame(results, columns=["type", "temp", "diversity", "det", "avg_likelihood", "max_likelihood", "avg_likelihood_own", "max_likelihood_own"])
    print(df_results)
    df_results.to_csv("gflownet_results_new.csv", index=False)

def huggingface():
    results = []
    for temp in [0.3, 0.5, 0.8, 1.0, 1.2]:
        print(f"Temperature: {temp}")
        out = compute_huggingface(temp)
        results.append([
            "ancestral",
            temp,
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5]
        ])
    for top_p in [0.95]:
        print(f"Top-p: {top_p}")
        out = compute_huggingface(top_p=top_p)
        results.append([
            "top-p",
            top_p,
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5]
        ])
    for n_beams in [10]:
        print(f"Num beams: {n_beams}")
        out = compute_huggingface(n_beams=n_beams, num_beam_groups=n_beams, do_sample=False, 
                                  diversity_penalty=1.0, length_penalty=0.0)
        results.append([
            "beam",
            n_beams,
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5]
        ])
    df_results = pd.DataFrame(results, columns=["type", "temp", "diversity", "det", "avg_likelihood", "max_likelihood", "avg_likelihood_own", "max_likelihood_own"])
    print(df_results)
    csv_file = "ancestral_results.csv"
    df_results.to_csv(csv_file, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experiments for comparing GFlowNet and HuggingFace models')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # GFlowNet command
    gfn_parser = subparsers.add_parser('gfnet', help='Run GFlowNet experiments')
    
    # HuggingFace command
    hf_parser = subparsers.add_parser('hf', help='Run HuggingFace experiments')
    
    args = parser.parse_args()
    
    if args.command == 'gfnet':
        gflownet()
    elif args.command == 'hf':
        huggingface()
    else:
        parser.print_help()