import numpy as np
import torch
from generate_samples import load_model, generate_samples
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers.util import cos_sim
import pandas as pd
from tqdm import tqdm


temp_to_name = {
    # 0.825: "2025-03-22_00-12-06", # [  0.78071487   0.20636478 -75.96899086 -30.27881896]
    0.85: "2025-03-22_10-08-33",
    0.875: "2025-03-22_10-08-33",
    0.9: "2025-03-22_10-08-39",
    0.925: "2025-03-22_10-08-42",
    0.95: "2025-03-22_10-08-46"
}

models_dir = "runs/checkpoints"
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

def compute_gflownet(temp):
    model = load_model(f"{models_dir}/{temp_to_name[temp]}/last.ckpt")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompts = get_prompts()
    results = np.zeros((len(prompts), 4))
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), leave=False):
        print(f"Prompt {i + 1}")
        samples, log_pf = generate_samples(
            model,
            tokenizer,
            prompt,
            n_samples=10,
            min_sentence_len=1,
            max_sentence_len=30
        )
        samples_diversity, samples_det = compute_diversity(samples)
        samples_likelihood = log_pf.sum(dim=-1)
        avg_likelihood = samples_likelihood.mean().item()
        max_likelihood = samples_likelihood.max().item()
        results[i] = np.array([samples_diversity, samples_det, avg_likelihood, max_likelihood])
    final_results = np.mean(results, axis=0)
    print(f"results: {final_results}")
    return final_results

def compute_huggingface(temp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompts = get_prompts()
    results = np.zeros((len(prompts), 4))
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), leave=False):
        print(f"Prompt {i + 1}. ", end="")
        samples, log_pf = model.generate(
            *tokenizer(prompt, return_tensors="pt"),
            do_sample=True,
            temperature=temp,
            top_k=0,
            max_length=30,
            num_return_sequences=10
        )
        samples_diversity, samples_det = compute_diversity(samples)
        samples_likelihood = log_pf.sum(dim=-1)
        avg_likelihood = samples_likelihood.mean().item()
        max_likelihood = samples_likelihood.max().item()
        results[i] = np.array([samples_diversity, samples_det, avg_likelihood, max_likelihood])
    final_results = np.mean(results, axis=0)
    print(f"results: {final_results}")
    return final_results


def main():
    results = []
    for temp in temp_to_name.keys():
        print(f"Temperature: {temp}")
        results = compute_gflownet(temp)
        results.append([
            temp,
            results[0],
            results[1],
            results[2],
            results[3]
        ])
    df_results = pd.DataFrame(results, columns=["temp", "diversity", "det", "avg_likelihood", "max_likelihood"])
    df_results.to_csv("gflownet_results.csv", index=False)


if __name__ == "__main__":
    main()