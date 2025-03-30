import numpy as np
import torch
from generate_samples import load_model, generate_samples
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers.util import cos_sim
import pandas as pd


temp_to_name = {
    # 0.825: "2025-03-22_00-12-06", # [  0.78071487   0.20636478 -75.96899086 -30.27881896], [  0.80001899   0.23968567 -73.68188591 -25.84072336]
    # 0.85: "2025-03-22_10-08-33",# [  0.78038039   0.21906277 -72.01942116 -26.20678624]
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
    for i, prompt in enumerate(prompts):
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

def compute_transition_scores_nonbeam(sequences, scores, normalize_logits=True):
    if len(sequences.shape) == 1:
        sequences = sequences.unsqueeze(0)
    scores = torch.stack(scores, dim=1) # batch_size x seq_len x vocab_size
    if normalize_logits:
        scores = scores.log_softmax(dim=-1)
    sequences = sequences[:, 1:]
    transition_scores = scores.gather(dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)
    return transition_scores

def compute_huggingface(temp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompts = get_prompts()
    results = np.zeros((len(prompts), 4))
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i + 1}. ", end="")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_k=0,
            max_new_tokens=30,
            num_return_sequences=10,
            return_dict_in_generate=True,
            output_scores=True,
        )
        samples_diversity, samples_det = compute_diversity(tokenizer.batch_decode(output.sequences[:, prompt_len:]))
        samples_likelihood = compute_transition_scores_nonbeam(
            output.sequences[:, prompt_len:],
            output.scores,
            normalize_logits=True
        ).sum(dim=1)
        avg_likelihood = samples_likelihood.mean().item()
        max_likelihood = samples_likelihood.max().item()
        results[i] = np.array([samples_diversity, samples_det, avg_likelihood, max_likelihood])
    final_results = np.mean(results, axis=0)
    print(f"results: {final_results}")
    return final_results


def gflownet():
    results = []
    for temp in temp_to_name.keys():
        print(f"Temperature: {temp}")
        out = compute_gflownet(temp)
        results.append([
            "gflownet",
            temp,
            out[0],
            out[1],
            out[2],
            out[3]
        ])
    df_results = pd.DataFrame(results, columns=["type", "temp", "diversity", "det", "avg_likelihood", "max_likelihood"])
    df_results.to_csv("gflownet_results.csv", index=False)

def huggingface():
    results = []
    for temp in temp_to_name.keys():
        print(f"Temperature: {temp}")
        out = compute_huggingface(temp)
        results.append([
            "ancestral",
            temp,
            out[0],
            out[1],
            out[2],
            out[3]
        ])
    df_results = pd.DataFrame(results, columns=["type", "temp", "diversity", "det", "avg_likelihood", "max_likelihood"])
    print(df_results)
    saved_csv = pd.read_csv("gflownet_results.csv")
    # add type column if not present
    if "type" not in saved_csv.columns:
        saved_csv["type"] = "gflownet"
    df_results = pd.concat([saved_csv, df_results])
    df_results.to_csv("gflownet_results.csv", index=False)

if __name__ == "__main__":
    # gflownet()
    huggingface()