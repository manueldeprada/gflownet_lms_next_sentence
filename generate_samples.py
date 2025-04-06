import argparse
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from functools import partial
from utils import (
    generate_and_return_termination_logprob,
    FrozenModelSentenceGivenPrompt,
)
from decoders.toolbox import compute_true_logprobs

def compute_transition_scores_nonbeam(sequences, scores, normalize_logits=True, eos_token_id=None):
    if len(sequences.shape) == 1:
        sequences = sequences.unsqueeze(0)
    scores = torch.stack(scores, dim=1) if not isinstance(scores, torch.Tensor) else scores # batch_size x seq_len x vocab_size
    if normalize_logits:
        scores = scores.log_softmax(dim=-1)
    # labels = sequences[:, 1:]
    transition_scores = scores[:,:,:].gather(dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)
    if eos_token_id is not None:
        #mask: all but the first eos token
        mask = sequences.eq(eos_token_id)
        not_first_eos = (mask.cumsum(dim=1) != 1)
        mask = mask & not_first_eos
        transition_scores = transition_scores.masked_fill(mask, 0.0)
    return transition_scores

def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
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

    model.disable_adapter_layers()
    # evaluate prob over original model
    output = model(generated_text,return_dict=True)
    model.enable_adapter_layers()

    transition_scores = compute_transition_scores_nonbeam(
            generated_text[:, len(prompt_ids):],
            output.logits[:, len(prompt_ids)-1:-1],
            normalize_logits=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("."),
        )

    # retain only first occurence of "." in each sentence
    generated_sentences = [sent.replace(".", "").strip() for sent in generated_sentences]
    
    return generated_sentences, transition_scores.sum(dim=1), log_pf
