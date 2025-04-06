import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
import os
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

models_dir = "runs/checkpoints"
output_dir = "runs/lora_checkpoints"
os.makedirs(output_dir, exist_ok=True)

for folder_name in Path(models_dir).iterdir():
    checkpoint_path = f"{folder_name}/last.ckpt"

    # 1. Load model skeleton (no weights)
    config = AutoConfig.from_pretrained("gpt2-xl")
    model = AutoModelForCausalLM.from_config(config).to(device)

    # 2. Wrap with LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)

    # 3. Load checkpoint (full Lightning checkpoint)
    print(f"Loading checkpoint for from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    temp = state_dict['hyper_parameters']['reward_temp_end']
    print(f"Temperature: {temp}")

    if "state_dict" in state_dict:  # If it's a Lightning checkpoint
        state_dict = state_dict["state_dict"]

    # 4. Clean up keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'model.' prefix if present
        if k.startswith("model."):
            k = k[len("model."):]
        # Adjust keys for transformer layers
        if k.startswith("transformer."):
            k = f"base_model.model.transformer.{k}"
        elif k == "lm_head.weight":
            k = "base_model.model.lm_head.weight"
        cleaned_state_dict[k] = v

    # 5. Load into PEFT-wrapped model
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys during load: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys during load: {unexpected}")

    # 6. Save only the PEFT adapter
    adapter_save_path = os.path.join(output_dir, f"peft_lora_temp_{temp}")
    print(f"Saving PEFT adapter to {adapter_save_path}")
    model.save_pretrained(adapter_save_path)

print("Done converting all checkpoints to PEFT adapters!")
