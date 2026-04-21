import os
# JAX memory optimizations
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import json
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from tokenizers import Tokenizer
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
from tunix.sft import peft_trainer
from tunix.sft.utils import make_causal_attn_mask, build_positions_from_mask
from qwix import LoraProvider, apply_lora_to_model

# Constants
MODEL_DIR = "models/gemma-4-E2B-it"
TRAIN_DATA = "data/train.jsonl"
OUTPUT_DIR = "checkpoints_tpu"
SEQ_LEN = 1024 # Increased back to 1024 for better performance
BATCH_SIZE = 4 # Increased to leverage 4 cores (1 per core)
LEARNING_RATE = 2e-5
LORA_RANK = 16 # Increased rank for better quality
LORA_ALPHA = 32

def get_tokenizer():
    return Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))

def format_example(example, tokenizer):
    # Gemma 4 Chat Template
    bos = "<bos>"
    eos = "<eos>"
    sot = "<|turn|>"
    eot = "<turn|>"
    
    prompt = f"{bos}{sot}user\n{example['instruction']}{eot}\n{sot}model\n"
    response = f"{example['response']}{eot}{eos}"
    
    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids
    
    input_tokens = prompt_ids + response_ids
    input_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    
    if len(input_tokens) > SEQ_LEN:
        input_tokens = input_tokens[:SEQ_LEN]
        input_mask = input_mask[:SEQ_LEN]
    else:
        pad_len = SEQ_LEN - len(input_tokens)
        input_tokens += [0] * pad_len
        input_mask += [0] * pad_len
        
    return np.array(input_tokens), np.array(input_mask)

def train_ds_gen():
    tokenizer = get_tokenizer()
    while True:
        with open(TRAIN_DATA, "r") as f:
            for line in f:
                # Accumulate for batch
                batch_tokens = []
                batch_masks = []
                
                for _ in range(BATCH_SIZE):
                    example = json.loads(line)
                    tokens, mask = format_example(example, tokenizer)
                    batch_tokens.append(tokens)
                    batch_masks.append(mask)
                    
                    try:
                        line = next(f)
                    except StopIteration:
                        break # End of file
                
                if not batch_tokens: break

                tokens_np = np.array(batch_tokens)
                masks_np = np.array(batch_masks)
                
                valid_mask = (tokens_np != 0) 
                positions = build_positions_from_mask(valid_mask)
                attn_mask = make_causal_attn_mask(valid_mask)
                
                yield {
                    "input_tokens": jnp.array(tokens_np),
                    "input_mask": jnp.array(masks_np),
                    "positions": jnp.array(positions),
                    "attention_mask": jnp.array(attn_mask),
                }

def main():
    # 0. Setup Mesh
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Using {num_devices} devices: {devices}")
    
    # Create mesh for FSDP + TP
    # For v5e-4, topology is 2x2. We can use all 4 for FSDP.
    mesh = jax.sharding.Mesh(np.array(devices).reshape(-1, 1), ('fsdp', 'tp'))

    # 1. Config
    config = gemma4_model.ModelConfig.gemma4_e2b()
    # Use bfloat16 for both computation and parameters
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    with mesh:
        # 2. Model Initialization
        print("Loading model from safetensors...")
        model = params_safetensors.create_model_from_safe_tensors(
            file_dir=MODEL_DIR,
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16
        )
        
        # 3. Apply LoRA (Standard BF16)
        print("Applying LoRA...")
        rngs = nnx.Rngs(0)
        provider = LoraProvider(rank=LORA_RANK, alpha=LORA_ALPHA)
        
        # Trace with full batch size to establish sharding
        dummy_tokens = jnp.zeros((BATCH_SIZE, 8), dtype=jnp.int32)
        dummy_mask = make_causal_attn_mask(jnp.ones((BATCH_SIZE, 8), dtype=jnp.bool_))
        model = apply_lora_to_model(model, provider, dummy_tokens, attention_mask=dummy_mask, rngs=rngs)

        # 4. Trainer Setup
        training_config = peft_trainer.TrainingConfig(
            max_steps=500, # Sufficient for demonstration
            eval_every_n_steps=50,
            gradient_accumulation_steps=1,
            checkpoint_root_directory=os.path.abspath(OUTPUT_DIR)
        )

        optimizer = optax.adamw(learning_rate=LEARNING_RATE)
        
        trainer = peft_trainer.PeftTrainer(
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )

        # 5. Training
        print("Starting training on TPU Pod...")
        trainer.train(train_ds_gen())
        print("Training completed.")

if __name__ == "__main__":
    main()
