import os
# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"
# Disable JIT to avoid all tracing/mutation issues
os.environ["JAX_DISABLE_JIT"] = "1"

import json
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
import sys

def manual_generate_no_jit(model, tokenizer, prompt, max_len=16):
    token_ids = tokenizer.encode(prompt).ids
    input_ids = jnp.array([token_ids])
    
    generated = []
    print(f"Generating {max_len} tokens...")
    for i in range(max_len):
        seq_len = input_ids.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
        
        # Gemma 4 returns (logits, cache)
        logits, _ = model(input_ids, positions=positions, attention_mask=mask)
        next_token = jnp.argmax(logits[0, -1, :])
        
        if next_token == tokenizer.token_to_id("<eos>"):
            break
            
        generated.append(int(next_token))
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        print(".", end="", flush=True)
        
    print("\nDone.")
    return tokenizer.decode(generated)

def run_eval(model_dir, output_file, num_samples=3):
    print(f"Loading model from {model_dir}...")
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_flash_attention=False)

    model = params_safetensors.create_model_from_safe_tensors(
        file_dir=model_dir,
        config=config,
        dtype=jnp.bfloat16
    )
    
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    
    test_data = "data/test.jsonl"
    results = []
    
    with open(test_data, "r") as f:
        lines = f.readlines()
        
    for line in lines[:num_samples]:
        example = json.loads(line)
        instruction = example["instruction"]
        # Correct Gemma 4 IT prompt format
        prompt = f"<bos><|turn>user\n{instruction}<turn|>\n<|turn>model\n"
        
        response = manual_generate_no_jit(model, tokenizer, prompt)
            
        results.append({
            "instruction": instruction,
            "expected": example["response"],
            "generated": response
        })
        print(f"\nQ: {instruction}\nA: {response}\n")
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python manual_eval_no_jit.py <model_dir> <output_file>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2])
