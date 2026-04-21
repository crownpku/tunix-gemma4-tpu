import os
import json
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
from tunix.generate import sampler as sampler_lib

# Constants
MODEL_DIR = "models/gemma-4-E2B-it-finetuned"
TEST_DATA = "data/test.jsonl"
RESULTS_FILE = "results/finetuned_results.json"
MAX_SEQ_LEN = 1024

def main():
    # 1. Config
    config = gemma4_model.ModelConfig.gemma4_e2b()
    # Use bfloat16 for inference
    config = dataclasses.replace(config, dtype=jnp.bfloat16)

    print("Loading merged model...")
    model = params_safetensors.create_model_from_safe_tensors(
        file_dir=MODEL_DIR,
        config=config,
        dtype=jnp.bfloat16
    )
    
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    
    # 2. Setup Sampler
    cache_config = sampler_lib.CacheConfig(
        cache_size=MAX_SEQ_LEN,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim
    )
    
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config
    )

    print("Running evaluation...")
    results = []
    with open(TEST_DATA, "r") as f:
        lines = f.readlines()
        
    for line in lines[:50]: # 50 samples
        example = json.loads(line)
        instruction = example["instruction"]
        
        # Format prompt
        prompt = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n"
        
        # Generate
        # Sampler.sample returns a list of token IDs
        output_ids = sampler.sample(
            prompt,
            max_len=256,
            temperature=0.7,
            top_p=0.9
        )
        
        # Sampler might return a string or list of IDs depending on implementation
        # Looking at Sampler.sample signature in base_sampler might help, 
        # but usually it handles strings if tokenizer is provided.
        response = output_ids # If it's already a string
        if isinstance(output_ids, list) or isinstance(output_ids, jnp.ndarray):
            response = tokenizer.decode(np.array(output_ids).tolist())
            
        results.append({
            "instruction": instruction,
            "expected": example["response"],
            "generated": response
        })
        print(f"Q: {instruction}\nA: {response}\n")
                
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
