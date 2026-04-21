import os
import json
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
from tunix.generate import sampler as sampler_lib
import sys

# Use TPU
os.environ["JAX_PLATFORMS"] = "tpu"

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._bos_id = tokenizer.token_to_id("<bos>")
        self._eos_id = tokenizer.token_to_id("<eos>")
        self._pad_id = tokenizer.token_to_id("<pad>")
        if self._pad_id is None: self._pad_id = self._eos_id
    def bos_id(self): return self._bos_id
    def eos_id(self): return self._eos_id
    def pad_id(self): return self._pad_id
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids): return self.tokenizer.decode(ids)

def run_eval(model_dir, output_file, num_samples=10):
    print(f"Loading model from {model_dir} on TPU...")
    
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(-1, 1), ('fsdp', 'tp'))
    
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(
        config, 
        dtype=jnp.bfloat16, 
        param_dtype=jnp.bfloat16,
        use_flash_attention=False
    )

    with mesh:
        model = params_safetensors.create_model_from_safe_tensors(
            file_dir=model_dir,
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16
        )
        
        raw_tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tokenizer = TokenizerWrapper(raw_tokenizer)
        
        cache_config = sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim
        )
        
        sampler = sampler_lib.Sampler(
            transformer=model,
            tokenizer=tokenizer,
            cache_config=cache_config
        )

        print(f"Running evaluation on {num_samples} samples...")
        test_data = "data/test.jsonl"
        results = []
        
        with open(test_data, "r") as f:
            lines = f.readlines()
            
        for line in lines[:num_samples]:
            example = json.loads(line)
            instruction = example["instruction"]
            prompt = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n"
            
            output_data = sampler(
                prompt,
                max_generation_steps=128,
                temperature=0.1,
                top_p=0.9,
                return_logits=False
            )
            
            response = output_data[0] if isinstance(output_data, list) else str(output_data)
                
            results.append({
                "instruction": instruction,
                "expected": example["response"],
                "generated": response
            })
            print(f"Q: {instruction}\nA: {response}\n")
            
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_tpu_v2.py <model_dir> <output_file>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2])
