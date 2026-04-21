import os
os.environ["JAX_PLATFORMS"] = "tpu"

import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors
import sys

def run_quick_test(model_dir):
    print(f"Loading model from {model_dir}...")
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
        
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        
        instruction = "What Does Medicare IME Stand For?"
        prompt = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n"
        token_ids = tokenizer.encode(prompt).ids
        input_ids = jnp.array([token_ids])
        
        # Standalone JIT function for generation step
        @nnx.jit
        def decode_step(tokens):
            seq_len = tokens.shape[1]
            pos = jnp.arange(seq_len)[None, :]
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            logits = model(tokens, positions=pos, attention_mask=mask)
            return jnp.argmax(logits[0, -1, :])

        print("Generating response...")
        generated = []
        for _ in range(50):
            next_token = decode_step(input_ids)
            if next_token == tokenizer.token_to_id("<eos>"):
                break
            generated.append(int(next_token))
            input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
            
        response = tokenizer.decode(generated)
        print(f"\nPROMPT: {instruction}")
        print(f"RESPONSE: {response}\n")

if __name__ == "__main__":
    run_quick_test(sys.argv[1])
