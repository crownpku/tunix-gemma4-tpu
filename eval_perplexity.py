import os
os.environ["JAX_PLATFORMS"] = "tpu"

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

def compute_perplexity(model, tokenizer, data_file, num_samples=10):
    print(f"Computing perplexity for {data_file}...")
    total_loss = 0.0
    total_tokens = 0
    
    with open(data_file, "r") as f:
        lines = f.readlines()
        
    for line in lines[:num_samples]:
        example = json.loads(line)
        # Verify tokens
        instruction = example['instruction']
        response = example['response']
        text = f"<bos><|turn>user\n{instruction}<turn|>\n<|turn>model\n{response}<eos>"
        
        token_ids = tokenizer.encode(text).ids
        input_ids = jnp.array([token_ids])
        
        seq_len = input_ids.shape[1]
        pos = jnp.arange(seq_len)[None, :]
        # Use patch mask if needed, but here simple tril is fine
        mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
        
        logits, _ = model(input_ids, positions=pos, attention_mask=mask)
        
        # DEBUG: Print some predictions
        preds = jnp.argmax(logits[0, :-1, :], axis=-1)
        actual = input_ids[0, 1:]
        print(f"DEBUG Sample {line[:20]}... Predictions: {preds[:10]}")
        print(f"DEBUG Sample {line[:20]}... Actual     : {actual[:10]}")
        print(f"DEBUG Decoded Preds: {tokenizer.decode(preds[:10].tolist())}")
        print(f"DEBUG Decoded Actual: {tokenizer.decode(actual[:10].tolist())}")
        
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        label_log_probs = jnp.take_along_axis(log_probs, shift_labels[:, None], axis=-1).squeeze()
        
        total_loss -= jnp.sum(label_log_probs)
        total_tokens += len(shift_labels)
        
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return float(perplexity), float(avg_loss)

def main(model_dir):
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_flash_attention=False)

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(-1, 1), ('fsdp', 'tp'))
    
    with mesh:
        model = params_safetensors.create_model_from_safe_tensors(
            file_dir=model_dir,
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16
        )
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        print(f"DEBUG: Tokenizer vocab size: {tokenizer.get_vocab_size()}")
        
        # Verify weight loading
        emb_weight = model.embedder.input_embedding[...]
        print(f"DEBUG: Input embedding mean: {jnp.mean(emb_weight)}, std: {jnp.std(emb_weight)}, max: {jnp.max(emb_weight)}")
        ple_weight = model.embedder.per_layer_input_embedding[...]
        print(f"DEBUG: PLE table mean: {jnp.mean(ple_weight)}, std: {jnp.std(ple_weight)}, max: {jnp.max(ple_weight)}")
        q_proj_weight = model.layers[0].attn.q_einsum.w[...]
        print(f"DEBUG: Q_proj weight mean: {jnp.mean(q_proj_weight)}, std: {jnp.std(q_proj_weight)}, max: {jnp.max(q_proj_weight)}")
        
        ppl, loss = compute_perplexity(model, tokenizer, "data/test.jsonl")
        print(f"MODEL: {model_dir}")
        print(f"PERPLEXITY: {ppl:.4f}")
        print(f"LOSS: {loss:.4f}")

if __name__ == "__main__":
    main(sys.argv[1])
