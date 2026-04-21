import os
# Force JAX to use CPU for merging to leverage 188GB RAM
os.environ["JAX_PLATFORMS"] = "cpu"

import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.safetensors_saver import save_lora_merged_model_as_safetensors, join_path
from tunix.sft.utils import make_causal_attn_mask
from qwix import LoraProvider, apply_lora_to_model
from tunix.sft.checkpoint_manager import CheckpointManager

# Constants
BASE_MODEL_DIR = "models/gemma-4-E2B-it"
LORA_CHECKPOINT_DIR = "checkpoints_tpu/500"
MERGED_OUTPUT_DIR = "models/gemma-4-E2B-it-finetuned"
LORA_RANK = 16
LORA_ALPHA = 32

def custom_layer_extractor(model):
    """Correctly extracts LoRA pairs and splits multi-part weights (like KV)."""
    extracted = {}
    for path, value in nnx.iter_graph(model):
        if isinstance(value, nnx.LoRAParam):
            full_path = join_path(path)
            suffix = None
            if full_path.endswith('_lora_a'): suffix = '_lora_a'
            elif full_path.endswith('_lora_b'): suffix = '_lora_b'
            
            if suffix:
                raw_path = full_path[:-len(suffix)]
                layer_path = raw_path
                for ext in ['.kernel', '.w', '.value', '.scale']:
                    if layer_path.endswith(ext):
                        layer_path = layer_path[:-len(ext)]
                        break
                
                if layer_path not in extracted: extracted[layer_path] = [None, None]
                idx = 0 if suffix == '_lora_a' else 1
                extracted[layer_path][idx] = value
    
    final_lora_layers = {}
    for k, v in extracted.items():
        if v[0] is None or v[1] is None: continue
        
        lora_a, lora_b = v
        # Special case: KV Einsum needs to be split for HF compatibility
        if k.endswith('.attn.kv_einsum'):
            base_k = k.replace('.attn.kv_einsum', '.attn.k_proj')
            base_v = k.replace('.attn.kv_einsum', '.attn.v_proj')
            
            # lora_b for kv_einsum is usually [Rank, 2, Heads, Dim]
            # We must access the underlying array
            b_val = lora_b.value if hasattr(lora_b, 'value') else lora_b
            
            final_lora_layers[base_k] = (lora_a, b_val[:, 0, ...])
            final_lora_layers[base_v] = (lora_a, b_val[:, 1, ...])
        else:
            final_lora_layers[k] = (lora_a, lora_b)
            
    return final_lora_layers

def state_key_transform_fn(nnx_key: str) -> str:
    """Transforms FULL NNX layer paths to HF safetensors keys."""
    prefix = "model.language_model."
    
    if nnx_key == "embedder.input_embedding":
        return prefix + "embed_tokens.weight"
    if nnx_key == "embedder.per_layer_input_embedding":
        return prefix + "embed_tokens_per_layer.weight"
    if nnx_key == "embedder.per_layer_model_projection":
        return prefix + "per_layer_model_projection.weight"
    if nnx_key == "embedder.per_layer_projection_norm":
        return prefix + "per_layer_projection_norm.weight"
        
    if "mlp.gate_proj" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.") + ".weight"
    if "mlp.up_proj" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.") + ".weight"
    if "mlp.down_proj" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.") + ".weight"
    
    if "attn.q_einsum" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.").replace(".attn.q_einsum", ".self_attn.q_proj.weight")
    if "attn.k_proj" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.").replace(".attn.k_proj", ".self_attn.k_proj.weight")
    if "attn.v_proj" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.").replace(".attn.v_proj", ".self_attn.v_proj.weight")
    if "attn.attn_vec_einsum" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.").replace(".attn.attn_vec_einsum", ".self_attn.o_proj.weight")
        
    if "per_layer" in nnx_key:
        return nnx_key.replace("layers.", prefix + "layers.") + ".weight"

    if nnx_key.startswith("layers."):
        return nnx_key.replace("layers.", prefix + "layers.") + ".weight"
        
    return nnx_key

def main():
    print("Exporting model on CPU...")
    # 1. Initialize model structure
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    rngs = nnx.Rngs(0)
    model = gemma4_model.Gemma4(config, rngs=rngs)
    
    # 2. Apply LoRA
    print("Applying LoRA...")
    provider = LoraProvider(rank=LORA_RANK, alpha=LORA_ALPHA)
    dummy_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_mask = make_causal_attn_mask(jnp.ones((1, 1), dtype=jnp.bool_))
    model = apply_lora_to_model(model, provider, dummy_tokens, attention_mask=dummy_mask, rngs=rngs)
    
    # 3. Restore trained LoRA weights
    print(f"Restoring LoRA weights from {LORA_CHECKPOINT_DIR}...")
    cm = CheckpointManager(root_directory=os.path.abspath(LORA_CHECKPOINT_DIR))
    cm.maybe_restore(model, restore_only_lora_params=True)
    
    # 4. Merge and Save
    print("Merging LoRA weights and saving to safetensors...")
    transpose_rules = {
        "": (1, 0), # Universal rule for 2D kernels
    }

    def extractor_factory(ignored):
        return custom_layer_extractor(model)

    save_lora_merged_model_as_safetensors(
        local_model_path=BASE_MODEL_DIR,
        output_dir=MERGED_OUTPUT_DIR,
        lora_model=model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        state_key_transform_fn=state_key_transform_fn,
        custom_layer_extractor_fn=extractor_factory,
        transpose_rules=transpose_rules
    )
    print(f"Export complete. Merged model is in {MERGED_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
