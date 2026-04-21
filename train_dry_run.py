import os
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.sft import peft_trainer
from tunix.sft.utils import make_causal_attn_mask
from qwix import LoraProvider, apply_lora_to_model

# Force CPU for dry-run
os.environ["JAX_PLATFORMS"] = "cpu"

def dry_run():
    print("Setting up config...")
    # 1. Config (scaled down for CPU)
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(
        config,
        num_layers=1,
        num_embed=1000,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=1,
        per_layer_input_dim=16,
        override_kv_shared_ffw_hidden=256,
        use_flash_attention=False
    )

    print("Initializing model...")
    # 2. Model Initialization
    rngs = nnx.Rngs(0)
    model = gemma4_model.Gemma4(config, rngs=rngs)
    
    print("Applying LoRA...")
    # 3. Apply LoRA
    provider = LoraProvider(rank=4, alpha=8.0)
    dummy_tokens = jnp.zeros((1, 8), dtype=jnp.int32)
    dummy_mask = make_causal_attn_mask(jnp.ones((1, 8), dtype=jnp.bool_))
    model = apply_lora_to_model(model, provider, dummy_tokens, attention_mask=dummy_mask, rngs=rngs)

    print("Setting up trainer...")
    # 4. Trainer Setup
    training_config = peft_trainer.TrainingConfig(
        max_steps=2,
        eval_every_n_steps=10,
        gradient_accumulation_steps=1,
        checkpoint_root_directory=os.path.abspath("./checkpoints_dry_run")
    )

    optimizer = optax.adamw(learning_rate=1e-4)
    
    trainer = peft_trainer.PeftTrainer(
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )

    # 5. Mock Dataset
    def train_ds_gen():
        batch_size = 1
        seq_len = 8
        while True:
            yield {
                "input_tokens": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
                "input_mask": jnp.ones((batch_size, seq_len), dtype=jnp.bool_),
                "positions": jnp.tile(jnp.arange(seq_len), (batch_size, 1)),
                "attention_mask": make_causal_attn_mask(jnp.ones((batch_size, seq_len), dtype=jnp.bool_)),
            }

    print("Starting dry-run training...")
    trainer.train(train_ds_gen())
    print("Dry-run completed successfully.")

if __name__ == "__main__":
    dry_run()
