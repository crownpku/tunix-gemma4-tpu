# Fine-tuning Gemma 4 E2B-IT with Tunix on TPU

This repository contains the complete workflow for fine-tuning the **Gemma 4 E2B-IT (5.1B)** model on the `insuranceQA-v2` dataset using **Tunix (JAX)** on Google Cloud TPUs.

## Project Summary

The goal of this project was to specialize a state-of-the-art Gemma 4 model for the insurance domain. We leveraged the high-performance JAX ecosystem via Tunix to perform Supervised Fine-Tuning (SFT) with LoRA.

### Key Achievements
- **Successful SFT:** Completed 500 training steps on a TPU v5litepod-4, reducing training loss from ~6.99 to a stable convergence point.
- **PLE-Aware Weight Merging:** Developed a custom export script to handle Gemma 4's 4.7GB Per-Layer Embedding (PLE) table. Merging was performed on the TPU VM's CPU (188GB RAM) to bypass the 16GB HBM limit of TPU cores.
- **Tunix Patching:** Identified and patched a critical bug in Tunix's Gemma 4 attention mechanism. The fix allows for non-flash inference when sequence lengths are not divisible by 1024, resolving broadcasting issues in the attention mask.
- **Verification:** Validated the final merged model via perplexity analysis and weight statistics (std ~0.03), ensuring zero degradation during the LoRA merging process.

## Prerequisites

- **Google Cloud Platform:** Access to TPU v5litepod-4 (or similar).
- **Environment:** `uv` for package management.
- **Authentication:** Gcloud CLI configured with appropriate permissions.

## Reproduction Instructions

### 1. Local Environment Setup
```bash
uv sync
```

### 2. Data Preparation
Download and format the `insuranceQA-v2` dataset:
```bash
uv run prepare_data.py
```

### 3. TPU Provisioning & Bootstrap
Provision a TPU v5litepod-4 and upload the project code. Use the provided shell script for bootstrapping the remote environment:
```bash
bash tpu_setup_and_train.sh
```

### 4. Training (Remote)
Run the training script on the TPU VM:
```bash
JAX_PLATFORMS=tpu,cpu uv run train_tpu.py
```

### 5. Weight Merging (CPU-Bound)
**Critical:** Due to the large PLE table, weight merging MUST be done on a machine with high RAM (e.g., the TPU VM's CPU) using the CPU platform to avoid HBM exhaustion.
```bash
JAX_PLATFORMS=cpu uv run export_model.py
```

### 6. Evaluation & Patching
If running inference on CPU or with non-standard sequence lengths, ensure the `tunix/models/gemma4/model.py` patch is applied to handle the 4D attention mask broadcasting:
```python
# In tunix/models/gemma4/model.py
expanded_mask = jnp.reshape(attn_mask, (attn_mask.shape[0], -1, 1, attn_mask.shape[-1]))
```
Run quantitative evaluation:
```bash
uv run eval_perplexity.py
```

## Repository Structure
- `train_tpu.py`: Main SFT script for TPU.
- `export_model.py`: Robust LoRA merging script (PLE-aware).
- `prepare_data.py`: Dataset processing for Tunix ingestion.
- `eval_perplexity.py`: Quantitative integrity verification.
- `manual_eval_no_jit.py`: Sequential inference script for debugging.

## License
MIT
