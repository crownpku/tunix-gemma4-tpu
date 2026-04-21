from huggingface_hub import snapshot_download
import os

def download_gemma():
    model_id = "google/gemma-4-E2B-it"
    local_dir = "models/gemma-4-E2B-it"
    
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main"
    )
    print("Download complete.")

if __name__ == "__main__":
    download_gemma()
