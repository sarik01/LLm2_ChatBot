from huggingface_hub import snapshot_download
model_id="ai-forever/mGPT"
snapshot_download(repo_id=model_id, local_dir="mGPT",
                  local_dir_use_symlinks=False, revision="main")