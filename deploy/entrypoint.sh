#!/bin/bash

set -e  # Exit immediately if any command fails

# Check if Hugging Face token is set
if [[ -n "${HF_TOKEN}" ]]; then
    echo "Logging into Hugging Face using HF_TOKEN..." >&2
    python3 -c "import huggingface_hub; import os; huggingface_hub.login(os.getenv('HF_TOKEN'))" || {
        echo "Failed to log in to Hugging Face. Ensure huggingface_hub is installed." >&2
        exit 1
    }
else
    echo "HF_TOKEN is not set. Skipping Hugging Face login." >&2
fi

# Execute the main vLLM API server
exec python3 -u -m vllm.entrypoints.openai.api_server "$@"
