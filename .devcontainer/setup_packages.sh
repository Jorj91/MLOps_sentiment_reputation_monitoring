
set -e

# Minimal packages first
pip install --no-cache-dir numpy pydantic fastapi uvicorn pytest


# Heavy packages installed after container is ready

# CPU version of torch
pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
# Transformers and datasets
pip install --no-cache-dir transformers datasets


echo "All packages installed successfully!"