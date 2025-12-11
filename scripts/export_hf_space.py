import os
import shutil

MAIN_SRC = "src"
HF_DEPLOY_DIR = "hf_deploy"

# 1. Copy necessary files
shutil.copy(os.path.join(MAIN_SRC, "app.py"), HF_DEPLOY_DIR)
shutil.copy(os.path.join(MAIN_SRC, "sentiment.py"), HF_DEPLOY_DIR)

# 2. Copy tiny model
MODEL_SRC = "/tmp/local_model"  # path used in CI/CD
MODEL_DST = os.path.join(HF_DEPLOY_DIR, "model")
if os.path.exists(MODEL_DST):
    shutil.rmtree(MODEL_DST)
shutil.copytree(MODEL_SRC, MODEL_DST)

# 3. Copy requirements
shutil.copy("requirements.txt", HF_DEPLOY_DIR)

print("HF deployment folder is ready!")