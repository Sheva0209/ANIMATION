"""
download_model.py
Download model ResembleAI/chatterbox-turbo via hf-mirror.com
Run: D:\animasi\TTS\chatterbox-env\Scripts\python.exe D:\animasi\download_model.py
"""
import os
import sys

# ── Set env vars BERSIH sebelum import apapun ─────────────────────────────────
HF_MIRROR  = "https://hf-mirror.com"
HF_CACHE   = r"D:\hf_cache"

os.environ["HF_ENDPOINT"]            = HF_MIRROR
os.environ["HF_HOME"]                = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"]  = os.path.join(HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"]     = os.path.join(HF_CACHE, "transformers")

print(f"HF_ENDPOINT : {repr(os.environ['HF_ENDPOINT'])}")
print(f"HF_HOME     : {os.environ['HF_HOME']}")

# ── Import SETELAH env di-set ──────────────────────────────────────────────────
from huggingface_hub import snapshot_download, constants

print(f"HF Hub endpoint (live): {repr(constants.ENDPOINT)}")

REPO_ID = "ResembleAI/chatterbox-turbo"
PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]

print(f"\nDownloading {REPO_ID} ...")
print(f"  dari : {HF_MIRROR}")
print(f"  ke   : {HF_CACHE}\\hub\n")

try:
    local_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=PATTERNS,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    print(f"\n✅ Model berhasil didownload ke:\n   {local_path}")
except Exception as e:
    print(f"\n❌ Download gagal: {e}")
    sys.exit(1)
