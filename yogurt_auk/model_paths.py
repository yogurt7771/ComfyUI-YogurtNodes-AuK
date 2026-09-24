import os
from pathlib import Path
import tempfile
import urllib.request

from safetensors import safe_open

import folder_paths
import comfy.model_management as mm
import comfy.utils


REPOSITORY = "https://huggingface.co/drbaph/AuK-comfyui/resolve/main"
MODELS = {
    "diffusion_models": [f"auk_{variant}_{precision}.safetensors" for variant in ("base", "flash") for precision in ("bf16", "fp32", "int8", "w4a8")],
    "text_encoders": [f"qwen_omni_{precision}.safetensors" for precision in ("bf16", "int8", "w4a8")],
    "vae": ["auk_vae.safetensors"],
}


def model_choices(category):
    return list(dict.fromkeys(MODELS[category] + folder_paths.get_filename_list(category)))


def model_path(category, name, download_if_missing=False):
    existing = folder_paths.get_full_path(category, name)
    if existing is not None:
        return existing
    if not download_if_missing:
        raise FileNotFoundError(f"AuK checkpoint not found: {name}. Put it in models/{category} or enable download_if_missing.")
    if name not in MODELS[category]:
        raise ValueError(f"No download is configured for {name}; install this local checkpoint yourself.")
    target = Path(folder_paths.models_dir) / category / name
    target.parent.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted transfer must never become a selectable checkpoint.
    descriptor, partial = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(f"{REPOSITORY}/{category}/{name}", timeout=60) as response:
            expected = int(response.headers.get("Content-Length", 0))
            progress = comfy.utils.ProgressBar(max(expected, 1))
            received = 0
            while chunk := response.read(8 * 1024 * 1024):
                mm.throw_exception_if_processing_interrupted()
                output.write(chunk)
                received += len(chunk)
                progress.update_absolute(received, max(expected, received))
            if expected and received != expected:
                raise IOError(f"Incomplete AuK download: {received} of {expected} bytes.")
        with safe_open(partial, framework="pt", device="cpu") as checkpoint:
            if not list(checkpoint.keys()):
                raise ValueError("Downloaded AuK checkpoint contains no tensors.")
        if not target.exists():
            os.replace(partial, target)
    finally:
        Path(partial).unlink(missing_ok=True)
    return str(target)
