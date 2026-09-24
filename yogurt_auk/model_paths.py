import os
from pathlib import Path
import tempfile
import urllib.request

from safetensors import safe_open

import folder_paths
import comfy.model_management as mm
import comfy.utils


OFFICIAL_ENCODER = "Qwen2.5-Omni-3B"
ENCODER_SHARDS = ("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors")
SOURCES = {
    ("diffusion_models", "auk_base.safetensors"): ("tencent/AuK", "auk_base.safetensors"),
    ("diffusion_models", "auk_flash.safetensors"): ("tencent/AuK-Flash", "auk_flash.safetensors"),
    ("vae", "AuK/vae.safetensors"): ("tencent/AuK", "vae.safetensors"),
}
for variant in ("base", "flash"):
    for precision in ("bf16", "fp32", "int8", "w4a8"):
        name = f"auk_{variant}_{precision}.safetensors"
        SOURCES["diffusion_models", name] = ("drbaph/AuK-comfyui", f"diffusion_models/{name}")
for precision in ("bf16", "int8", "w4a8"):
    name = f"qwen_omni_{precision}.safetensors"
    SOURCES["text_encoders", name] = ("drbaph/AuK-comfyui", f"text_encoders/{name}")
SOURCES["vae", "auk_vae.safetensors"] = ("drbaph/AuK-comfyui", "vae/auk_vae.safetensors")


def model_choices(category):
    known = [name for kind, name in SOURCES if kind == category]
    if category == "text_encoders":
        known.insert(0, OFFICIAL_ENCODER)
    available = folder_paths.get_filename_list(category)
    if category == "text_encoders":
        available = [name for name in available if not name.replace("\\", "/").startswith(f"{OFFICIAL_ENCODER}/")]
    return list(dict.fromkeys(known + available))


def download(target, repository, filename):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    try:
        url = f"https://huggingface.co/{repository}/resolve/main/{filename}"
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
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


def model_path(category, name, download_if_missing=False):
    if category == "text_encoders" and name == OFFICIAL_ENCODER:
        for root in folder_paths.get_folder_paths(category):
            directory = Path(root) / OFFICIAL_ENCODER
            if all((directory / shard).is_file() for shard in ENCODER_SHARDS):
                return str(directory)
        if not download_if_missing:
            raise FileNotFoundError(f"Official Qwen encoder is missing. Put both model shards in models/{category}/{OFFICIAL_ENCODER} or enable download_if_missing.")
        directory = Path(folder_paths.models_dir) / category / OFFICIAL_ENCODER
        for shard in ENCODER_SHARDS:
            if not (directory / shard).is_file():
                download(directory / shard, "Qwen/Qwen2.5-Omni-3B", shard)
        return str(directory)

    existing = folder_paths.get_full_path(category, name)
    if existing is not None:
        return existing
    if not download_if_missing:
        raise FileNotFoundError(f"AuK checkpoint not found: {name}. Put it in models/{category} or enable download_if_missing.")
    source = SOURCES.get((category, name))
    if source is None:
        raise ValueError(f"No download is configured for {name}; install this local checkpoint yourself.")
    target = Path(folder_paths.models_dir) / category / name
    download(target, *source)
    return str(target)
