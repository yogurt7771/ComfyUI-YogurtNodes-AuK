import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.quant_ops import QUANT_ALGOS

from .auk_model import AuKModel
from .auk_vae import BigVGANFlowVAE, fold_weight_norm
from .encoder import AuKEncoder, Processor, make_encoder
from .model_paths import ENCODER_SHARDS


FLASH_TIMES = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
FRAMES_PER_SECOND = 50
ENCODER_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "qwen2.5-omni-3b"


def make_patcher(model, load_device, offload_device):
    mm.archive_model_dtypes(model)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)


def metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def compute_dtype(choice, device):
    if choice == "auto":
        return torch.bfloat16 if mm.supports_dtype(device, torch.bfloat16) and device.type != "cpu" else torch.float32
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[choice]
    if not mm.supports_dtype(device, dtype):
        raise ValueError(f"{device} does not support {choice}; select auto or fp32.")
    return dtype


def operations(state, dtype):
    for key, value in state.items():
        if key.endswith(".comfy_quant"):
            kind = json.loads(value.numpy().tobytes())["format"]
            if kind not in QUANT_ALGOS:
                raise ValueError(f"This ComfyUI does not support {kind}. Update ComfyUI/Comfy Kitchen or use the BF16 checkpoint.")
    return comfy.ops.mixed_precision_ops(compute_dtype=dtype)


def model_variant(path, info):
    """Use checkpoint metadata, or the original release's unambiguous filename."""
    variant = info.get("auk_variant")
    if variant is None:
        variant = {"auk_base.safetensors": "base", "auk_flash.safetensors": "flash"}.get(Path(path).name.lower())
    if variant is None:
        raise ValueError("Cannot identify this AuK checkpoint. Keep the original auk_base.safetensors/auk_flash.safetensors filename or use a converted checkpoint with variant metadata.")
    if variant not in ("base", "flash"):
        raise ValueError(f"Unsupported AuK variant: {variant}")
    return variant


def load_model(path, precision):
    variant = model_variant(path, metadata(path))
    state = load_file(str(path))
    device = mm.get_torch_device()
    dtype = compute_dtype(precision, device)
    model = AuKModel(operations(state, dtype), variant)
    model.load_state_dict(state, strict=True)
    model.eval()
    patcher = make_patcher(model, device, mm.unet_offload_device())
    patcher.auk_dtype = dtype
    return patcher


def load_encoder(path, precision):
    device = mm.text_encoder_device()
    dtype = compute_dtype(precision, device)
    if Path(path).is_dir():
        state = {}
        for shard in ENCODER_SHARDS:
            with safe_open(str(Path(path) / shard), framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    if key.startswith(("thinker.model.", "thinker.audio_tower.", "thinker.lm_head.")):
                        state[key.removeprefix("thinker.")] = checkpoint.get_tensor(key)
    else:
        state = load_file(str(path))
        if "thinker.model.embed_tokens.weight" in state:
            state = {key.removeprefix("thinker."): value for key, value in state.items()
                     if key.startswith(("thinker.model.", "thinker.audio_tower.", "thinker.lm_head."))}
    has_lm_head = "lm_head.weight" in state
    model = make_encoder(ENCODER_ASSETS, operations(state, dtype), dtype, has_lm_head)
    model.load_state_dict(state, strict=True)
    managed = torch.nn.Module()
    managed.add_module("encoder", model)
    patcher = make_patcher(managed, device, mm.text_encoder_offload_device())
    return AuKEncoder(patcher, Processor(ENCODER_ASSETS), dtype, has_lm_head)


class AuKVAE:
    audio_sample_rate = 24000
    audio_sample_rate_output = 24000
    latent_channels = 64

    def __init__(self, path):
        state = fold_weight_norm(load_file(str(path)))
        model = BigVGANFlowVAE()
        model.load_state_dict(state, strict=True)
        model.eval()
        self.patcher = make_patcher(model, mm.vae_device(), mm.vae_offload_device())

    def encode_reference(self, waveform, seed):
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        waveform = waveform.to(device=device, dtype=torch.float32)
        # The strided encoder needs at least one codec frame.
        if waveform.shape[-1] < 480:
            waveform = F.pad(waveform, (0, 480 - waveform.shape[-1]))
        generator = torch.Generator(device=device).manual_seed(seed)
        return self.patcher.model.encode(waveform, generator).cpu()

    def encode(self, waveform):
        return self.encode_reference(waveform.movedim(-1, 1).mean(1, keepdim=True), 0).transpose(1, 2)

    def decode(self, samples):
        return self.decode_target(samples.transpose(1, 2)).movedim(1, -1)

    def decode_target(self, latent):
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        return self.patcher.model.decode(latent.to(device=device, dtype=torch.float32)).cpu()


def encode_instruction(encoder, model, instruction, audio=None):
    weights = model.model.layer_weights.detach().cpu()
    scale = model.model.layer_scale.detach().cpu()
    count = 1 if audio is None else audio["waveform"].shape[0]
    result = []
    for index in range(count):
        mm.throw_exception_if_processing_interrupted()
        source = None if audio is None else {"waveform": audio["waveform"][index:index + 1].clone(), "sample_rate": audio["sample_rate"]}
        hidden, mask = encoder.encode(instruction, source, weights, scale)
        result.append([hidden, {"auk_attention_mask": mask, "auk_audio": source}])
    return result


def time_grid(is_flash, steps, sway):
    if is_flash:
        return FLASH_TIMES
    return [t + sway * (math.cos(math.pi * t / 2) - 1 + t) for t in (i / steps for i in range(steps + 1))]


def generate(model, vae, conditioning, seconds, seed, steps, guidance, sway):
    if not isinstance(vae, AuKVAE):
        raise ValueError("AuK Generate / Edit requires the VAE from AuK VAE Loader.")
    outputs = []
    for index, (hidden, options) in enumerate(conditioning):
        if "auk_attention_mask" not in options:
            raise ValueError("Use conditioning from AuK Instruction Encode.")
        source = options["auk_audio"]
        item_seed = (seed + index) % (2**64)
        if source is None:
            if seconds <= 0:
                raise ValueError("Set a positive generation duration for text-only speech.")
            ref = torch.empty(1, 0, 64)
        else:
            waveform = source["waveform"].mean(1, keepdim=True)
            waveform = torchaudio.functional.resample(waveform, source["sample_rate"], 24000)
            ref = vae.encode_reference(waveform, item_seed)
        target_seconds = seconds if seconds > 0 else source["waveform"].shape[-1] / source["sample_rate"]
        length = max(1, math.ceil(target_seconds * FRAMES_PER_SECOND))
        mm.throw_exception_if_processing_interrupted()
        mm.load_models_gpu([model])
        device, dtype = model.load_device, model.auk_dtype
        ref = ref.to(device=device, dtype=dtype)
        hidden = hidden.to(device=device, dtype=dtype)
        mask = options["auk_attention_mask"].to(device)
        backbone = model.model.transformer
        guided = not model.model.is_flash and guidance >= 1e-5
        context, prompt = backbone.prepare(hidden, ref, guided)
        times = time_grid(model.model.is_flash, steps, sway)
        rng = torch.Generator(device=device).manual_seed(item_seed)
        x = torch.randn((1, length, 64), device=device, dtype=torch.float32, generator=rng)
        progress = comfy.utils.ProgressBar(len(times) - 1)
        pairs = list(zip(times[:-1], times[1:]))
        if tqdm is not None:
            label = f"AuK {'flash' if model.model.is_flash else 'base'}"
            if len(conditioning) > 1:
                label += f" {index + 1}/{len(conditioning)}"
            pairs = tqdm(pairs, desc=label, unit="step", leave=True)
        for start, end in pairs:
            mm.throw_exception_if_processing_interrupted()
            velocity = backbone(x.to(dtype), torch.tensor(start, device=device, dtype=torch.float32), context, prompt, mask, guided)
            if guided:
                conditional, unconditional = velocity.chunk(2)
                velocity = conditional + guidance * (conditional - unconditional)
            x = x + (end - start) * velocity.float()
            progress.update(1)
        del context, prompt, ref, hidden, velocity
        audio = vae.decode_target(x.cpu())
        if not torch.isfinite(audio).all():
            raise RuntimeError("AuK generated non-finite audio; try the BF16 model with fp32 compute.")
        outputs.append(audio[..., :round(target_seconds * 24000)])
    return {"waveform": torch.cat(outputs), "sample_rate": 24000}
