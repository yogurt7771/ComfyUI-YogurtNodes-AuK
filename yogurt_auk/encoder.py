"""Qwen2.5-Omni text/audio inference using ComfyUI operations and tokenizers."""
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from tokenizers import Tokenizer

import comfy.model_management as mm
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen25_3BConfig


def audio_features(waveform, sample_rate):
    waveform = waveform.detach().float().mean(1)
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    # Two convolution frames are needed by the audio tower's final average pool.
    samples = max(waveform.shape[-1], 480)
    frames = (samples + 159) // 160
    waveform = F.pad(waveform, (0, frames * 160 + 400 - waveform.shape[-1]))
    window = torch.hann_window(400, device=waveform.device)
    spectrum = torch.stft(waveform, 400, 160, window=window, return_complex=True)
    filters = torchaudio.functional.melscale_fbanks(201, 0, 8000, 128, 16000, norm="slaney", mel_scale="slaney").to(waveform.device)
    mel = filters.T @ spectrum[..., :-1].abs().square()
    log_mel = mel.clamp_min(1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.amax(dim=(-2, -1), keepdim=True) - 8)
    return ((log_mel + 4) / 4)[0, :, :frames].contiguous()


class Processor:
    def __init__(self, assets):
        self.tokenizer = Tokenizer.from_file(str(Path(assets) / "tokenizer.json"))
        self.audio_token_id = self.tokenizer.token_to_id("<|AUDIO|>")
        self.stop_ids = {self.tokenizer.token_to_id(token) for token in ("<|im_end|>", "<|endoftext|>")}

    def chat(self, system, content):
        text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False).ids], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def instruction(self, text, audio=None):
        features = None
        if audio is None:
            text += "|<no_prompt_audio>|"
        else:
            features = audio_features(audio["waveform"], audio["sample_rate"])
            tokens = ((features.shape[-1] + 1) // 2) // 2
            text += "<|audio_bos|>" + "<|AUDIO|>" * tokens + "<|audio_eos|>"
        inputs = self.chat("You are a helpful assistant.", text)
        if features is not None:
            inputs["input_features"] = features
        return inputs


class AudioAttention(nn.Module):
    def __init__(self, dim, heads, ops, dtype):
        super().__init__()
        self.heads = heads
        self.q_proj = ops.Linear(dim, dim, dtype=dtype)
        self.k_proj = ops.Linear(dim, dim, bias=False, dtype=dtype)
        self.v_proj = ops.Linear(dim, dim, dtype=dtype)
        self.out_proj = ops.Linear(dim, dim, dtype=dtype)

    def forward(self, x, mask):
        q, k, v = (projection(x).reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
                   for projection in (self.q_proj, self.k_proj, self.v_proj))
        attention = optimized_attention_for_device(x.device, mask=mask is not None)
        return self.out_proj(attention(q, k, v, self.heads, mask=mask, skip_reshape=True))


class AudioLayer(nn.Module):
    def __init__(self, config, ops, dtype):
        super().__init__()
        dim = config["d_model"]
        self.self_attn = AudioAttention(dim, config["encoder_attention_heads"], ops, dtype)
        self.self_attn_layer_norm = comfy.ops.manual_cast.LayerNorm(dim, dtype=dtype)
        self.fc1 = ops.Linear(dim, config["encoder_ffn_dim"], dtype=dtype)
        self.fc2 = ops.Linear(config["encoder_ffn_dim"], dim, dtype=dtype)
        self.final_layer_norm = comfy.ops.manual_cast.LayerNorm(dim, dtype=dtype)

    def forward(self, x, mask):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class QwenAudioEncoder(nn.Module):
    def __init__(self, config, ops, dtype):
        super().__init__()
        self.n_window = config["n_window"]
        dim = config["d_model"]
        self.conv1 = comfy.ops.manual_cast.Conv1d(config["num_mel_bins"], dim, 3, padding=1, dtype=dtype)
        self.conv2 = comfy.ops.manual_cast.Conv1d(dim, dim, 3, stride=2, padding=1, dtype=dtype)
        self.audio_bos_eos_token = comfy.ops.manual_cast.Embedding(2, config["output_dim"], dtype=dtype)
        self.layers = nn.ModuleList([AudioLayer(config, ops, dtype) for _ in range(config["encoder_layers"])])
        self.ln_post = comfy.ops.manual_cast.LayerNorm(dim, dtype=dtype)
        self.proj = ops.Linear(dim, config["output_dim"], dtype=dtype)

    def forward(self, features):
        chunks = features.split(self.n_window * 2, dim=-1)
        lengths = [chunk.shape[-1] for chunk in chunks]
        width = max(lengths)
        padded = torch.stack([F.pad(chunk, (0, width - length)) for chunk, length in zip(chunks, lengths)])
        valid = torch.zeros((len(chunks), 1, width), device=features.device, dtype=features.dtype)
        for index, length in enumerate(lengths):
            valid[index, :, :length] = 1
        x = F.gelu(self.conv1(padded)) * valid
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        dim = x.shape[-1]
        frequency = torch.exp(torch.arange(dim // 2, device=x.device, dtype=torch.float32) * (-math.log(10000) / (dim // 2 - 1)))
        angle = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * frequency
        x = x + torch.cat((angle.sin(), angle.cos()), -1).to(x.dtype)
        after_cnn = [(length + 1) // 2 for length in lengths]
        mask = torch.zeros((len(chunks), 1, 1, x.shape[1]), dtype=x.dtype, device=x.device)
        for index, length in enumerate(after_cnn):
            mask[index, :, :, length:] = torch.finfo(x.dtype).min
        # Each two-second window is independent; batch them instead of a quadratic block mask.
        for layer in self.layers:
            x = layer(x, mask)
        x = torch.cat([part[:length] for part, length in zip(x, after_cnn)], 0)
        x = F.avg_pool1d(x.T[None], 2, stride=2)[0].T
        return self.proj(self.ln_post(x))


class QwenEncoder(nn.Module):
    def __init__(self, config, ops, dtype, has_lm_head=True):
        super().__init__()
        text = config["text_config"]
        llm_config = Qwen25_3BConfig(**{key: text[key] for key in Qwen25_3BConfig.__dataclass_fields__ if key in text})
        llm_config.head_dim = text["hidden_size"] // text["num_attention_heads"]
        self.model = Llama2_(llm_config, device="cpu", dtype=dtype, ops=ops)
        self.audio_tower = QwenAudioEncoder(config["audio_config"], ops, dtype)
        self.audio_token_id = config["audio_token_index"]
        self.dtype = dtype
        if has_lm_head:
            self.lm_head = ops.Linear(text["hidden_size"], text["vocab_size"], bias=False, dtype=dtype)

    def encode_layers(self, inputs):
        ids = inputs["input_ids"]
        embeds = self.model.embed_tokens(ids, out_dtype=self.dtype)
        if "input_features" in inputs:
            features = self.audio_tower(inputs["input_features"])
            embeds = embeds.masked_scatter((ids == self.audio_token_id)[..., None], features.to(embeds))
        # Audio-only MRoPE has identical sequential positions on all three axes.
        _, hidden = self.model(None, embeds=embeds, attention_mask=inputs["attention_mask"],
                               intermediate_output=list(range(1, len(self.model.layers) + 1)),
                               final_layer_norm_intermediate=False)
        return hidden.unbind(1)

    def complete(self, inputs, stop_ids, max_new_tokens=256):
        token = inputs["input_ids"]
        cache = self.model.init_kv_cache(token.shape[0], token.shape[1] + max_new_tokens, token.device, self.dtype)
        result = []
        for _ in range(max_new_tokens):
            mm.throw_exception_if_processing_interrupted()
            hidden, _, cache = self.model(token, dtype=self.dtype, past_key_values=cache)
            token = self.lm_head(hidden[:, -1:]).argmax(-1)
            value = token.item()
            if value in stop_ids:
                break
            result.append(value)
        return result


def make_encoder(config_path, ops, dtype, has_lm_head=True):
    config = json.loads((Path(config_path) / "config.json").read_text(encoding="utf-8"))
    return QwenEncoder(config, ops, dtype, has_lm_head).eval()


class AuKEncoder:
    def __init__(self, patcher, processor, dtype, has_lm_head=True):
        self.patcher = patcher
        self.processor = processor
        self.dtype = dtype
        self.has_lm_head = has_lm_head

    def encode(self, instruction, audio, layer_weights, layer_scale):
        inputs = self.processor.instruction(instruction, audio)
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        inputs = {key: value.to(device=device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        states = self.patcher.model.encoder.encode_layers(inputs)
        if len(states) != layer_weights.numel():
            raise ValueError("AuK layer-fusion weights do not match the Qwen encoder depth.")
        weights = layer_weights.to(device=device, dtype=torch.float32).softmax(0)
        fused = sum(F.layer_norm(state, (state.shape[-1],)).float() * weight for weight, state in zip(weights, states))
        fused = fused * layer_scale.to(device=device, dtype=torch.float32)
        return fused.cpu(), inputs["attention_mask"].bool().cpu()

    def complete(self, system, content, max_new_tokens=256):
        if not self.has_lm_head:
            raise ValueError("Prompt Enhance requires an encoder checkpoint containing lm_head.weight.")
        inputs = self.processor.chat(system, content)
        mm.load_models_gpu([self.patcher])
        inputs = {key: value.to(self.patcher.load_device) for key, value in inputs.items()}
        ids = self.patcher.model.encoder.complete(inputs, self.processor.stop_ids, max_new_tokens)
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
