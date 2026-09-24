from comfy_api.latest import ComfyExtension, io, ui

from . import prompt_enhance, runtime
from .model_paths import model_choices, model_path


AUK_MODEL = io.Custom("YOGURT_AUK_MODEL")
AUK_ENCODER = io.Custom("YOGURT_AUK_ENCODER")


class AuKModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKModelLoader",
            display_name="Yogurt AuK Model Loader",
            category="YogurtNodes/AuK",
            inputs=[
                io.Combo.Input("model_name", options=model_choices("diffusion_models"), tooltip="Converted AuK checkpoint. Base = full quality, Flash = distilled 4-step. The file suffix sets the weight format: bf16 (baseline), int8 or w4a8 (smaller, less VRAM)."),
                io.Combo.Input("precision", options=["auto", "bf16", "fp32"], tooltip="Compute precision. auto = bf16 on GPU, fp32 on CPU. The weight format (bf16/int8/w4a8) is read from the checkpoint itself, not from this setting."),
                io.Boolean.Input("download_if_missing", default=False, optional=True, tooltip="Download only the selected known checkpoint if missing. Existing local files are reused."),
            ],
            outputs=[AUK_MODEL.Output()],
        )

    @classmethod
    def execute(cls, model_name, precision, download_if_missing=False):
        return io.NodeOutput(runtime.load_model(model_path("diffusion_models", model_name, download_if_missing), precision))


class AuKEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKEncoderLoader",
            display_name="Yogurt AuK Encoder Loader",
            category="YogurtNodes/AuK",
            inputs=[
                io.Combo.Input("encoder_name", options=model_choices("text_encoders"), tooltip="Converted Qwen2.5-Omni text/audio encoder. Tokenizer and architecture are bundled in this plugin. One encoder can feed any number of AuK models."),
                io.Combo.Input("precision", options=["auto", "bf16", "fp32"], tooltip="Compute precision for the encoder. auto = bf16 on GPU, fp32 on CPU. Weight format (bf16/int8/w4a8) comes from the checkpoint."),
                io.Boolean.Input("download_if_missing", default=False, optional=True, tooltip="Download only the selected known checkpoint if missing. Existing local files are reused."),
            ],
            outputs=[AUK_ENCODER.Output()],
        )

    @classmethod
    def execute(cls, encoder_name, precision, download_if_missing=False):
        return io.NodeOutput(runtime.load_encoder(model_path("text_encoders", encoder_name, download_if_missing), precision))


class AuKVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKVAELoader",
            display_name="Yogurt AuK VAE Loader",
            category="YogurtNodes/AuK",
            inputs=[
                io.Combo.Input("vae_name", options=model_choices("vae"), tooltip="Original unquantized AuK VAE (auk_vae). Always runs in fp32; decoding through it preserves AuK's volume edits."),
                io.Boolean.Input("download_if_missing", default=False, optional=True, tooltip="Download only the selected known checkpoint if missing. Existing local files are reused."),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name, download_if_missing=False):
        return io.NodeOutput(runtime.AuKVAE(model_path("vae", vae_name, download_if_missing)))


class AuKInstructionEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKInstructionEncode",
            display_name="Yogurt AuK Instruction Encode",
            category="YogurtNodes/AuK",
            inputs=[
                AUK_MODEL.Input("model", tooltip="AuK model from AuK Model Loader. Base and Flash carry their own learned Qwen layer-fusion weights, so both must feed this node."),
                AUK_ENCODER.Input("encoder", tooltip="Encoder from AuK Encoder Loader."),
                io.Audio.Input("audio", optional=True, tooltip="Optional reference/source audio. Enables voice cloning and every editing task. Leave empty for text-only speech."),
                io.String.Input("instruction", multiline=True, default='Generate speech in a warm, clear voice. Say: "Hello, welcome to AuK."', tooltip="The task instruction. Examples: 'Say the following with the same voice: \"...\".' (clone), 'Replace 'old words' with 'new words'.', 'Raise the pitch by 2 semitones.', 'Extract only the singing voice and remove accompaniment.'"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, model, encoder, instruction, audio=None):
        return io.NodeOutput(runtime.encode_instruction(encoder, model, instruction, audio))


class AuKGenerateEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKGenerateEdit",
            display_name="Yogurt AuK Generate / Edit",
            category="YogurtNodes/AuK",
            inputs=[
                AUK_MODEL.Input("model", tooltip="AuK model from AuK Model Loader."),
                io.Conditioning.Input("conditioning", tooltip="Conditioning from AuK Instruction Encode."),
                io.Vae.Input("vae", tooltip="AuK VAE from AuK VAE Loader. Its own decode preserves AuK's volume edits."),
                io.Float.Input("seconds", default=3, min=0, max=3600, step=0.1, tooltip="Output length in seconds. 0 matches the source audio's duration. Text-only generation needs a positive value; roughly 2-3 spoken words per second."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Drives the diffusion noise and reference-latent sampling. Each batch item uses seed + index."),
                io.Int.Input("steps", default=32, min=1, max=1000, tooltip="Euler steps for Base; 32 is the upstream default and 16-50 is sensible. Flash ignores this and always runs its fixed 4-step schedule."),
                io.Float.Input("guidance", default=2, min=0, max=100, step=0.1, tooltip="AuK guidance strength for Base; 2 is the upstream default. This is not KSampler CFG. Flash ignores this."),
                io.Float.Input("sway", default=-1, min=-1, max=0, step=0.05, tooltip="Sway sampling for Base; -1 is the upstream default, 0 disables it. Flash ignores this."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, model, vae, conditioning, seconds, seed, steps, guidance, sway):
        return io.NodeOutput(runtime.generate(model, vae, conditioning, seconds, seed, steps, guidance, sway))


TASKS = {
    "Voice description TTS": ('Generate speech based on the following description: "{description}". The content to speak is: "{text}".', {"description": "Warm, clear voice", "text": "Hello, welcome to AuK."}),
    "Convert to whisper": ("用小声耳语的方式把这段话说出来。", {}),
    "Voice cloning": ('Say the following with the same voice: "{text}".', {"text": "Hello, welcome to AuK."}),
    "Replace speech": ("Replace '{original}' with '{replacement}'.", {"original": "old words", "replacement": "new words"}),
    "Insert speech before": ("Add '{text}' before '{anchor}'.", {"text": "new words", "anchor": "existing words"}),
    "Insert speech after": ("Add '{text}' after '{anchor}'.", {"text": "new words", "anchor": "existing words"}),
    "Remove speech": ("Remove '{text}'.", {"text": "words to remove"}),
    "Edit lyrics": ('Change "{original}" to "{replacement}" in the vocal recording.', {"original": "old lyrics", "replacement": "new lyrics"}),
    "Raise pitch": ("Raise the pitch by {semitones} semitones.", {"semitones": 2.0}),
    "Lower pitch": ("Lower the pitch by {semitones} semitones.", {"semitones": 2.0}),
    "Change speed": ("Adjust the speech speed to {factor}x.", {"factor": 1.25}),
    "Increase volume": ("Increase the volume by {decibels} dB.", {"decibels": 5.0}),
    "Decrease volume": ("Decrease the volume by {decibels} dB.", {"decibels": 5.0}),
    "Change emotion": ("Change the emotion to {description}.", {"description": "happy"}),
    "Change timbre": ('Keep the spoken content unchanged and change the timbre to: "{description}".', {"description": "a deep, calm male voice"}),
    "Remove accent": ("Remove the regional accent while preserving the speaker's voice and content.", {}),
    "Remove nonverbal sounds": ("Remove all {sound} from the audio.", {"sound": "breaths"}),
    "Add nonverbal sound": ("Add a {sound} at the {position} of the speech.", {"sound": "cough", "position": "beginning"}),
    "Whisper to speech": ("Convert this whispered speech into a normal speaking voice while preserving the speaker and content.", {}),
    "Enhance speech": ("Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.", {}),
    "Denoise only": ("Remove only the background noise, preserve everything else, and output audio of the same length.", {}),
    "Dereverberate only": ("Remove only the room reverberation, preserve everything else, and output audio of the same length.", {}),
    "Repair quality": ('Repair the {defect} and restore natural, clear speech.', {"defect": "telephone effect"}),
    "Separate speaker": ("Keep only the {speaker} speaker to start talking and remove all other speakers.", {"speaker": "first"}),
    "Extract singing": ("Keep only the singing voice and remove everything else.", {}),
    "Keep human voices": ("Keep all human voices, including speech and singing, and remove everything else.", {}),
    "Extract target speaker": ('Keep only the speaker who says "{text}" and remove all other speakers.', {"text": "words spoken by the target"}),
}


class AuKInstructionBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for name, (_, fields) in TASKS.items():
            inputs = [io.Float.Input(key, default=value, min=0, max=100, step=0.1) if isinstance(value, float) else io.String.Input(key, default=value, multiline=True) for key, value in fields.items()]
            options.append(io.DynamicCombo.Option(name, inputs))
        return io.Schema(node_id="YogurtAuKInstructionBuilder", display_name="Yogurt AuK Instruction Builder", category="YogurtNodes/AuK", inputs=[io.DynamicCombo.Input("task", options=options, tooltip="Task template. Fill its fields and wire the STRING output into Instruction Encode's instruction input, or edit the generated text afterwards.")], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, task):
        template, fields = TASKS[task["task"]]
        return io.NodeOutput(template.format(**{key: task[key] for key in fields}))


class AuKPromptEnhance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtAuKPromptEnhance",
            display_name="Yogurt AuK Prompt Enhance",
            category="YogurtNodes/AuK",
            description="Local Prompt Enhancer driven by Qwen2.5-Omni-3B. It uses an optional ASR transcript to render the model instruction, calculate task-aware duration, and prepare whisper audio levels. Wire all applicable outputs forward.",
            inputs=[
                AUK_ENCODER.Input("encoder", tooltip="Encoder from AuK Encoder Loader. Its checkpoint must include the language head - use a released qwen_omni checkpoint if the language head is missing."),
                io.String.Input("instruction", multiline=True, default="make her sound excited and say welcome home", tooltip="Loose request in any wording. The language model picks the closest AuK task and fills its template."),
                io.String.Input("context", multiline=True, default="", optional=True, tooltip="Optional extra context, e.g. an existing transcript of the source audio."),
                io.Audio.Input("audio", optional=True, tooltip="Source/reference audio. Required for edit-task duration calculation and whisper audio preparation. When its transcript is wired into context, TTS seconds are scaled to this voice's measured speaking pace instead of the generic estimate. Connect the prepared_audio output to Instruction Encode.audio."),
                io.Int.Input("max_new_tokens", default=256, min=32, max=1024, optional=True, tooltip="Generation budget for the language-model answer."),
            ],
            outputs=[io.String.Output(display_name="instruction"), io.Float.Output(display_name="seconds"), io.String.Output(display_name="task"), io.Audio.Output(display_name="prepared_audio")],
        )

    @classmethod
    def execute(cls, encoder, instruction, context="", audio=None, max_new_tokens=256):
        text, seconds, task, prepared = prompt_enhance.prepare(
            encoder, instruction, TASKS, audio=audio, context=context or None,
            max_new_tokens=max_new_tokens,
        )
        return io.NodeOutput(text, seconds, task, prepared, ui=ui.PreviewText(text))


class AuKExtension(ComfyExtension):
    async def get_node_list(self):
        return [AuKModelLoader, AuKEncoderLoader, AuKVAELoader, AuKInstructionEncode, AuKGenerateEdit, AuKInstructionBuilder, AuKPromptEnhance]
