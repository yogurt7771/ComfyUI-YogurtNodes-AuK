# ComfyUI-YogurtNodes-AuK

AuK Base / Flash 的本地语音生成、声音克隆与音频编辑插件。节点分类：`YogurtNodes/AuK`。

Qwen2.5-Omni 使用 ComfyUI 原生 Qwen 文本层及插件内的音频编码器，分词使用 `tokenizers`。插件不导入外部 `transformers`，不修改 `sys.path`，不需要安装另一个 AuK 仓库。模型由 ComfyUI 的 model patcher 管理加载和卸载，注意力走 ComfyUI 已选择的优化后端。

## 安装与权重

将本目录放入 `ComfyUI/custom_nodes/`，使用运行 ComfyUI 的 Python 安装 `requirements.txt`，然后重启 ComfyUI。需要包含原生 Qwen2.5 和 `mixed_precision_ops` 的当前 ComfyUI；量化权重还需要相应的 Comfy Kitchen 格式支持。不会安装或替换 PyTorch、CUDA、Transformers。

三个加载节点均有 `download_if_missing` 开关，默认关闭。开启后仅下载选中的已知权重；已有文件直接复用。默认选择官方权重：[Tencent AuK Base](https://huggingface.co/tencent/AuK/tree/main)、[Tencent AuK-Flash](https://huggingface.co/tencent/AuK-Flash/tree/main) 和 [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/main)。中断或失败的下载不会留下可加载的半成品模型。

| 组件 | 保存位置 | 可选权重 |
| --- | --- | --- |
| AuK | `ComfyUI/models/diffusion_models/` | 官方 `auk_base.safetensors` 或 `auk_flash.safetensors` |
| Qwen 编码器 | `ComfyUI/models/text_encoders/Qwen2.5-Omni-3B/` | 官方 `model-00001-of-00003.safetensors` 和 `model-00002-of-00003.safetensors` |
| VAE | `ComfyUI/models/vae/AuK/` | 官方 `vae.safetensors` |

权重保存在上述 ComfyUI 模型目录，**不写入 Hugging Face 缓存目录**。官方 Qwen 的第三分片只有本插件不用的 Talker 等权重，加载节点只下载前两片中的 Thinker 文本和音频编码器。本地模型路径使用 `folder_paths`，因此可读取 `extra_model_paths.yaml` 注册的对应目录及子目录；自动下载目标使用 ComfyUI 当前 `models_dir`。配置和分词器随插件提供。

如需减少磁盘和显存占用，仍可选择 [drbaph/AuK-comfyui](https://huggingface.co/drbaph/AuK-comfyui/tree/main) 发布的第三方 BF16、INT8、W4A8 转换文件：AuK 放在 `diffusion_models/`，`qwen_omni_*.safetensors` 放在 `text_encoders/`，`auk_vae.safetensors` 放在 `vae/`。选择这些文件才会从该第三方仓库下载。`precision` 只控制运算精度，量化格式从 checkpoint 读取；VAE 始终 FP32。

## 工作流

将 `example_workflows/` 中的 JSON 拖入 ComfyUI：

- `01_text_to_speech.json`：描述音色并输入需要朗读的文字。
- `02_audio_edit.json`：先在 Load Audio 选择音频，默认清理噪声与混响。
- `03_flash_voice_clone.json`：先选择参考音频，再修改朗读内容；Flash 固定四步。

基本连接：三个加载节点 → Instruction Encode → Generate / Edit → ComfyUI 的 Save Audio。模型同时连接 Instruction Encode 和 Generate / Edit，编码器连接 Instruction Encode，VAE 连接 Generate / Edit。

| 节点 | 用途 |
| --- | --- |
| Yogurt AuK Model Loader | 加载 Base / Flash 模型 |
| Yogurt AuK Encoder Loader | 加载 Qwen 文本和音频编码器 |
| Yogurt AuK VAE Loader | 加载音频 VAE |
| Yogurt AuK Instruction Encode | 指令和可选参考音频编码 |
| Yogurt AuK Generate / Edit | 生成 24 kHz 单声道 `AUDIO` |
| Yogurt AuK Instruction Builder | 选择任务模板并填写字段 |
| Yogurt AuK Prompt Enhance | 用本地 Qwen 语言头整理自然语言请求，估计时长并准备参考音频 |

Prompt Enhance 是可选步骤：其 instruction、seconds、prepared_audio 分别连接 Instruction Encode.instruction、Generate / Edit.seconds、Instruction Encode.audio。context 可填写已有的参考音频转录文本，用于估计说话速度。无需外部 LLM API，也不自动运行 Whisper。

Base 默认 32 步、guidance=2、sway=-1。Flash 使用固定四步并忽略这三个设置。`seconds=0` 使用参考音频时长；纯文本生成必须填写正数。输出批次逐项使用 `seed + index`，同时控制参考 VAE 采样和扩散噪声。不会截断长参考音频，也不强加 30 秒的总时长限制；长音频会增加耗时和显存。

音频编辑直接填写自然语言指令，例如 `Replace 'Tuesday' with 'Friday'.`、`Raise the pitch by 2 semitones.`、`Extract only the singing voice and remove accompaniment.`。模型对不同编辑指令的遵循程度不同，结果应实际试听。这里的条件和模型接口专用于 AuK，不能直接连接 KSampler 或 CLIP Text Encode；VAE 适配器不实现 tiled 编解码。

## 来源

实现参考 [Saganaki22/ComfyUI-AuK](https://github.com/Saganaki22/ComfyUI-AuK)、[UnlimitedEditing/ComfyUI-AuK](https://github.com/UnlimitedEditing/ComfyUI-AuK) 和本地 YogurtNodes。具体来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
