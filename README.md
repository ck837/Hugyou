# Hugyou

Hugyou 是一个本地 Hugging Face 文生图 Demo。项目默认使用轻量模型 `segmind/SSD-1B`，通过 `diffusers` 在本机 GPU 上生成 PNG 图片，并内置低显存优化策略。

已在 RTX 4060 Laptop GPU 8GB 上跑通：`512x512`、`5 steps` 峰值已分配显存约 `2.65 GB`。

## 模型

默认模型：

```text
segmind/SSD-1B
```

可选实验 profile：

```text
black-forest-labs/FLUX.1-schnell
```

FLUX profile 使用 4-bit bitsandbytes 量化加载，但对 CUDA、bitsandbytes 和显存更挑剔；默认推荐先使用 SSD-1B。

## 环境

推荐 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安装 CUDA 12.1 版 PyTorch：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```powershell
pip install -r requirements.txt
```

如需运行 FLUX 4-bit profile：

```powershell
pip install -r requirements-flux.txt
```

## 运行

默认生成：

```powershell
python .\scripts\generate_image.py
```

低成本快速测试：

```powershell
python .\scripts\generate_image.py --width 512 --height 512 --steps 5
```

自定义 prompt：

```powershell
python .\scripts\generate_image.py `
  --prompt "A retro-futuristic cyberpunk city center, cinematic lighting, Wes Anderson color palette, 8k resolution" `
  --width 768 --height 768 --steps 20
```

FLUX schnell 4-bit：

```powershell
python .\scripts\generate_image.py --profile flux-schnell-4bit --steps 4 --width 768 --height 768
```

## 在其他电脑 / Codex 中运行

如果你在另一台电脑或另一个 Codex 工作区 clone 本项目，直接按下面文档执行：

[docs/CODEX_RUNBOOK.md](docs/CODEX_RUNBOOK.md)

注意：模型权重不会提交到 GitHub。第一次运行会自动从 Hugging Face 下载 `segmind/SSD-1B` 到本地 `models/`，之后会复用缓存。

输出文件会保存到：

```text
outputs/
```

模型缓存会保存到：

```text
models/
```

## 显存优化

脚本默认启用：

- `torch.float16`
- `pipe.enable_model_cpu_offload()`
- attention slicing
- VAE slicing
- VAE tiling
- `tqdm` Denoising 进度条

## 项目结构

```text
.
├── scripts/
│   └── generate_image.py
├── requirements.txt
├── requirements-flux.txt
├── .gitignore
└── README.md
```
