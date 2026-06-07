# Hugyou Codex Runbook

这份文档用于在另一台电脑、另一个 Codex 工作区或全新环境里复现本项目。

## 项目做什么

Hugyou 是一个本地 Hugging Face 文生图 Demo：

- 默认模型：`segmind/SSD-1B`
- 推理框架：`diffusers`
- 输出格式：PNG
- 默认脚本：`scripts/generate_image.py`
- 模型缓存目录：`models/`
- 图片输出目录：`outputs/`

`models/` 和 `outputs/` 不会提交到 GitHub。换电脑第一次运行会重新下载模型。

## 已验证环境

当前项目已经在下面环境跑通：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8 GB
PyTorch: torch 2.5.1+cu121
CUDA wheel: cu121
Model: segmind/SSD-1B
Test run: 512x512, 20 steps
Peak allocated VRAM: about 2.65 GB
```

首次运行因为要下载模型，用时约 18 分钟。模型缓存后，`512x512`、`20 steps` 约 25 秒，其中 Denoising 约 12 秒。

## 全新电脑运行步骤

Windows PowerShell：

```powershell
git clone https://github.com/ck837/Hugyou.git
cd Hugyou

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

低成本测试：

```powershell
python .\scripts\generate_image.py --width 512 --height 512 --steps 5
```

正常测试：

```powershell
python .\scripts\generate_image.py --width 512 --height 512 --steps 20
```

自定义 prompt：

```powershell
python .\scripts\generate_image.py `
  --prompt "A rich hamburger combo meal featuring a juicy burger, crispy french fries, fried chicken, and cola, on a bold red background, showcasing variety, abundance, and delicious fast food appeal" `
  --width 512 --height 512 --steps 20
```

## 中文提示词怎么用

建议先把中文提示词翻译成英文再传给脚本，SDXL 系模型通常英文更稳定。

示例中文：

```text
一个丰富的汉堡套餐、有汉堡薯条炸鸡可乐，红色背景展现套餐多样性和美味
```

推荐英文：

```text
A rich hamburger combo meal featuring a juicy burger, crispy french fries, fried chicken, and cola, on a bold red background, showcasing variety, abundance, and delicious fast food appeal
```

运行：

```powershell
python .\scripts\generate_image.py `
  --prompt "A rich hamburger combo meal featuring a juicy burger, crispy french fries, fried chicken, and cola, on a bold red background, showcasing variety, abundance, and delicious fast food appeal" `
  --width 512 --height 512 --steps 20
```

## CUDA / PyTorch 注意事项

README 里的 PyTorch 安装命令使用 CUDA 12.1 wheel：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果另一台电脑的显卡驱动不支持 CUDA 12.1，请去 PyTorch 官网换对应安装命令。

没有 NVIDIA CUDA GPU 也能运行，但 CPU 会非常慢，不建议用于实际生成。

## 依赖版本说明

`requirements.txt` 特意把这些依赖锁在稳定范围内：

```text
huggingface_hub>=0.24.0,<1.0
safetensors>=0.4.4,<0.7
transformers>=4.44.0,<5.0
```

原因：当前验证环境使用 `torch 2.5.1+cu121`。如果安装 `transformers 5.x`，可能触发新版 float8 dtype 相关兼容错误。

## FLUX 4-bit 可选 profile

项目保留了 FLUX schnell 4-bit profile：

```powershell
pip install -r requirements-flux.txt
python .\scripts\generate_image.py --profile flux-schnell-4bit --steps 4 --width 768 --height 768
```

但默认不推荐先跑 FLUX，因为它对 CUDA、bitsandbytes、显存和依赖版本更敏感。优先用 `segmind/SSD-1B` 跑通链路。

## 常见问题

### ModuleNotFoundError: No module named 'torch'

说明还没在当前虚拟环境安装 PyTorch。执行：

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 第一次运行很慢

正常。第一次会从 Hugging Face 下载模型到 `models/`。下载完成后再次运行会直接复用缓存。

### Hugging Face 下载失败

检查网络、代理和 Hugging Face 访问权限。这个模型默认不需要登录，但网络需要能访问 Hugging Face。

### GitHub push 失败，提示 127.0.0.1:7890

说明 Git 全局代理指向本地端口，但代理服务没启动。可以临时绕过代理：

```powershell
git -c http.proxy= -c https.proxy= push
```

### Git 提示 dubious ownership

在某些 Codex/Windows 环境里，Git 可能认为目录 owner 不一致。可以把项目加入 safe directory：

```powershell
git config --global --add safe.directory D:/project/huggingface
```

换成当前项目的实际路径即可。

