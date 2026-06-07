from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


DEFAULT_PROMPT = (
    "A retro-futuristic cyberpunk city center, cinematic lighting, "
    "Wes Anderson color palette, 8k resolution"
)


@dataclass(frozen=True)
class ModelProfile:
    repo_id: str
    kind: str
    default_width: int
    default_height: int
    default_steps: int
    default_guidance: float
    negative_prompt: str | None = None


PROFILES = {
    "ssd1b": ModelProfile(
        repo_id="segmind/SSD-1B",
        kind="sdxl",
        default_width=1024,
        default_height=1024,
        default_steps=25,
        default_guidance=9.0,
        negative_prompt="ugly, blurry, low quality, distorted, watermark, text artifacts",
    ),
    "flux-schnell-4bit": ModelProfile(
        repo_id="black-forest-labs/FLUX.1-schnell",
        kind="flux4bit",
        default_width=768,
        default_height=768,
        default_steps=4,
        default_guidance=0.0,
        negative_prompt=None,
    ),
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def print_device_report() -> None:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        log(f"CUDA device: {props.name} | VRAM: {total_gb:.1f} GB")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        log("Apple MPS device detected")
    else:
        log("No GPU detected; CPU mode will work but can be very slow")


def download_sdxl_snapshot(repo_id: str, cache_dir: Path, local_files_only: bool) -> str:
    log(f"下载中: {repo_id}")
    log("提示: huggingface_hub 会显示文件级进度条；首次运行需要较长时间。")
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        ignore_patterns=[
            "*.ckpt",
            "*.bin",
            "*.onnx",
            "*.msgpack",
            "SSD-1B*.safetensors",
            "*modelspec*",
        ],
    )


def load_sdxl_pipe(
    repo_id: str,
    cache_dir: Path,
    dtype: torch.dtype,
    local_files_only: bool,
) -> Any:
    from diffusers import StableDiffusionXLPipeline

    model_path = download_sdxl_snapshot(repo_id, cache_dir, local_files_only)
    log("加载到内存/显存: StableDiffusionXLPipeline")

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": True,
    }
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"

    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **kwargs)
    return pipe


def load_flux_4bit_pipe(repo_id: str, dtype: torch.dtype, local_files_only: bool) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("flux-schnell-4bit 需要 CUDA + bitsandbytes；无 CUDA 时请使用默认 ssd1b。")

    log(f"下载/加载中: {repo_id} | 4-bit NF4 quantized transformer + T5 encoder")
    from diffusers import (
        BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
        FluxPipeline,
        FluxTransformer2DModel,
    )
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from transformers import T5EncoderModel

    compute_dtype = torch.bfloat16
    quant_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_use_double_quant": True,
    }

    transformer = FluxTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        quantization_config=DiffusersBitsAndBytesConfig(**quant_kwargs),
        torch_dtype=compute_dtype,
        local_files_only=local_files_only,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder_2",
        quantization_config=TransformersBitsAndBytesConfig(**quant_kwargs),
        torch_dtype=compute_dtype,
        local_files_only=local_files_only,
    )
    pipe = FluxPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=compute_dtype,
        local_files_only=local_files_only,
    )
    return pipe


def apply_memory_optimizations(pipe: Any) -> None:
    log("显存优化: attention slicing / VAE slicing / VAE tiling / CPU offload")

    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(pipe, method_name, None)
        if callable(method):
            method()

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        pipe.to("mps")
    else:
        pipe.to("cpu")


def run_with_denoising_progress(pipe: Any, call_kwargs: dict[str, Any], steps: int) -> Any:
    signature = inspect.signature(pipe.__call__)
    supports_step_callback = "callback_on_step_end" in signature.parameters

    if supports_step_callback:
        pipe.set_progress_bar_config(disable=True)
        with tqdm(total=steps, desc="Denoising", unit="step") as bar:

            def callback_on_step_end(pipe_obj: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]):
                bar.update(1)
                return callback_kwargs

            log("正在去噪渲染（Denoising）")
            return pipe(callback_on_step_end=callback_on_step_end, **call_kwargs)

    pipe.set_progress_bar_config(desc="Denoising")
    log("正在去噪渲染（Denoising）")
    return pipe(**call_kwargs)


def build_call_kwargs(args: argparse.Namespace, profile: ModelProfile, generator: torch.Generator) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.width or profile.default_width,
        "height": args.height or profile.default_height,
        "num_inference_steps": args.steps or profile.default_steps,
        "guidance_scale": args.guidance if args.guidance is not None else profile.default_guidance,
        "generator": generator,
    }

    if profile.kind == "sdxl":
        kwargs["negative_prompt"] = args.negative_prompt or profile.negative_prompt
    elif profile.kind == "flux4bit":
        kwargs["max_sequence_length"] = args.max_sequence_length

    return kwargs


def save_image(image: Any, output_dir: Path, profile_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{profile_name}_{stamp}.png"
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Hugging Face image generation demo.")
    parser.add_argument("--profile", choices=PROFILES.keys(), default="ssd1b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    args = parse_args()
    profile = PROFILES[args.profile]

    log(f"Profile: {args.profile} | Model: {profile.repo_id}")
    print_device_report()
    dtype = pick_dtype()
    log(f"Torch dtype: {dtype}")

    if args.profile == "ssd1b":
        pipe = load_sdxl_pipe(profile.repo_id, args.cache_dir, dtype, args.local_files_only)
    else:
        pipe = load_flux_4bit_pipe(profile.repo_id, dtype, args.local_files_only)

    apply_memory_optimizations(pipe)

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    call_kwargs = build_call_kwargs(args, profile, generator)

    log(
        "推理参数: "
        f"{call_kwargs['width']}x{call_kwargs['height']} | "
        f"steps={call_kwargs['num_inference_steps']} | "
        f"guidance={call_kwargs['guidance_scale']} | seed={args.seed}"
    )

    with torch.inference_mode():
        result = run_with_denoising_progress(pipe, call_kwargs, call_kwargs["num_inference_steps"])

    output_path = save_image(result.images[0], args.output_dir, args.profile)
    log(f"保存完成: {output_path.resolve()}")

    if torch.cuda.is_available():
        used_gb = torch.cuda.max_memory_allocated() / 1024**3
        log(f"本次峰值已分配显存: {used_gb:.2f} GB")


if __name__ == "__main__":
    main()
