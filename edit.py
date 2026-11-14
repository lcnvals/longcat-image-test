"""LongCat-Image-Edit-Turbo 在 Apple Silicon (MPS, 32GB) 上的分段推理。

文本编码器 (~16.6GB) 和 transformer (~12.5GB) 不同时驻留内存：
  阶段 1：只加载文本编码器，算 prompt_embeds，然后释放；
  阶段 2：只加载 transformer + VAE，用缓存的 embeds 去噪出图。

用法：
  .venv/bin/python edit.py 输入图片 "编辑指令" [-o 输出.png] [--steps 8] [--seed 43]
"""
import argparse
import gc
import time

import torch
from PIL import Image
from diffusers import LongCatImageEditPipeline
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import calculate_dimensions

MODEL = "./model"
DEVICE = "mps"
DTYPE = torch.bfloat16


def free():
    gc.collect()
    torch.mps.empty_cache()


def encode(image: Image.Image, prompt: str) -> torch.Tensor:
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL, transformer=None, vae=None, torch_dtype=DTYPE)
    pipe.text_encoder.to(DEVICE)
    # 与 pipeline.__call__ 内部一致的尺寸计算：文本编码器看的是半分辨率图
    w, h = calculate_dimensions(1024 * 1024, image.size[0] / image.size[1])
    prompt_image = pipe.image_processor.resize(pipe.image_processor.resize(image, h, w), h // 2, w // 2)
    with torch.no_grad():
        embeds = pipe._encode_prompt([prompt], prompt_image).to("cpu")
    del pipe
    free()
    return embeds


def denoise(image: Image.Image, embeds: torch.Tensor, steps: int, seed: int) -> Image.Image:
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL, text_encoder=None, torch_dtype=DTYPE).to(DEVICE)
    out = pipe(
        image,
        prompt=None,
        prompt_embeds=embeds.to(DEVICE),
        guidance_scale=1,  # Turbo 蒸馏模型不用 CFG，也就不需要 negative embeds
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    del pipe
    free()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", default="edit_output.png")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    t0 = time.time()
    embeds = encode(img, a.prompt)
    t1 = time.time()
    print(f"[1/2] 文本编码完成 {t1 - t0:.0f}s, embeds {tuple(embeds.shape)}")
    result = denoise(img, embeds, a.steps, a.seed)
    t2 = time.time()
    result.save(a.output)
    print(f"[2/2] 去噪完成 {t2 - t1:.0f}s -> {a.output}")


if __name__ == "__main__":
    main()
