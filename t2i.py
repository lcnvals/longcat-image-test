"""LongCat-Image 文生图，Apple Silicon (MPS, 32GB) 分段推理。

文本编码器 (~16.6GB) 与 transformer (~12.5GB) 不同时驻留内存：
  阶段 1：只加载文本编码器（Qwen2.5-VL），可选改写提示词，编码正/负提示词后释放；
  阶段 2：只加载 transformer + VAE，用缓存的 embeds 去噪出图。

用法：
  .venv/bin/python t2i.py "提示词" [-o out.png] [--width 1024 --height 1024] [--steps 50]
                          [--cfg 4.0] [--seed 43] [--rewrite]
官方示例：1344x768、50 步、cfg 4.0、rewrite 开。M1 Pro 上很慢，先用小图少步数试。
"""
import argparse
import gc
import time

import torch
from diffusers import LongCatImagePipeline
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

MODEL = "./model-t2i"
DEVICE = "mps"
DTYPE = torch.bfloat16


def free():
    gc.collect()
    torch.mps.empty_cache()


def encode(prompt: str, negative: str, rewrite: bool, te_device: str):
    pipe = LongCatImagePipeline.from_pretrained(MODEL, transformer=None, vae=None, torch_dtype=DTYPE)
    pipe.text_encoder.to(te_device)
    with torch.no_grad():
        if rewrite:
            prompt = pipe.rewire_prompt(prompt, te_device)[0]
            print(f"改写后的提示词：{prompt}")
        pos = pipe._encode_prompt([prompt]).to("cpu")
        neg = pipe._encode_prompt([negative]).to("cpu")
    del pipe
    free()
    return pos, neg


def denoise(pos, neg, a):
    pipe = LongCatImagePipeline.from_pretrained(MODEL, text_encoder=None, torch_dtype=DTYPE).to(DEVICE)

    # 上游 encode_prompt 在 prompt=None 时会 len(None) 报错，这里直接返回缓存的 embeds
    def cached_encode(prompt=None, num_images_per_prompt=1, prompt_embeds=None):
        ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1])
        return prompt_embeds.to(DEVICE), ids.to(DEVICE)

    pipe.encode_prompt = cached_encode

    img = pipe(
        prompt=None,
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        height=a.height,
        width=a.width,
        guidance_scale=a.cfg,
        num_inference_steps=a.steps,
        generator=torch.Generator("cpu").manual_seed(a.seed),
        enable_cfg_renorm=True,
        enable_prompt_rewrite=False,  # 已在阶段 1 处理
    ).images[0]
    del pipe
    free()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", default="t2i_output.png")
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--rewrite", action="store_true", help="用文本编码器先改写提示词（更慢）")
    ap.add_argument("--te-device", default=DEVICE, choices=["mps", "cpu"],
                    help="文本编码器运行设备；24GB 及以下内存的 Mac 用 cpu")
    a = ap.parse_args()

    t0 = time.time()
    pos, neg = encode(a.prompt, a.negative, a.rewrite, a.te_device)
    print(f"[1/2] 文本编码完成 {time.time() - t0:.0f}s, embeds {tuple(pos.shape)}")
    t1 = time.time()
    img = denoise(pos, neg, a)
    img.save(a.output)
    print(f"[2/2] 去噪完成 {time.time() - t1:.0f}s -> {a.output}")


if __name__ == "__main__":
    main()
