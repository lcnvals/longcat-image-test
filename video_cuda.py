"""Linux + NVIDIA 上的视频生成运行器（LTX-2，diffusers）。由 video.py 通过子进程调用，也可以单独运行。

  .venv-cuda/bin/python video_cuda.py -p "橘猫慢慢眨眼" --image cat.png -W 768 -H 512 -f 49 --fps 24 -o out.mp4

显存策略（自动）：
  ≥ 80GB  全部放进显存，最快
  ≥ 40GB  按组件在显存和内存之间搬运（enable_model_cpu_offload），需要约 64GB 内存
  < 40GB  逐层搬运（enable_sequential_cpu_offload），能跑但很慢；RTX 4090 属于这一档

注意：作者没有 NVIDIA 机器，这个文件尚未在真实 GPU 上测试过。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="LTX-2 视频生成（CUDA / diffusers）")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--image", help="首帧图片（图生视频）")
    ap.add_argument("--negative", default="worst quality, inconsistent motion, blurry, jittery, distorted")
    ap.add_argument("-W", "--width", type=int, default=768)
    ap.add_argument("-H", "--height", type=int, default=512)
    ap.add_argument("-f", "--frames", type=int, default=49)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ltx-2"))
    ap.add_argument("--offload", choices=["auto", "none", "model", "sequential"], default="auto")
    ap.add_argument("--audio", action="store_true", help="把生成的音频一起写进视频")
    a = ap.parse_args()

    import torch
    from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline
    from diffusers.utils import encode_video, load_image

    if not torch.cuda.is_available():
        sys.exit("没有检测到 NVIDIA GPU（torch.cuda 不可用）")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    offload = a.offload
    if offload == "auto":
        offload = "none" if vram >= 80 else "model" if vram >= 40 else "sequential"
    print(f"GPU {torch.cuda.get_device_name(0)}，显存 {vram:.0f}GB，策略：{offload}", flush=True)

    t0 = time.time()
    cls = LTX2ImageToVideoPipeline if a.image else LTX2Pipeline
    pipe = cls.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    if offload == "none":
        pipe.to("cuda")
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
    print(f"加载 {time.time() - t0:.0f}s", flush=True)

    kw = dict(prompt=a.prompt, negative_prompt=a.negative, width=a.width, height=a.height,
              num_frames=a.frames, frame_rate=a.fps, num_inference_steps=a.steps, guidance_scale=a.guidance,
              generator=torch.Generator("cpu").manual_seed(a.seed), output_type="np", return_dict=False)
    if a.image:
        kw["image"] = load_image(a.image)
    t1 = time.time()
    video, audio = pipe(**kw)
    print(f"生成 {time.time() - t1:.0f}s，显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    extra = {}
    if a.audio and audio is not None:
        extra = dict(audio=audio[0].float().cpu(), audio_sample_rate=pipe.vocoder.config.output_sampling_rate)
    encode_video(video[0], fps=a.fps, output_path=a.output, **extra)
    print(f"-> {a.output}", flush=True)


if __name__ == "__main__":
    main()
