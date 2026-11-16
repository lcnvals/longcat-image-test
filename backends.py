"""studio.py 的模型后端。每个后端把一个模型包装成统一的 generate() 接口。

运行环境（依赖互不兼容，分开装）：
  Mac：
    longcat                        -> .venv      （PyTorch + diffusers，MPS）
    z-image-turbo / flux2-klein-4b -> .venv-mlx  （mflux，Apple MLX 原生，更省内存、更快）
  Linux + NVIDIA（deploy/setup_cuda.sh 一键安装）：
    三个模型都用 .venv-cuda（PyTorch + diffusers，CUDA）；同名 --model，自动选用 CUDA 实现

依赖都在各后端内部导入，所以 studio.py 在两个环境里都能运行。
"""
import gc
import os
import platform
import time


def torch_device():
    """CUDA（NVIDIA）优先，其次 MPS（Mac），最后 CPU。"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_empty_cache(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Backend:
    key = ""                 # --model 的名字
    title = ""               # 显示用名称
    env = ""                 # 需要的虚拟环境
    preview_steps = 8
    final_steps = 25
    default_size = "768x768"
    cfg = None               # 默认 CFG；None 表示模型不支持（蒸馏模型）
    supports_neg = False     # 是否支持负面提示词
    out_dir = "./outputs"

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        """返回 (PIL.Image, 实际执行的步数)。init/strength：以 init 图为底稿精修，strength = 重画比例。"""
        raise NotImplementedError

    def cleanup(self):
        gc.collect()


class LongCatBackend(Backend):
    key = "longcat"
    title = "LongCat-Image（6B，PyTorch/MPS）"
    env = ".venv"
    preview_steps, final_steps, default_size = 8, 25, "768x768"
    cfg, supports_neg = 4.0, True
    out_dir = "./outputs"    # 沿用原有目录，保留历史记录
    MODEL = "./model-t2i"

    def __init__(self, quant_transformer=False, **_):
        import torch
        from diffusers import LongCatImagePipeline
        from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids
        from optimum.quanto import freeze, qint8, quantize
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device, self.dtype = torch_device(), torch.bfloat16
        t0 = time.time()
        # 先单独加载并量化文本编码器（此时 transformer 还没进内存，避免峰值叠加）
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"{self.MODEL}/text_encoder", dtype=self.dtype)
        quantize(te, weights=qint8)
        freeze(te)
        gc.collect()
        te.to(self.device)
        print(f"  文本编码器 int8 就绪 {time.time() - t0:.0f}s")

        pipe = LongCatImagePipeline.from_pretrained(self.MODEL, text_encoder=te, dtype=self.dtype)
        if quant_transformer:
            quantize(pipe.transformer, weights=qint8)
            freeze(pipe.transformer)
            gc.collect()
        pipe.transformer.to(self.device)
        pipe.vae.to(self.device)

        # 上游 encode_prompt 在 prompt=None 时会 len(None)，这里直接用缓存的 embeds
        def cached_encode(prompt=None, num_images_per_prompt=1, prompt_embeds=None):
            ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1])
            return prompt_embeds.to(self.device), ids.to(self.device)

        pipe.encode_prompt = cached_encode
        pipe.set_progress_bar_config(disable=self.device == "mps")  # MPS 异步，进度条时间不准
        self.pipe = pipe
        self.cache = {}
        print(f"  全部加载完成 {time.time() - t0:.0f}s")

    def embeds(self, text):
        if text not in self.cache:
            with self.torch.no_grad():
                self.cache[text] = self.pipe._encode_prompt([text])
        return self.cache[text]

    def refine_inputs(self, src, w, h, steps, strength, seed):
        """把源图编码成潜变量，按强度加噪，返回 pipeline 的 latents 和截断后的 sigmas。"""
        import copy

        import numpy as np
        from diffusers.pipelines.longcat_image.pipeline_longcat_image import calculate_shift
        from PIL import Image

        torch, pipe = self.torch, self.pipe
        full = np.linspace(1.0, 1.0 / steps, steps)  # 与 pipeline 默认一致
        sigmas = full[full <= strength + 1e-6]
        if len(sigmas) == 0:
            sigmas = full[-1:]
        # pipeline 会对 sigmas 做动态 shift，起始噪声强度要按 shift 后的值算
        seq_len = (h // 16) * (w // 16)
        cfg = pipe.scheduler.config
        mu = calculate_shift(seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        sched = copy.deepcopy(pipe.scheduler)
        sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        s0 = float(sched.sigmas[0])

        with torch.no_grad():
            img = Image.open(src).convert("RGB")
            x = pipe.image_processor.preprocess(img, h, w).to(self.device, pipe.vae.dtype)
            lat = pipe.vae.encode(x).latent_dist.mode()
            lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            noise = torch.randn(lat.shape, generator=torch.Generator("cpu").manual_seed(seed))
            lat = (1 - s0) * lat + s0 * noise.to(self.device, lat.dtype)  # flow matching：x_t = (1-σ)·x0 + σ·ε
            b, c, lh, lw = lat.shape
            packed = pipe._pack_latents(lat, b, c, lh, lw)
        return {"latents": packed.to(self.dtype), "sigmas": list(sigmas)}

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        extra = self.refine_inputs(init, w, h, steps, strength, seed) if init else {}
        img = self.pipe(
            **extra,
            prompt=None,
            prompt_embeds=self.embeds(prompt),
            negative_prompt_embeds=self.embeds(neg),
            height=h,
            width=w,
            guidance_scale=cfg,
            num_inference_steps=len(extra["sigmas"]) if init else steps,
            generator=self.torch.Generator("cpu").manual_seed(seed),
            enable_cfg_renorm=True,
            enable_prompt_rewrite=False,
        ).images[0]
        return img, len(extra["sigmas"]) if init else steps

    def cleanup(self):
        torch_empty_cache(self.torch)


class MfluxBackend(Backend):
    """mflux（Apple MLX）模型。蒸馏模型：不支持 CFG 和负面提示词，步数很少。"""
    env = ".venv-mlx"
    cfg, supports_neg = None, False
    default_size = "1024x1024"
    quantize = 8

    def __init__(self, quantize=None, **_):
        q = quantize or self.quantize
        path = f"./models/{self.key}"
        if not os.path.isdir(path):
            raise SystemExit(f"找不到模型目录 {path}，先按 README 从魔搭下载")
        t0 = time.time()
        self.model = self.load(path, q)
        self.out_dir = f"./outputs/{self.key}"
        print(f"  {self.title} 加载完成（{q}-bit）{time.time() - t0:.0f}s")

    def load(self, path, q):
        raise NotImplementedError

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        kw = {}
        if init:
            # mflux 的 image_strength = 保留原图的比例（从第 steps×strength 步开始），与这里的“重画比例”相反
            kw = {"image_path": init, "image_strength": 1 - strength}
        result = self.model.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps,
                                           height=h, width=w, **kw)
        ran = steps - max(1, int(steps * (1 - strength))) if init else steps
        return result.image, ran

    def cleanup(self):
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


class ZImageTurboBackend(MfluxBackend):
    key = "z-image-turbo"
    title = "Z-Image-Turbo（6B，MLX）"
    preview_steps = final_steps = 9

    def load(self, path, q):
        from mflux.models.z_image import ZImageTurbo
        return ZImageTurbo(quantize=q, model_path=path)


class Flux2KleinBackend(MfluxBackend):
    key = "flux2-klein-4b"
    title = "FLUX.2 klein 4B（MLX）"
    preview_steps = final_steps = 4

    def load(self, path, q):
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein
        return Flux2Klein(quantize=q, model_path=path, model_config=ModelConfig.flux2_klein_4b())


class DiffusersImageBackend(Backend):
    """Linux + NVIDIA 上用 diffusers 跑 Z-Image / FLUX.2 klein。模型目录与 Mac 上 mflux 用的相同（diffusers 格式）。

    注意：这部分是为云服务器准备的，作者没有 NVIDIA 机器，尚未实测。
    """
    env = ".venv-cuda"
    cfg, supports_neg = None, False
    default_size = "1024x1024"
    pipeline = ""
    guidance = 0.0

    def __init__(self, **_):
        import diffusers
        import torch
        self.torch = torch
        self.device = torch_device()
        path = f"./models/{self.key}"
        if not os.path.isdir(path):
            raise SystemExit(f"找不到模型目录 {path}，先运行 deploy/setup_cuda.sh 下载模型")
        t0 = time.time()
        cls = getattr(diffusers, self.pipeline)
        self.pipe = cls.from_pretrained(path, torch_dtype=torch.bfloat16)
        vram = torch.cuda.mem_get_info()[1] / 1e9 if self.device == "cuda" else 0  # 显存总量
        if self.device == "cuda" and vram < 20:
            self.pipe.enable_model_cpu_offload()   # 显存小于 20GB：按组件在显存和内存之间搬运
        else:
            self.pipe.to(self.device)
        self.out_dir = f"./outputs/{self.key}"
        print(f"  {self.title} 加载完成（{self.device}）{time.time() - t0:.0f}s")

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        torch = self.torch
        kw = dict(prompt=prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=self.guidance,
                  generator=torch.Generator("cpu").manual_seed(seed))
        if init:
            img2img = getattr(self, "img2img", None)
            if not img2img:
                raise RuntimeError(f"{self.title} 的 CUDA 实现暂不支持 /refine")
            from PIL import Image
            pipe = img2img.from_pipe(self.pipe)
            img = pipe(image=Image.open(init).convert("RGB"), strength=strength, **kw).images[0]
            return img, max(1, int(steps * strength))
        return self.pipe(**kw).images[0], steps

    def cleanup(self):
        torch_empty_cache(self.torch)


class ZImageTurboCuda(DiffusersImageBackend):
    key = "z-image-turbo"
    title = "Z-Image-Turbo（6B，CUDA）"
    pipeline = "ZImagePipeline"
    preview_steps = final_steps = 8
    guidance = 0.0

    @property
    def img2img(self):
        import diffusers
        return diffusers.ZImageImg2ImgPipeline


class Flux2KleinCuda(DiffusersImageBackend):
    key = "flux2-klein-4b"
    title = "FLUX.2 klein 4B（CUDA）"
    pipeline = "Flux2KleinPipeline"
    preview_steps = final_steps = 4
    guidance = 1.0


if platform.system() == "Darwin":
    BACKENDS = {b.key: b for b in (LongCatBackend, ZImageTurboBackend, Flux2KleinBackend)}
else:  # Linux / NVIDIA：同名模型换成 diffusers 实现，分镜文件和命令不用改
    BACKENDS = {b.key: b for b in (LongCatBackend, ZImageTurboCuda, Flux2KleinCuda)}
