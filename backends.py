"""生图模型后端：把每个模型包装成统一的 generate() 接口。

模型清单在 models.json（随仓库）和 models.local.json（你自己加的，不进仓库）里配置，不用改代码：
  engine 选通用引擎：
    mflux     Apple MLX（Mac），按 module + class 加载 mflux 里的模型
    diffusers PyTorch（NVIDIA CUDA，也能在 Mac 的 MPS 上跑），按 pipeline 名加载 diffusers 管线
    longcat   LongCat-Image 专用实现（文本编码器量化、提示词缓存、精修）
  同一个模型可以在 "darwin" / "linux" 里给 Mac 和 Linux 分别配置（例如 Mac 用 mflux、Linux 用 diffusers）。

依赖都在引擎内部导入，所以 studio.py 在各个虚拟环境里都能运行。
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
    supports_refs = False    # 是否支持参考图生图（generate_refs）
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
    spec = {}

    def __init__(self, quant_transformer=False, **_):
        self.MODEL = resolve_path(self.spec.get("path", "./model-t2i"))
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
    """mflux（Apple MLX）通用引擎：按配置里的 module + class 加载。蒸馏模型一般不支持 CFG 和负面提示词。"""
    env = ".venv-mlx"
    spec = {}

    def __init__(self, quantize=None, **_):
        sp = self.spec
        q = quantize or sp.get("quantize", 8)
        path = resolve_path(sp.get("path", f"./models/{self.key}"))
        if not os.path.isdir(path):
            raise SystemExit(f"找不到模型目录 {path}，先下载模型（见 README 或 models.json 里的 download）")
        t0 = time.time()
        self.path, self.q = path, q
        self.model = self._build(sp.get("module"), sp.get("class"), path, q)
        self.out_dir = f"./outputs/{self.key}"
        print(f"  {self.title} 加载完成（{q}-bit）{time.time() - t0:.0f}s")

    def _build(self, module, cls_name, path, q):
        import importlib
        if not module or not cls_name:
            raise SystemExit(f"models.json 里 {self.key} 缺少 module 或 class")
        cls = getattr(importlib.import_module(module), cls_name)
        kw = dict(quantize=q, model_path=path)
        if self.spec.get("model_config"):  # 例如 flux2_klein_4b -> ModelConfig.flux2_klein_4b()
            from mflux.models.common.config import ModelConfig
            kw["model_config"] = getattr(ModelConfig, self.spec["model_config"])()
        return cls(**kw)

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        kw = {}
        if init:
            # mflux 的 image_strength = 保留原图的比例（从第 steps×strength 步开始），与这里的“重画比例”相反
            kw = {"image_path": init, "image_strength": 1 - strength}
        if self.spec.get("guidance") is not None:
            kw["guidance"] = self.spec["guidance"]
        result = self.model.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps,
                                           height=h, width=w, **kw)
        ran = steps - max(1, int(steps * (1 - strength))) if init else steps
        return result.image, ran

    def generate_refs(self, prompt, refs, w, h, steps, seed):
        """以设定图为参考生成（配置了 refs_class 的模型，如 FLUX.2 klein 的多参考图编辑）。按需加载。"""
        if not self.supports_refs:
            raise RuntimeError(f"{self.title} 没有配置 refs_class，不支持参考图")
        if not hasattr(self, "edit"):
            self.edit = self._build(self.spec.get("refs_module", self.spec.get("module")), self.spec["refs_class"],
                                    self.path, self.q)
        result = self.edit.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps,
                                          height=h, width=w, image_paths=list(refs))
        return result.image

    def cleanup(self):
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


class DiffusersBackend(Backend):
    """diffusers 通用引擎：按配置里的 pipeline 名加载（NVIDIA CUDA 优先，Mac 上用 MPS）。

    NVIDIA 上的配置作者没有实测过。
    """
    env = ".venv-cuda"
    spec = {}

    def __init__(self, **_):
        sp = self.spec
        path = resolve_path(sp.get("path", f"./models/{self.key}"))
        if not os.path.isdir(path):  # 先检查模型，再导入依赖：没下载时给出更直接的提示
            hint = f"：{sp['download']}" if sp.get("download") else "（Linux 上可用 deploy/setup_cuda.sh）"
            raise SystemExit(f"找不到模型目录 {path}，先下载模型{hint}")
        import diffusers
        import torch
        self.torch = torch
        self.device = torch_device()
        if not hasattr(diffusers, sp.get("pipeline", "")):
            raise SystemExit(f"当前 diffusers 里没有 {sp.get('pipeline')}，检查 models.json 或升级 diffusers")
        t0 = time.time()
        self.pipe = getattr(diffusers, sp["pipeline"]).from_pretrained(path, torch_dtype=torch.bfloat16)
        vram = torch.cuda.mem_get_info()[1] / 1e9 if self.device == "cuda" else 0  # 显存总量
        if self.device == "cuda" and vram < sp.get("offload_below_gb", 20):
            self.pipe.enable_model_cpu_offload()   # 显存不够：按组件在显存和内存之间搬运
        else:
            self.pipe.to(self.device)
        self.out_dir = f"./outputs/{self.key}"
        print(f"  {self.title} 加载完成（{self.device}）{time.time() - t0:.0f}s")

    def generate(self, prompt, neg, w, h, steps, seed, cfg, init=None, strength=None):
        torch = self.torch
        guidance = cfg if cfg is not None else self.spec.get("guidance", 0.0)
        kw = dict(prompt=prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=guidance,
                  generator=torch.Generator("cpu").manual_seed(seed))
        if self.supports_neg and neg:
            kw["negative_prompt"] = neg
        if init:
            name = self.spec.get("img2img_pipeline")
            if not name:
                raise RuntimeError(f"{self.title} 没有配置 img2img_pipeline，不支持 /refine")
            import diffusers
            from PIL import Image
            pipe = getattr(diffusers, name).from_pipe(self.pipe)
            img = pipe(image=Image.open(init).convert("RGB"), strength=strength, **kw).images[0]
            return img, max(1, int(steps * strength))
        return self.pipe(**kw).images[0], steps

    def cleanup(self):
        torch_empty_cache(self.torch)


# ---------------- 按配置生成后端 ----------------

ENGINES = {"mflux": MfluxBackend, "diffusers": DiffusersBackend, "longcat": LongCatBackend}
HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORM = "darwin" if platform.system() == "Darwin" else "linux"


def resolve_path(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def load_models():
    """读取 models.json + models.local.json（同名模型按字段覆盖），并套用当前平台（darwin/linux）的配置。"""
    import json
    merged = {}
    for name in ("models.json", "models.local.json"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for key, spec in json.load(f).items():
                if key.startswith("_") or not isinstance(spec, dict):
                    continue
                merged.setdefault(key, {}).update(spec)
    out = {}
    for key, spec in merged.items():
        spec = {**{k: v for k, v in spec.items() if k not in ("darwin", "linux")}, **(spec.get(PLATFORM) or {})}
        if spec.get("enabled") is False or not spec.get("engine"):
            continue  # 当前平台没有配置引擎，或被禁用
        out[key] = spec
    return out


def make_backend(key, spec):
    base = ENGINES.get(spec["engine"])
    if base is None:
        raise SystemExit(f"models.json 里 {key} 的 engine“{spec['engine']}”不支持，可选：{'、'.join(ENGINES)}")
    steps = spec.get("steps", base.final_steps)
    attrs = dict(key=key, spec=spec, title=spec.get("title", key), env=spec.get("env", base.env),
                 final_steps=steps, preview_steps=spec.get("preview_steps", steps),
                 default_size=spec.get("size", base.default_size), cfg=spec.get("cfg"),
                 supports_neg=spec.get("supports_neg", False), supports_refs=bool(spec.get("refs_class")))
    if spec.get("out_dir"):
        attrs["out_dir"] = spec["out_dir"]
    return type(f"{base.__name__}_{key}", (base,), attrs)


BACKENDS = {key: make_backend(key, spec) for key, spec in load_models().items()}
