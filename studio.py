"""LongCat-Image 常驻交互模式：模型只加载一次，反复改提示词出图。

内存方案（32GB Mac）：文本编码器量化为 int8（~16.6GB -> ~8.5GB），与 bf16 transformer（~12.5GB）
同时常驻 MPS，合计约 21GB。24GB Mac 可加 --quant-transformer 把 transformer 也量化（未实测，可能变慢）。

启动：
  .venv/bin/python studio.py [--size 768x768] [--preview 8] [--final 25] [--seed 43]

交互命令（直接输入文字 = 用预览步数出图）：
  /final [步数]     用上一条提示词、同种子同尺寸出正式图（默认 --final 步数）
  /batch N         上一条提示词连换 N 个种子各出一张预览（seed, seed+1, ...）
  /seed N | /seed +   设种子 / 种子 +1
  /size WxH        设尺寸（预览和正式图要一致，否则构图不同）
  /steps N         设预览步数
  /cfg X           设 guidance_scale（默认 4.0）
  /neg 文本         设负面提示词（空 = 清除）
  /style [名称|off] 查看风格列表 / 选风格（追加到提示词后）/ 取消
  /open on|off     出图后是否自动用"预览"打开
  /show            显示当前设置
  /quit
"""
import argparse
import gc
import os
import subprocess
import time

try:
    import readline  # noqa: F401  让输入支持方向键和历史
except ImportError:
    pass

import torch
from diffusers import LongCatImagePipeline
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids
from optimum.quanto import freeze, qint8, quantize
from transformers import Qwen2_5_VLForConditionalGeneration

STYLES = {
    "写实": "写实摄影风格，自然光线，细节清晰，高清照片质感",
    "电影": "电影剧照风格，电影级布光，浅景深，宽银幕构图，色彩分级",
    "胶片": "胶片摄影风格，柔和颗粒感，复古色调，轻微漏光",
    "人像": "专业人像摄影，柔和侧光，背景虚化，皮肤质感细腻",
    "动漫": "日系动漫风格，赛璐璐上色，线条干净，色彩明快",
    "治愈动画": "手绘动画电影风格，温暖柔和的色彩，水彩质感背景，治愈氛围",
    "3D卡通": "3D 卡通渲染风格，圆润造型，柔和全局光照，色彩饱和",
    "水彩": "水彩画风格，颜料晕染，纸张纹理，留白，清新淡雅",
    "油画": "古典油画风格，厚涂笔触，画布纹理，浓郁色彩",
    "水墨": "中国水墨画风格，墨色浓淡变化，留白意境，宣纸质感",
    "国潮": "国潮插画风格，中国传统纹样，红金配色，扁平装饰感",
    "素描": "铅笔素描风格，黑白灰调，排线明暗，素描纸纹理",
    "扁平插画": "扁平矢量插画风格，简洁几何形状，纯色块，无阴影",
    "像素": "像素艺术风格，8-bit 复古游戏画面，有限调色板",
    "赛博朋克": "赛博朋克风格，霓虹灯光，夜晚雨天街道，蓝紫色调，高科技感",
    "蒸汽波": "蒸汽波美学风格，粉紫渐变，复古未来感，霓虹网格",
}

MODEL = "./model-t2i"
OUT_DIR = "./outputs"
DEVICE = "mps"
DTYPE = torch.bfloat16


def load(quant_transformer: bool) -> LongCatImagePipeline:
    t0 = time.time()
    # 先单独加载并量化文本编码器（此时 transformer 还没进内存，避免峰值叠加）
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"{MODEL}/text_encoder", dtype=DTYPE)
    quantize(te, weights=qint8)
    freeze(te)
    gc.collect()
    te.to(DEVICE)
    print(f"  文本编码器 int8 就绪 {time.time() - t0:.0f}s")

    pipe = LongCatImagePipeline.from_pretrained(MODEL, text_encoder=te, dtype=DTYPE)
    if quant_transformer:
        quantize(pipe.transformer, weights=qint8)
        freeze(pipe.transformer)
        gc.collect()
    pipe.transformer.to(DEVICE)
    pipe.vae.to(DEVICE)

    # 上游 encode_prompt 在 prompt=None 时会 len(None)，这里直接用缓存的 embeds
    def cached_encode(prompt=None, num_images_per_prompt=1, prompt_embeds=None):
        ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1])
        return prompt_embeds.to(DEVICE), ids.to(DEVICE)

    pipe.encode_prompt = cached_encode
    pipe.set_progress_bar_config(disable=True)  # MPS 异步，进度条时间不准，不如不显示
    print(f"  全部加载完成 {time.time() - t0:.0f}s")
    return pipe


class Studio:
    def __init__(self, pipe, a):
        self.pipe = pipe
        self.w, self.h = map(int, a.size.lower().split("x"))
        self.preview_steps = a.preview
        self.final_steps = a.final
        self.seed = a.seed
        self.cfg = 4.0
        self.neg = ""
        self.auto_open = True
        self.style = a.style
        self.last_prompt = None
        self.cache = {}
        os.makedirs(OUT_DIR, exist_ok=True)
        self.idx = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])

    def embeds(self, text):
        if text not in self.cache:
            with torch.no_grad():
                self.cache[text] = self.pipe._encode_prompt([text])
        return self.cache[text]

    def generate(self, prompt, steps, seed=None):
        seed = self.seed if seed is None else seed
        if self.style:
            prompt = f"{prompt}，{STYLES[self.style]}"
        print(f"  生成中：{self.w}x{self.h}，{steps} 步，seed {seed}（算完才有输出，请稍候）...", flush=True)
        t0 = time.time()
        img = self.pipe(
            prompt=None,
            prompt_embeds=self.embeds(prompt),
            negative_prompt_embeds=self.embeds(self.neg),
            height=self.h,
            width=self.w,
            guidance_scale=self.cfg,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            enable_cfg_renorm=True,
            enable_prompt_rewrite=False,
        ).images[0]
        self.idx += 1
        path = os.path.join(OUT_DIR, f"{self.idx:03d}_s{seed}_{steps}st_{self.w}x{self.h}.png")
        img.save(path)
        with open(os.path.join(OUT_DIR, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(path)}\tcfg={self.cfg}\tneg={self.neg!r}\t{prompt}\n")
        print(f"  -> {path}  ({time.time() - t0:.0f}s)")
        if self.auto_open:
            subprocess.run(["open", path])
        torch.mps.empty_cache()

    def show(self):
        print(f"  尺寸 {self.w}x{self.h} | 预览 {self.preview_steps} 步 | 正式 {self.final_steps} 步 | "
              f"seed {self.seed} | cfg {self.cfg} | 风格 {self.style or '无'} | neg {self.neg!r} | "
              f"自动打开 {self.auto_open}")
        if self.last_prompt:
            print(f"  上一条提示词：{self.last_prompt}")

    def need_prompt(self):
        if not self.last_prompt:
            print("  还没有提示词，先直接输入一句描述")
            return False
        return True

    def command(self, line):
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/final":
            if self.need_prompt():
                self.generate(self.last_prompt, int(arg) if arg else self.final_steps)
        elif cmd == "/batch":
            if self.need_prompt():
                for s in range(self.seed, self.seed + (int(arg) if arg else 4)):
                    self.generate(self.last_prompt, self.preview_steps, seed=s)
                print("  挑中哪张，就 /seed 那个数字，再 /final")
        elif cmd == "/seed":
            self.seed = self.seed + 1 if arg == "+" else int(arg)
            self.show()
        elif cmd == "/size":
            self.w, self.h = map(int, arg.lower().split("x"))
            self.show()
        elif cmd == "/steps":
            self.preview_steps = int(arg)
            self.show()
        elif cmd == "/cfg":
            self.cfg = float(arg)
            self.show()
        elif cmd == "/neg":
            self.neg = arg
            self.show()
        elif cmd == "/style":
            if not arg:
                for name, desc in STYLES.items():
                    print(f"  {'*' if name == self.style else ' '} {name:<6} {desc}")
                print("  用法：/style 动漫   取消：/style off")
            elif arg == "off":
                self.style = None
                self.show()
            elif arg in STYLES:
                self.style = arg
                self.show()
            else:
                print(f"  没有“{arg}”这个风格，/style 查看列表")
        elif cmd == "/open":
            self.auto_open = arg != "off"
            self.show()
        elif cmd == "/show":
            self.show()
        elif cmd in ("/help", "/?"):
            print(__doc__.split("交互命令")[1])
        else:
            print("  未知命令，/help 查看")

    def loop(self):
        self.show()
        print("  直接输入提示词出预览图；/help 查看命令；/quit 退出")
        while True:
            try:
                line = input("\nlongcat> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                break
            try:
                if line.startswith("/"):
                    self.command(line)
                else:
                    self.last_prompt = line
                    self.generate(line, self.preview_steps)
            except Exception as e:  # 单次出错不退出常驻进程
                print(f"  出错：{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="LongCat-Image 常驻交互出图")
    ap.add_argument("--size", default="768x768")
    ap.add_argument("--preview", type=int, default=8, help="预览步数")
    ap.add_argument("--final", type=int, default=25, help="正式图步数")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--style", choices=list(STYLES), help="风格预设，会追加到每条提示词后")
    ap.add_argument("--quant-transformer", action="store_true", help="transformer 也量化为 int8（24GB Mac 用）")
    a = ap.parse_args()
    print("加载模型中（只需一次）...")
    Studio(load(a.quant_transformer), a).loop()


if __name__ == "__main__":
    main()
