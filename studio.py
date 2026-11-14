"""LongCat-Image 常驻交互模式：模型只加载一次，反复改提示词出图。

内存方案（32GB Mac）：文本编码器量化为 int8（~16.6GB -> ~8.5GB），与 bf16 transformer（~12.5GB）
同时常驻 MPS，合计约 21GB。24GB Mac 可加 --quant-transformer 把 transformer 也量化（未实测，可能变慢）。

启动：
  .venv/bin/python studio.py [--size 768x768] [--preview 8] [--final 25] [--seed 43]
          [--ratio 竖图] [--style 人像] [--shot 半身] [--light 窗光] [--tone 温馨] [--time 黄昏] [--neg 人像]

交互命令（直接输入文字 = 用预览步数出图）：
  /final [步数]     用上一条提示词、同种子同尺寸出正式图（默认 --final 步数）
  /refine [强度] [步数] [文件名.png]  以当前种子的预览图（或指定图片）为底稿精修（默认 0.5，步数默认 --final）；
                   保留预览的构图和结构，适合手部等易变形的画面
  /batch N         上一条提示词连换 N 个种子各出一张预览（seed, seed+1, ...）
  /seed N | /seed +   设种子 / 种子 +1
  /size WxH        设尺寸（预览和正式图要一致，否则构图不同）
  /steps N         设预览步数
  /cfg X           设 guidance_scale（默认 4.0）
  /ratio [名称]     画幅预设（方图/竖图/壁纸/横图/宽屏/小方图），会改尺寸
  /shot  [名称|off] 镜头构图      ┐
  /light [名称|off] 光线          │ 不带参数 = 列出可选项
  /tone  [名称|off] 色调氛围      │ 选中后自动追加到提示词后
  /time  [名称|off] 时间天气      │ off = 取消
  /detail [名称|off] 细节质感     │ 可多选，再选一次 = 去掉该项
  /style [名称|off] 风格          ┘
  /neg [预设名|文本|off]  负面提示词：预设（通用/人像/手部/风景/插画）或自定义文本；不带参数列出预设
  /reset           清空所有追加预设和负面提示词
  /open on|off     出图后是否自动用"预览"打开
  /show            显示当前设置
  /quit
"""
import argparse
import copy
import gc
import os
import subprocess
import time

import sys

try:
    import readline  # noqa: F401  没有 prompt_toolkit 时的退路：方向键和历史
except ImportError:
    pass

import numpy as np
import torch
from diffusers import LongCatImagePipeline
from diffusers.pipelines.longcat_image.pipeline_longcat_image import calculate_shift, prepare_pos_ids
from PIL import Image
from optimum.quanto import freeze, qint8, quantize
from transformers import Qwen2_5_VLForConditionalGeneration

from presets import GROUPS, MULTI, NEGATIVES, RATIOS

COMMANDS = {
    "/final": "用上一条提示词出正式图（重新生成，结构可能变）",
    "/refine": "以当前种子的预览图为底稿精修（保留构图和手指结构）",
    "/batch": "换 N 个种子各出一张预览",
    "/seed": "设种子（+ 表示加 1）",
    "/ratio": "画幅预设",
    "/shot": "镜头构图预设",
    "/light": "光线预设",
    "/tone": "色调氛围预设",
    "/time": "时间天气预设",
    "/detail": "细节质感预设（可多选）",
    "/style": "风格预设",
    "/neg": "负面提示词",
    "/reset": "清空所有预设",
    "/size": "自定义尺寸 WxH",
    "/steps": "预览步数",
    "/cfg": "guidance_scale",
    "/open": "出图后自动打开 on/off",
    "/show": "查看设置和实际提示词",
    "/help": "帮助",
    "/quit": "退出",
}


# 数值类命令的常用值提示（也可以直接输入其他数字）
NUMERIC = {
    "/final": [("25", "一般画面，默认"), ("30", "手部、毛发等细节多的画面"), ("20", "较快，细节略少"),
               ("50", "官方默认，比 30 提升很小，耗时近翻倍")],
    "/steps": [("8", "预览默认，看构图和内容"), ("6", "更快，较糊"), ("12", "预览更清楚一些")],
    "/refine": [("0.5", "默认：保留结构，补细节"), ("0.4", "更忠于预览图，改动最小"),
                ("0.6", "细节更多，结构可能小改"), ("0.7", "改动较大")],
    "/batch": [("4", "一次抽 4 个种子"), ("2", "快速对比"), ("8", "多抽几个挑选，耗时较长")],
    "/seed": [("+", "换下一个种子"), ("43", "默认种子")],
    "/cfg": [("4", "默认，推荐"), ("3", "更自由，贴合度略低"), ("5", "更贴合提示词"),
             ("6", "过高会发硬、过饱和"), ("1", "关闭 CFG，快一倍但质量明显下降")],
    "/size": [("1024x1024", "约 100 万像素，模型最擅长，细节最好"), ("1344x768", "官方示例横图"),
              ("768x1344", "竖图高清"), ("768x768", "方图"), ("512x512", "最快，试提示词用")],
}


def arg_choices(cmd):
    """命令参数的候选项：[(值, 说明)]"""
    for key, (c, _, table) in GROUPS.items():
        if c == cmd:
            return list(table.items()) + [("off", "取消")]
    if cmd == "/ratio":
        return [(n, f"{w}x{h}") for n, (w, h) in RATIOS.items()]
    if cmd == "/neg":
        return list(NEGATIVES.items()) + [("off", "清除")]
    if cmd == "/open":
        return [("on", "自动打开"), ("off", "不打开")]
    return NUMERIC.get(cmd, [])


def make_prompt_session():
    """有 TTY 且装了 prompt_toolkit 时返回带补全菜单的输入函数，否则返回 None。"""
    if not sys.stdin.isatty():
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return None

    class StudioCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            if " " not in text:  # 补命令
                for cmd, desc in COMMANDS.items():
                    if cmd.startswith(text):
                        yield Completion(cmd + " ", start_position=-len(text), display=cmd, display_meta=desc)
                return
            cmd, _, arg = text.partition(" ")
            for value, desc in arg_choices(cmd):
                if value.startswith(arg.strip()):
                    yield Completion(value, start_position=-len(arg), display_meta=desc)

    from prompt_toolkit.filters import completion_is_selected
    from prompt_toolkit.application import get_app
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    from prompt_toolkit.filters import Condition, has_completions

    @Condition
    def partial_command():
        text = get_app().current_buffer.text
        return text.startswith("/") and " " not in text and text not in COMMANDS

    @kb.add("enter", filter=has_completions & ~completion_is_selected & partial_command)
    def _(event):
        # 输入 "/ne" 回车：补成第一个匹配的命令（/neg ），不提交半截命令
        buf = event.current_buffer
        buf.apply_completion(buf.complete_state.completions[0])

    @kb.add("enter", filter=completion_is_selected)
    def _(event):
        # 菜单里选中的是命令（以空格结尾）：只填入，等参数；选中的是参数：填入并执行
        buf = event.current_buffer
        completion = buf.complete_state.current_completion
        buf.apply_completion(completion)
        if not completion.text.endswith(" "):
            buf.validate_and_handle()

    os.makedirs(OUT_DIR, exist_ok=True)
    session = PromptSession(
        completer=StudioCompleter(),
        complete_while_typing=True,
        key_bindings=kb,
        history=FileHistory(os.path.join(OUT_DIR, ".history")),
    )

    # 选中 "/style " 这类命令后，立刻弹出它的参数菜单
    def on_change(buf):
        cmd, sep, arg = buf.text.partition(" ")
        if sep and not arg and arg_choices(cmd) and not buf.complete_state:
            buf.start_completion(select_first=False)

    session.default_buffer.on_text_changed += on_change
    return lambda: session.prompt("\nlongcat> ")


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
        self.w, self.h = RATIOS[a.ratio] if a.ratio else map(int, a.size.lower().split("x"))
        self.preview_steps = a.preview
        self.final_steps = a.final
        self.seed = a.seed
        self.cfg = 4.0
        self.neg = NEGATIVES.get(a.neg, a.neg or "")
        self.auto_open = True
        # 各分组当前选中的预设名；多选分组是列表
        self.sel = {key: (list(getattr(a, key, None) or []) if key in MULTI else getattr(a, key, None))
                    for key in GROUPS}
        self.last_prompt = None
        self.cache = {}
        self.done = {}  # (实际提示词, 负面, 宽, 高, 步数, 种子, cfg) -> 图片路径
        os.makedirs(OUT_DIR, exist_ok=True)
        self.idx = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
        self.load_history()

    def load_history(self):
        """从 outputs/log.txt 恢复已生成图片的参数，重启后 /refine 和去重仍然可用。"""
        import ast
        import re
        log = os.path.join(OUT_DIR, "log.txt")
        if not os.path.exists(log):
            return
        pat = re.compile(r"^\d+_s(\d+)_(\d+)st_(\d+)x(\d+)(?:_refine([\d.]+))?\.png$")
        records = []  # 早期记录里提示词带换行会断成多行：没有 tab 的行接到上一条后面
        for line in open(log, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4 and parts[1].startswith("cfg=") and parts[2].startswith("neg="):
                records.append(parts)
            elif records and "\t" not in line:
                records[-1][3] += " " + line.rstrip("\n")  # 换行在 key 里等价于空格
        for name, cfg, neg, prompt in records:
            m = pat.match(name)
            path = os.path.join(OUT_DIR, name)
            if not m or not os.path.exists(path):
                continue
            seed, steps, w, h, strength = m.groups()
            try:
                k = (" ".join(prompt.split()), ast.literal_eval(neg[4:]), int(w), int(h), int(steps), int(seed),
                     float(cfg[4:]))
            except (ValueError, SyntaxError):
                continue
            self.done[k + ((float(strength),) if strength else ())] = path

    def embeds(self, text):
        if text not in self.cache:
            with torch.no_grad():
                self.cache[text] = self.pipe._encode_prompt([text])
        return self.cache[text]

    def compose(self, prompt):
        parts = [prompt]
        for key, (_, _, table) in GROUPS.items():
            names = self.sel[key] if key in MULTI else [self.sel[key]] if self.sel[key] else []
            parts += [table[n] for n in names]
        return "，".join(parts)

    def key(self, prompt, steps, seed):
        return (" ".join(self.compose(prompt).split()), self.neg, self.w, self.h, steps, seed, self.cfg)

    def generate(self, prompt, steps, seed=None, refine=None):
        """refine=(源图路径, 强度)：以源图为底稿精修，只重画后半段去噪，保留构图和结构。"""
        seed = self.seed if seed is None else seed
        k = self.key(prompt, steps, seed) + ((refine[1],) if refine else ())
        if k in self.done:
            print(f"  参数和种子都没变，结果会一模一样，直接打开已有的：{self.done[k]}")
            print("  想要不同的图：/seed + 换种子，或 /batch N")
            if self.auto_open:
                subprocess.run(["open", self.done[k]])
            return
        prompt = self.compose(prompt)
        extra, tag = {}, ""
        if refine:
            extra = self.refine_inputs(refine[0], steps, refine[1], seed)
            tag = f"_refine{refine[1]:g}"
            print(f"  精修中：以 {refine[0]} 为底稿，强度 {refine[1]:g}，"
                  f"实际只跑 {len(extra['sigmas'])}/{steps} 步（算完才有输出，请稍候）...", flush=True)
        else:
            print(f"  生成中：{self.w}x{self.h}，{steps} 步，seed {seed}（算完才有输出，请稍候）...", flush=True)
        t0 = time.time()
        img = self.pipe(
            **extra,
            prompt=None,
            prompt_embeds=self.embeds(prompt),
            negative_prompt_embeds=self.embeds(self.neg),
            height=self.h,
            width=self.w,
            guidance_scale=self.cfg,
            num_inference_steps=len(extra["sigmas"]) if refine else steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            enable_cfg_renorm=True,
            enable_prompt_rewrite=False,
        ).images[0]
        self.idx += 1
        path = os.path.join(OUT_DIR, f"{self.idx:03d}_s{seed}_{steps}st_{self.w}x{self.h}{tag}.png")
        img.save(path)
        self.done[k] = path
        with open(os.path.join(OUT_DIR, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(path)}\tcfg={self.cfg}\tneg={self.neg!r}\t{' '.join(prompt.split())}\n")
        print(f"  -> {path}  ({time.time() - t0:.0f}s)")
        if self.auto_open:
            subprocess.run(["open", path])
        torch.mps.empty_cache()

    @torch.no_grad()
    def refine_inputs(self, src, steps, strength, seed):
        """把源图编码成潜变量，按强度加噪，返回 pipeline 的 latents 和截断后的 sigmas。"""
        pipe = self.pipe
        full = np.linspace(1.0, 1.0 / steps, steps)  # 与 pipeline 默认一致
        sigmas = full[full <= strength + 1e-6]
        if len(sigmas) == 0:
            sigmas = full[-1:]
        # pipeline 会对 sigmas 做动态 shift，起始噪声强度要按 shift 后的值算
        seq_len = (self.h // 16) * (self.w // 16)
        cfg = pipe.scheduler.config
        mu = calculate_shift(seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        sched = copy.deepcopy(pipe.scheduler)
        sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        s0 = float(sched.sigmas[0])

        img = Image.open(src).convert("RGB")
        x = pipe.image_processor.preprocess(img, self.h, self.w).to(DEVICE, pipe.vae.dtype)
        lat = pipe.vae.encode(x).latent_dist.mode()
        lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        noise = torch.randn(lat.shape, generator=torch.Generator("cpu").manual_seed(seed)).to(DEVICE, lat.dtype)
        lat = (1 - s0) * lat + s0 * noise  # flow matching：x_t = (1-σ)·x0 + σ·ε
        b, c, h, w = lat.shape
        packed = pipe._pack_latents(lat, b, c, h, w)
        return {"latents": packed.to(DTYPE), "sigmas": list(sigmas)}

    def show(self):
        print(f"  尺寸 {self.w}x{self.h} | 预览 {self.preview_steps} 步 | 正式 {self.final_steps} 步 | "
              f"seed {self.seed} | cfg {self.cfg} | 自动打开 {self.auto_open}")
        def fmt(key):
            v = self.sel[key]
            return ("+".join(v) if key in MULTI else v) or "-"

        print("  " + " | ".join(f"{label} {fmt(key)}" for key, (_, label, _) in GROUPS.items()))
        print(f"  负面提示词：{self.neg or '-'}")
        if self.last_prompt:
            print(f"  上一条提示词：{self.last_prompt}")
            print(f"  实际提示词：{self.compose(self.last_prompt)}")

    def pick(self, key, arg):
        cmd, label, table = GROUPS[key]
        if not arg:
            for name, desc in table.items():
                chosen = name in self.sel[key] if key in MULTI else name == self.sel[key]
                print(f"  {'*' if chosen else ' '} {name:<6} {desc}")
            if key in MULTI:
                print(f"  可多选：再选一次同一项就是取消该项")
            print(f"  用法：{cmd} {next(iter(table))}   取消：{cmd} off")
        elif arg == "off":
            self.sel[key] = [] if key in MULTI else None
            self.show()
        elif arg in table:
            if key in MULTI:
                self.sel[key] = [n for n in self.sel[key] if n != arg] if arg in self.sel[key] else self.sel[key] + [arg]
            else:
                self.sel[key] = arg
            self.show()
        else:
            print(f"  没有“{arg}”这个{label}，{cmd} 查看列表")

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
        elif cmd == "/refine":
            if self.need_prompt():
                parts = arg.split()
                files = [x for x in parts if x.endswith(".png")]
                nums = [x for x in parts if not x.endswith(".png")]
                strength = float(nums[0]) if nums else 0.5
                steps = int(nums[1]) if len(nums) > 1 else self.final_steps
                if files:  # 直接指定底稿文件：种子和尺寸取自文件名
                    src = files[0] if os.path.exists(files[0]) else os.path.join(OUT_DIR, files[0])
                    import re
                    m = re.search(r"_s(\d+)_\d+st_(\d+)x(\d+)", os.path.basename(src))
                    if not os.path.exists(src):
                        print(f"  找不到文件：{src}")
                        return
                    if m:
                        self.seed, self.w, self.h = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    src = self.done.get(self.key(self.last_prompt, self.preview_steps, self.seed))
                if not src:
                    print(f"  种子 {self.seed} 还没有当前设置下的预览图。先出预览，或直接指定文件："
                          f"/refine 0.5 011_s45_8st_512x512.png")
                else:
                    self.generate(self.last_prompt, steps, refine=(src, strength))
        elif cmd == "/batch":
            if self.need_prompt():
                n, s, seeds = int(arg) if arg else 4, self.seed, []
                while len(seeds) < n:  # 已经出过图的种子跳过，凑够 n 张新的
                    if self.key(self.last_prompt, self.preview_steps, s) not in self.done:
                        seeds.append(s)
                    s += 1
                print(f"  批量预览种子：{', '.join(map(str, seeds))}")
                for s in seeds:
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
            if not arg:
                for name, desc in NEGATIVES.items():
                    print(f"  {'*' if desc == self.neg else ' '} {name:<4} {desc}")
                print("  用法：/neg 人像（预设）  /neg 自定义文字  /neg off（清除）")
            else:
                self.neg = "" if arg == "off" else NEGATIVES.get(arg, arg)
                self.show()
        elif cmd == "/ratio":
            if not arg:
                for name, (w, h) in RATIOS.items():
                    print(f"  {'*' if (w, h) == (self.w, self.h) else ' '} {name:<4} {w}x{h}")
                print("  用法：/ratio 竖图   （预览和正式图要用同一画幅）")
            elif arg in RATIOS:
                self.w, self.h = RATIOS[arg]
                self.show()
            else:
                print(f"  没有“{arg}”这个画幅，/ratio 查看列表")
        elif cmd in {c: k for k, (c, _, _) in GROUPS.items()}:
            self.pick({c: k for k, (c, _, _) in GROUPS.items()}[cmd], arg)
        elif cmd == "/reset":
            self.sel = {key: [] if key in MULTI else None for key in GROUPS}
            self.neg = ""
            self.show()
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
        read = make_prompt_session()
        tip = "输入 / 弹出命令菜单，方向键选择、Tab/回车确认；" if read else ""
        print(f"  直接输入提示词出预览图；{tip}/help 查看命令；/quit 退出")
        read = read or (lambda: input("\nlongcat> "))
        while True:
            try:
                line = read().strip()
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
    ap.add_argument("--ratio", choices=list(RATIOS), help="画幅预设（覆盖 --size）")
    for key, (cmd, label, table) in GROUPS.items():
        ap.add_argument(f"--{key}", choices=list(table), help=f"{label}预设" + ("（可多个）" if key in MULTI else ""),
                        nargs="+" if key in MULTI else None)
    ap.add_argument("--neg", help="负面提示词：预设名（" + "/".join(NEGATIVES) + "）或自定义文本")
    ap.add_argument("--quant-transformer", action="store_true", help="transformer 也量化为 int8（24GB Mac 用）")
    a = ap.parse_args()
    print("加载模型中（只需一次）...")
    Studio(load(a.quant_transformer), a).loop()


if __name__ == "__main__":
    main()
