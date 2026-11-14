"""本地生图常驻交互模式：模型只加载一次，反复改提示词出图。支持多个模型（见 backends.py）。

  --model longcat（默认）  LongCat-Image，用 .venv 运行（PyTorch/MPS）
  --model z-image-turbo    Z-Image-Turbo，用 .venv-mlx 运行（mflux/MLX）
  --model flux2-klein-4b   FLUX.2 klein 4B，用 .venv-mlx 运行（mflux/MLX）

启动：
  .venv/bin/python studio.py [--size 768x768] [--preview 8] [--final 25] [--seed 43]
  .venv-mlx/bin/python studio.py --model z-image-turbo [--quantize 8]
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
  /ai [工具] [文本|图片]  用大模型处理提示词：扩写、英译、精简、负面、检查、看图（不带参数列出工具）；
                   默认处理上一条提示词。工具和服务商可在 ai.local.json 里自定义
  /ai auto 工具|off  每次出图前自动执行某个工具（如英译，给英文模型用）
  /ai provider 名称  切换服务商（longcat、deepseek、qwen、ollama 或自定义）
  /use             采用 /ai 的结果作为新提示词并出预览
  /open on|off     出图后是否自动用"预览"打开
  /show            显示当前设置
  /quit
"""
import argparse
import os
import subprocess
import time

import sys

try:
    import readline  # noqa: F401  没有 prompt_toolkit 时的退路：方向键和历史
except ImportError:
    pass

from ai import AI, AIError, load_config as load_ai_config
from backends import BACKENDS
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
    "/ai": "用大模型处理提示词（扩写、英译、看图反推…，可自定义）",
    "/use": "采用 /ai 的结果作为提示词并出预览",
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


ACTIVE = None  # 当前模型后端，main() 里设置；用于按模型给出步数提示


def arg_choices(cmd):
    """命令参数的候选项：[(值, 说明)]"""
    if cmd in ("/final", "/steps") and ACTIVE is not None and ACTIVE.cfg is None:
        n = ACTIVE.final_steps  # 蒸馏模型：步数少，按模型默认值给提示
        return [(str(n), "模型默认，推荐"), (str(n + max(1, n // 2)), "多几步，细节可能略好，更慢"),
                (str(max(2, n - n // 3)), "少几步，更快，可能更糊")]
    for key, (c, _, table) in GROUPS.items():
        if c == cmd:
            return list(table.items()) + [("off", "取消")]
    if cmd == "/ratio":
        return [(n, f"{w}x{h}") for n, (w, h) in RATIOS.items()]
    if cmd == "/neg":
        return list(NEGATIVES.items()) + [("off", "清除")]
    if cmd == "/open":
        return [("on", "自动打开"), ("off", "不打开")]
    if cmd == "/ai":
        tools = load_ai_config()["tools"]
        return [(n, t.get("desc", "")) for n, t in tools.items()] + [
            ("auto", "每次出图前自动执行某个工具，如 /ai auto 英译"), ("provider", "切换大模型服务商")]
    return NUMERIC.get(cmd, [])


def make_prompt_session(label="longcat"):
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
    return lambda: session.prompt(f"\n{label}> ")


OUT_DIR = "./outputs"  # main() 里按模型改写


class Studio:
    def __init__(self, backend, a):
        self.backend = backend
        size = a.size or backend.default_size
        self.w, self.h = RATIOS[a.ratio] if a.ratio else map(int, size.lower().split("x"))
        self.preview_steps = a.preview or backend.preview_steps
        self.final_steps = a.final or backend.final_steps
        self.seed = a.seed
        self.cfg = backend.cfg
        self.neg = NEGATIVES.get(a.neg, a.neg or "")
        self.auto_open = True
        # 各分组当前选中的预设名；多选分组是列表
        self.sel = {key: (list(getattr(a, key, None) or []) if key in MULTI else getattr(a, key, None))
                    for key in GROUPS}
        self.last_prompt = None
        self.ai = AI(a.ai_provider)
        self.ai_auto = a.ai_auto
        self.ai_candidate = None
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
                neg = ast.literal_eval(neg[4:]) if self.backend.supports_neg else ""
                cfg = None if cfg[4:] == "None" else float(cfg[4:])
                k = (" ".join(prompt.split()), neg, int(w), int(h), int(steps), int(seed), cfg)
            except (ValueError, SyntaxError):
                continue
            self.done[k + ((float(strength),) if strength else ())] = path

    def compose(self, prompt):
        parts = [prompt]
        for key, (_, _, table) in GROUPS.items():
            names = self.sel[key] if key in MULTI else [self.sel[key]] if self.sel[key] else []
            parts += [table[n] for n in names]
        return "，".join(parts)

    def key(self, prompt, steps, seed):
        neg = self.neg if self.backend.supports_neg else ""  # 不支持负面提示词的模型，它不影响出图
        return (" ".join(self.compose(prompt).split()), neg, self.w, self.h, steps, seed, self.cfg)

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
        if self.ai_auto:
            print(f"  AI {self.ai_auto}中...", flush=True)
            prompt = self.ai.run(self.ai_auto, prompt)
            print(f"  AI {self.ai_auto}结果：{prompt}")
        tag = f"_refine{refine[1]:g}" if refine else ""
        if refine:
            print(f"  精修中：以 {refine[0]} 为底稿，强度 {refine[1]:g}，{steps} 步的后半程"
                  f"（算完才有输出，请稍候）...", flush=True)
        else:
            print(f"  生成中：{self.w}x{self.h}，{steps} 步，seed {seed}（算完才有输出，请稍候）...", flush=True)
        t0 = time.time()
        img, ran = self.backend.generate(prompt, self.neg, self.w, self.h, steps, seed, self.cfg,
                                         init=refine[0] if refine else None,
                                         strength=refine[1] if refine else None)
        self.idx += 1
        path = os.path.join(OUT_DIR, f"{self.idx:03d}_s{seed}_{steps}st_{self.w}x{self.h}{tag}.png")
        img.save(path)
        self.done[k] = path
        with open(os.path.join(OUT_DIR, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(path)}\tcfg={self.cfg}\tneg={self.neg!r}\t{' '.join(prompt.split())}\n")
        note = f"，实际 {ran} 步" if refine else ""
        print(f"  -> {path}  ({time.time() - t0:.0f}s{note})")
        if self.auto_open:
            subprocess.run(["open", path])
        self.backend.cleanup()

    def show(self):
        cfg = self.cfg if self.cfg is not None else "不支持"
        print(f"  模型 {self.backend.title}")
        print(f"  尺寸 {self.w}x{self.h} | 预览 {self.preview_steps} 步 | 正式 {self.final_steps} 步 | "
              f"seed {self.seed} | cfg {cfg} | 自动打开 {self.auto_open}")
        def fmt(key):
            v = self.sel[key]
            return ("+".join(v) if key in MULTI else v) or "-"

        print("  " + " | ".join(f"{label} {fmt(key)}" for key, (_, label, _) in GROUPS.items()))
        print(f"  负面提示词：{self.neg or '-'}" + ("" if self.backend.supports_neg else "（当前模型不支持，不生效）"))
        try:
            name, ok, env = self.ai.status()
            state = "已配置密钥" if ok else f"未设置 {env}"
            print(f"  AI：{name}（{state}）| 自动执行：{self.ai_auto or '-'}")
        except AIError as e:
            print(f"  AI：{e}")
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

    def ai_command(self, arg):
        name, _, rest = arg.partition(" ")
        rest = rest.strip()
        if not name:
            for n, t in self.ai.tools.items():
                where = {"prompt": "→ 新提示词", "neg": "→ 负面提示词", "show": "→ 只显示"}.get(t.get("target"), "")
                print(f"  {n:<4} {t.get('desc', '')} {where}")
            print("  用法：/ai 扩写（处理上一条提示词）  /ai 扩写 一只猫  /ai 看图 图片.png  /ai auto 英译")
            self.show()
            return
        if name == "auto":
            if rest in ("", "off"):
                self.ai_auto = None
            elif rest in self.ai.tools and self.ai.tools[rest].get("target") == "prompt" \
                    and self.ai.tools[rest].get("input") != "image":
                self.ai_auto = rest
            else:
                print(f"  “{rest}”不能自动执行（只有输出提示词的文本工具可以）")
                return
            self.show()
            return
        if name == "provider":
            if rest not in self.ai.providers:
                print(f"  可选服务商：{', '.join(self.ai.providers)}（在 ai.local.json 里可以加）")
                return
            self.ai.provider_name = rest
            self.show()
            return

        tool = self.ai.tools.get(name)
        if not tool:
            print(f"  没有“{name}”这个工具，/ai 查看列表")
            return
        image = None
        if tool.get("input") == "image":
            path = rest or ""
            if path and not os.path.exists(path) and os.path.exists(os.path.join(OUT_DIR, path)):
                path = os.path.join(OUT_DIR, path)
            image, text = path, ""
        else:
            text = rest or (self.compose(self.last_prompt) if self.last_prompt else "")
            if not text:
                print("  没有可处理的提示词：先输入一句描述，或 /ai 工具 文本")
                return
        print(f"  AI {name}中（{self.ai.provider_name}）...", flush=True)
        out = self.ai.run(name, text, image)
        target = tool.get("target", "prompt")
        if target == "neg":
            self.neg = out
            if not self.backend.supports_neg:
                print(f"  注意：{self.backend.title} 不支持负面提示词，不会生效")
            self.show()
        elif target == "show":
            print(out)
        else:
            self.ai_candidate = out
            print(f"  结果：{out}")
            print("  输入 /use 用它出预览图（预设会照常追加；可先 /reset 避免重复）")

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
            if self.backend.cfg is None:
                print(f"  {self.backend.title} 是蒸馏模型，不支持调 CFG")
            else:
                self.cfg = float(arg)
                self.show()
        elif cmd == "/neg":
            if not arg:
                for name, desc in NEGATIVES.items():
                    print(f"  {'*' if desc == self.neg else ' '} {name:<4} {desc}")
                print("  用法：/neg 人像（预设）  /neg 自定义文字  /neg off（清除）")
            else:
                self.neg = "" if arg == "off" else NEGATIVES.get(arg, arg)
                if not self.backend.supports_neg:
                    print(f"  注意：{self.backend.title} 不支持负面提示词，设置了也不会生效")
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
        elif cmd == "/ai":
            self.ai_command(arg)
        elif cmd == "/use":
            if not self.ai_candidate:
                print("  还没有可用的 AI 结果，先用 /ai 扩写（或其他工具）")
            else:
                self.last_prompt, self.ai_candidate = self.ai_candidate, None
                self.generate(self.last_prompt, self.preview_steps)
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
        read = make_prompt_session(self.backend.key)
        tip = "输入 / 弹出命令菜单，方向键选择、Tab/回车确认；" if read else ""
        print(f"  直接输入提示词出预览图；{tip}/help 查看命令；/quit 退出")
        read = read or (lambda: input(f"\n{self.backend.key}> "))
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
    ap = argparse.ArgumentParser(description="本地生图常驻交互模式")
    ap.add_argument("--model", default="longcat", choices=list(BACKENDS), help="模型（默认 longcat）")
    ap.add_argument("--size", help="尺寸 WxH（默认按模型：longcat 768x768，其他 1024x1024）")
    ap.add_argument("--preview", type=int, help="预览步数（默认按模型）")
    ap.add_argument("--final", type=int, help="正式图步数（默认按模型）")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--ratio", choices=list(RATIOS), help="画幅预设（覆盖 --size）")
    for key, (cmd, label, table) in GROUPS.items():
        ap.add_argument(f"--{key}", choices=list(table), help=f"{label}预设" + ("（可多个）" if key in MULTI else ""),
                        nargs="+" if key in MULTI else None)
    ap.add_argument("--neg", help="负面提示词：预设名（" + "/".join(NEGATIVES) + "）或自定义文本")
    ap.add_argument("--quant-transformer", action="store_true", help="longcat：transformer 也量化为 int8（24GB Mac 用）")
    ap.add_argument("--ai-provider", help="AI 提示词工具的服务商（默认见 ai.json）")
    ap.add_argument("--ai-auto", help="每次出图前自动执行的 AI 工具，如 英译")
    ap.add_argument("--quantize", type=int, choices=[3, 4, 5, 6, 8], help="mflux 模型的量化位数（默认 8）")
    a = ap.parse_args()

    global OUT_DIR
    cls = BACKENDS[a.model]
    print(f"加载 {cls.title} 中（只需一次）...")
    try:
        backend = cls(quant_transformer=a.quant_transformer, quantize=a.quantize)
    except ImportError as e:
        raise SystemExit(f"缺少依赖（{e.name}）：{a.model} 要用 {cls.env} 运行，例如 {cls.env}/bin/python studio.py --model {a.model}")
    OUT_DIR = backend.out_dir
    global ACTIVE
    ACTIVE = backend
    Studio(backend, a).loop()


if __name__ == "__main__":
    main()
