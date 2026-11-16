"""提示词工具：清理其他工具的专用语法；解析批量提示词文本。studio.py 和 video.py 共用。"""
import re

# Midjourney 参数：--ar 16:9、--v 6、--stylize 250、--no xxx、--style raw、--tile 等
MJ_PARAM = re.compile(r"\s--[a-zA-Z]+(?:\s+(?!--)[^\s,，]+)?")
LORA_TAG = re.compile(r"<(?:lora|lyco|hypernet|embedding):[^>]*>", re.I)
WEIGHT = re.compile(r"\(([^()]+?):\s*[0-9.]+\)")            # (word:1.2) -> word
EMPHASIS = re.compile(r"\({1,}([^()]+?)\){1,}")               # ((word)) -> word
DEEMPH = re.compile(r"\[([^\[\]|:]+?)\]")                     # [word] -> word（WebUI 减弱权重）
BREAK = re.compile(r"\bBREAK\b")


def clean_prompt(text):
    """去掉 Midjourney / Stable Diffusion WebUI 等工具的专用语法，返回 (清理后的文本, 去掉了什么)。

    这些语法本工具的模型不认识，会被当成普通文字画进画面或干扰理解。
    中文括号、引号（图中文字）不受影响。
    """
    removed = []
    t = " " + text.strip()
    for m in MJ_PARAM.findall(t):
        removed.append(m.strip())
    t = MJ_PARAM.sub("", t)
    for m in LORA_TAG.findall(t):
        removed.append(m)
    t = LORA_TAG.sub("", t)
    if WEIGHT.search(t):
        removed.append("权重写法 (词:数字)")
        t = WEIGHT.sub(r"\1", t)
    if EMPHASIS.search(t):
        removed.append("强调括号 ((词))")
        t = EMPHASIS.sub(r"\1", t)
    if DEEMPH.search(t):
        removed.append("减弱括号 [词]")
        t = DEEMPH.sub(r"\1", t)
    if BREAK.search(t):
        removed.append("BREAK")
        t = BREAK.sub(",", t)
    t = re.sub(r"\s*,\s*(,\s*)+", ", ", t)            # 合并多余逗号
    t = re.sub(r"\s+([,，])", r"\1", t)                 # 去掉逗号前的空格
    t = re.sub(r"[ \t]{2,}", " ", t).strip(" ,，")
    return t, removed


ID_PATTERN = re.compile(r"^[A-Za-z]{0,4}\d{1,4}[A-Za-z]?$")


def parse_shot_lines(text):
    """解析批量提示词文本。每行一个镜头，用 | 分隔：

        镜号 | 画面（出关键帧） | 动作（出视频） | 运镜 | 其他参数 key=value ...

    - 镜号可以省略（第一列不像 S01、12 这样的编号时），按顺序自动编号 S01、S02……
    - 后面几列可以省略；空行和 # 开头的行忽略；全角竖线 ｜ 也可以
    - 其他参数例：seconds=3 mode=still line=就是现在！ speaker=hero seeds=42,43
    返回 [dict]，每个 dict 带 _line（原始行号）。
    """
    shots, n = [], 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in re.split(r"[|｜]", line)]
        n += 1
        if cols[0] == "":                      # 以 | 开头：镜号空着，自动编号
            sid, cols = f"S{n:02d}", cols[1:]
        elif ID_PATTERN.match(cols[0]):
            sid, cols = cols[0].upper(), cols[1:]
            if sid.isdigit():
                sid = f"S{int(sid):02d}"
        else:
            sid = f"S{n:02d}"
        if sid.isdigit():
            sid = f"S{int(sid):02d}"
        shot = {"id": sid, "_line": lineno}
        keys = ["keyframe", "prompt", "camera"]
        for key, val in zip(keys, cols[:3]):
            if val:
                shot[key] = val
        for extra in cols[3:]:
            for kv in re.findall(r"(\w+)=([^\s]+)", extra):
                k, v = kv
                if k == "seeds":
                    shot[k] = [int(x) for x in v.split(",") if x.strip().isdigit()]
                elif re.fullmatch(r"-?\d+(\.\d+)?", v):
                    shot[k] = float(v) if "." in v else int(v)
                elif v.lower() in ("true", "false"):
                    shot[k] = v.lower() == "true"
                else:
                    shot[k] = v
        shots.append(shot)
    return shots
