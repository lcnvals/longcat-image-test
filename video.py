"""本地视频生成（LTX-2.3，MLX）。和 studio.py 共用预设与 AI 提示词工具，单独运行。

视频模型很吃内存（24GB 机器上约 15GB），不和生图模型同时加载：先用 studio 出关键帧，
退出后再用本工具生成视频。LTX 装在自己的虚拟环境里，这里通过子进程调用（路径见 video.json）。

单个镜头：
  .venv-mlx/bin/python video.py gen "橘猫慢慢眨眼，尾巴轻轻摆动" --image cat.png --camera 推进 --seconds 2
  .venv-mlx/bin/python video.py gen "..." --seed 42 43 44          # 一次出多个种子挑选
  .venv-mlx/bin/python video.py gen "..." --image a.png --end-image b.png   # 锁定首尾帧

分镜（批量）：
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json            # 生成全部镜头
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --only S03  # 只重做 S03
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --concat    # 拼接成片
  .venv-mlx/bin/python video.py init projects/猫咖啡店                             # 生成分镜模板

加 --dry-run 只打印将要执行的命令，不真正生成。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from presets import CAMERAS, GROUPS, MOTIONS, MULTI, NEGATIVES, VIDEO_RATIOS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "video")
DEFAULTS = {"size": "768x512", "fps": 25, "seconds": 2.0, "seeds": [42], "image_strength": 1.0,
            "low_ram": True, "audio": False}


# ---------------- 配置与参数 ----------------

def load_cfg():
    cfg = {}
    for name in ("video.json", "video.local.json"):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if isinstance(v, dict):
                        cfg.setdefault(k, {}).update(v)
                    elif not k.startswith("_"):
                        cfg[k] = v
    ltx = cfg.get("ltx", {})
    for k in ("cli", "model", "gemma"):
        if k in ltx and not os.path.isabs(ltx[k]):
            ltx[k] = os.path.normpath(os.path.join(HERE, ltx[k]))
    return cfg


def parse_size(job):
    """--ratio 优先，其次 --size；校验为 64 的倍数。"""
    if job.get("ratio"):
        if job["ratio"] not in VIDEO_RATIOS:
            raise SystemExit(f"没有“{job['ratio']}”这个画幅，可选：{'、'.join(VIDEO_RATIOS)}")
        w, h = VIDEO_RATIOS[job["ratio"]]
    else:
        w, h = map(int, str(job.get("size", DEFAULTS["size"])).lower().split("x"))
    if w % 64 or h % 64:
        fix = f"{round(w / 64) * 64}x{round(h / 64) * 64}"
        raise SystemExit(f"尺寸 {w}x{h} 必须是 64 的倍数，可以改成 {fix}")
    return w, h


def frames_for(job):
    """LTX 要求帧数满足 8n+1；按秒数换算并取最接近的合法值。"""
    if job.get("frames"):
        n = int(job["frames"])
    else:
        n = round(float(job.get("seconds", DEFAULTS["seconds"])) * int(job.get("fps", DEFAULTS["fps"])))
    return max(9, 8 * round((n - 1) / 8) + 1)


def compose(job):
    """用户提示词 + 镜头/光线/色调/时间/细节/风格预设 + 运镜 + 动作幅度。"""
    parts = [job["prompt"]]
    for key, (_, label, table) in GROUPS.items():
        val = job.get(key)
        names = (val if isinstance(val, list) else [val]) if val else []
        for n in names:
            if n not in table:
                raise SystemExit(f"没有“{n}”这个{label}预设，可选：{'、'.join(table)}")
            parts.append(table[n])
    for key, table, label in (("camera", CAMERAS, "运镜"), ("motion", MOTIONS, "动作幅度")):
        if job.get(key):
            if job[key] not in table:
                raise SystemExit(f"没有“{job[key]}”这个{label}，可选：{'、'.join(table)}")
            parts.append(table[job[key]])
    prompt = "，".join(parts)
    if job.get("ai"):
        from ai import AI
        print(f"  AI {job['ai']}中...", flush=True)
        prompt = AI(job.get("ai_provider")).run(job["ai"], prompt)
        print(f"  AI 结果：{prompt}")
    return prompt


def prepare_image(src, w, h, dst_dir):
    """把关键帧居中裁成目标比例并缩放，避免视频首帧被拉伸。"""
    if not src:
        return None
    if not os.path.exists(src):
        raise SystemExit(f"找不到关键帧图片：{src}")
    try:
        from PIL import Image
    except ImportError:
        return src  # 没有 PIL 就原样交给 LTX
    im = Image.open(src).convert("RGB")
    iw, ih = im.size
    if iw * h > ih * w:  # 原图更宽：裁左右
        nw = ih * w // h
        im = im.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    else:                # 原图更高：裁上下
        nh = iw * h // w
        im = im.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, f"key_{os.path.splitext(os.path.basename(src))[0]}_{w}x{h}.png")
    im.resize((w, h), Image.LANCZOS).save(out)
    return out


# ---------------- 生成 ----------------

def ltx_command(cfg, job, prompt, seed, w, h, frames, out):
    ltx = cfg.get("ltx", {})
    for k in ("cli", "model", "gemma"):
        if not os.path.exists(ltx.get(k, "")):
            raise SystemExit(f"LTX 的 {k} 路径不存在：{ltx.get(k)}。请检查 video.json / video.local.json")
    cmd = [ltx["cli"], "generate", "--model", ltx["model"], "--gemma", ltx["gemma"], "--distilled",
           "-H", str(h), "-W", str(w), "-f", str(frames), "--frame-rate", str(job.get("fps", DEFAULTS["fps"])),
           "--seed", str(seed), "-p", prompt, "-o", out]
    if job.get("low_ram", DEFAULTS["low_ram"]):
        cmd.append("--low-ram")
    if not job.get("audio", DEFAULTS["audio"]):
        cmd.append("--no-audio")
    if job.get("enhance"):
        cmd.append("--enhance-prompt")
    if job.get("_image"):
        cmd += ["--image", job["_image"], "0", str(job.get("image_strength", DEFAULTS["image_strength"]))]
    if job.get("_end_image"):
        cmd += ["--image", job["_end_image"], str(frames - 1), "1.0"]
    neg = NEGATIVES.get(job.get("neg"), job.get("neg"))
    if neg:  # 蒸馏流程没有 CFG，负面提示词要通过 NAG 生效
        cmd += ["--negative-prompt", neg, "--nag"]
    return cmd


def run_job(cfg, job, out_dir, name_prefix="", dry=False):
    """生成一个镜头的所有种子，返回 [(种子, 视频路径)]。已存在的同参数视频直接复用。"""
    w, h = parse_size(job)
    frames = frames_for(job)
    fps = int(job.get("fps", DEFAULTS["fps"]))
    os.makedirs(out_dir, exist_ok=True)
    job["_image"] = prepare_image(job.get("image"), w, h, out_dir)
    job["_end_image"] = prepare_image(job.get("end_image"), w, h, out_dir)
    prompt = compose(job)
    seeds = job.get("seeds") or DEFAULTS["seeds"]
    print(f"  {w}x{h}，{frames} 帧 @ {fps}fps（{frames / fps:.2f} 秒），种子 {seeds}")
    print(f"  实际提示词：{prompt}")
    results = []
    for seed in seeds:
        tag = "_i2v" if job["_image"] else ""
        out = os.path.join(out_dir, f"{name_prefix}s{seed}_{frames}f_{w}x{h}{tag}.mp4")
        meta = out[:-4] + ".json"
        record = {"prompt": prompt, "seed": seed, "size": [w, h], "frames": frames, "fps": fps,
                  "image": job.get("image"), "end_image": job.get("end_image"), "neg": job.get("neg"),
                  "image_strength": job.get("image_strength", DEFAULTS["image_strength"])}
        if os.path.exists(out) and os.path.exists(meta) and json.load(open(meta, encoding="utf-8")) == record:
            print(f"  种子 {seed}：参数完全相同的视频已存在，跳过 -> {out}")
            results.append((seed, out))
            continue
        cmd = ltx_command(cfg, job, prompt, seed, w, h, frames, out)
        if dry:
            print("  [dry-run] " + " ".join(f"'{c}'" if " " in c or "，" in c else c for c in cmd))
            continue
        print(f"  生成中：种子 {seed}（约 3 分钟，请稍候）...", flush=True)
        t0 = time.time()
        log = out[:-4] + ".log"
        env = dict(os.environ, HF_HUB_OFFLINE="1")  # 模型都在本地，禁止联网避免卡住
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0 or not os.path.exists(out):
            print(f"  ❌ 种子 {seed} 失败（exit={rc}），日志：{log}")
            continue
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        print(f"  -> {out}  ({time.time() - t0:.0f}s)")
        results.append((seed, out))
    return results


# ---------------- 分镜 ----------------

TEMPLATE = {
    "settings": {
        "_说明": "全片通用设置；镜头里同名字段会覆盖这里。{character}、{scene} 会被替换成下面的设定。",
        "character": "一只胖胖的橘色虎斑猫，绿色眼睛，戴红色项圈",
        "scene": "日式小咖啡店，木质吧台，暖黄吊灯，窗外下着小雨",
        "ratio": "横屏", "fps": 25, "seconds": 2, "style": "电影", "neg": None, "seeds": [42],
    },
    "shots": [
        {"id": "S01", "prompt": "{scene}，雨丝落下", "camera": "推进", "motion": "轻微"},
        {"id": "S02", "prompt": "{character}趴在吧台上，慢慢抬起头", "camera": "固定",
         "image": "S02.png", "_说明": "image 是关键帧（相对分镜文件所在目录），可以用 studio 先出图"},
        {"id": "S03", "prompt": "{character}的脸部特写，慢慢眨眼", "camera": "固定", "seeds": [42, 43],
         "pick": 43, "_说明": "pick：拼接成片时用哪个种子"},
    ],
}


def load_board(path):
    with open(path, encoding="utf-8") as f:
        board = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    settings = {k: v for k, v in board.get("settings", {}).items() if not k.startswith("_")}
    jobs = []
    for shot in board["shots"]:
        job = {k: v for k, v in settings.items() if k not in ("character", "scene")}
        job.update({k: v for k, v in shot.items() if not k.startswith("_")})
        job["prompt"] = job["prompt"].replace("{character}", settings.get("character", "")) \
                                     .replace("{scene}", settings.get("scene", ""))
        for key in ("image", "end_image"):
            if job.get(key) and not os.path.isabs(job[key]):
                job[key] = os.path.join(base, job[key])
        jobs.append(job)
    return base, jobs


def concat(clips, out):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("没有找到 ffmpeg，先 brew install ffmpeg")
    lst = out[:-4] + "_list.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    # 所有镜头来自同一模型、同一尺寸和帧率时可以直接拷贝拼接，不重新编码
    rc = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                         "-c", "copy", out]).returncode
    if rc != 0:
        raise SystemExit("拼接失败：各镜头的尺寸或帧率不一致时需要重新编码")
    print(f"成片 -> {out}")


# ---------------- 命令行 ----------------

def add_job_args(p):
    p.add_argument("--image", help="首帧关键帧图片（图生视频）")
    p.add_argument("--end-image", help="尾帧关键帧图片（锁定结尾画面）")
    p.add_argument("--image-strength", type=float, help="首帧锁定强度 0–1（默认 1.0）")
    p.add_argument("--seed", dest="seeds", type=int, nargs="+", help="一个或多个种子（默认 42）")
    p.add_argument("--seconds", type=float, help="时长秒数（默认 2；LTX 超过约 2 秒成功率下降）")
    p.add_argument("--frames", type=int, help="直接指定帧数（会调整为 8n+1）")
    p.add_argument("--fps", type=int, help="帧率（默认 25；LTX-2.3 训练用 24）")
    p.add_argument("--size", help="尺寸 WxH，64 的倍数（默认 768x512）")
    p.add_argument("--ratio", choices=list(VIDEO_RATIOS), help="视频画幅预设")
    p.add_argument("--camera", choices=list(CAMERAS), help="运镜")
    p.add_argument("--motion", choices=list(MOTIONS), help="动作幅度")
    for key, (_, label, table) in GROUPS.items():
        p.add_argument(f"--{key}", choices=list(table), nargs="+" if key in MULTI else None, help=f"{label}预设")
    p.add_argument("--neg", help="负面提示词：预设名（" + "/".join(NEGATIVES) + "）或自定义文本")
    p.add_argument("--ai", help="生成前先用 AI 工具处理提示词，如 扩写、英译")
    p.add_argument("--ai-provider", help="AI 服务商")
    p.add_argument("--enhance", action="store_true", help="用 LTX 自带的 Gemma 扩写提示词")
    p.add_argument("--audio", action="store_true", help="同时生成音频（默认关闭）")
    p.add_argument("--no-low-ram", dest="low_ram", action="store_false", default=None,
                   help="关闭低内存模式（内存充足时更快）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不生成")


def main():
    ap = argparse.ArgumentParser(description="本地视频生成（LTX-2.3）", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[-1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="生成单个镜头")
    g.add_argument("prompt")
    add_job_args(g)
    b = sub.add_parser("board", help="按分镜文件批量生成")
    b.add_argument("storyboard")
    b.add_argument("--only", nargs="+", help="只生成这些镜头，如 S02 S05")
    b.add_argument("--concat", action="store_true", help="生成后按镜头顺序拼接成片（用每个镜头的 pick 种子）")
    b.add_argument("--dry-run", action="store_true")
    i = sub.add_parser("init", help="在目录里生成分镜模板")
    i.add_argument("project_dir")
    a = ap.parse_args()

    if a.cmd == "init":
        os.makedirs(a.project_dir, exist_ok=True)
        path = os.path.join(a.project_dir, "storyboard.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} 已存在，不覆盖")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(TEMPLATE, f, ensure_ascii=False, indent=2)
        print(f"分镜模板 -> {path}")
        return

    cfg = load_cfg()
    if a.cmd == "gen":
        job = {k: v for k, v in vars(a).items() if v is not None and k not in ("cmd", "dry_run")}
        run_job(cfg, job, OUT_DIR, dry=a.dry_run)
        return

    base, jobs = load_board(a.storyboard)
    picked = []
    for job in jobs:
        sid = job["id"]
        if a.only and sid not in a.only:
            picked.append((sid, job))
            continue
        print(f"\n== {sid}")
        run_job(cfg, job, os.path.join(base, sid), name_prefix=f"{sid}_", dry=a.dry_run)
        picked.append((sid, job))
    if a.concat and not a.dry_run:
        clips = []
        for sid, job in picked:
            seed = job.get("pick", (job.get("seeds") or DEFAULTS["seeds"])[0])
            w, h = parse_size(job)
            tag = "_i2v" if job.get("image") else ""
            clip = os.path.join(base, sid, f"{sid}_s{seed}_{frames_for(job)}f_{w}x{h}{tag}.mp4")
            if not os.path.exists(clip):
                raise SystemExit(f"{sid} 缺少种子 {seed} 的视频：{clip}")
            clips.append(clip)
        concat(clips, os.path.join(base, "final.mp4"))


if __name__ == "__main__":
    sys.exit(main())
