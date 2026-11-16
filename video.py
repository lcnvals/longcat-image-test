"""本地视频生成。Mac 上用 LTX-2.3（MLX），Linux + NVIDIA 上用 LTX-2（diffusers，见 video_cuda.py）。
和 studio.py 共用预设与 AI 提示词工具，单独运行。

视频模型很吃内存（24GB 机器上约 15GB），不和生图模型同时加载：先用 studio 出关键帧，
退出后再用本工具生成视频。LTX 装在自己的虚拟环境里，这里通过子进程调用（路径见 video.json）。

单个镜头：
  .venv-mlx/bin/python video.py gen "橘猫慢慢眨眼，尾巴轻轻摆动" --image cat.png --camera 推进 --seconds 2
  .venv-mlx/bin/python video.py gen "..." --seed 42 43 44          # 一次出多个种子挑选
  .venv-mlx/bin/python video.py gen "..." --image a.png --end-image b.png   # 锁定首尾帧

分镜（批量）：
  .venv-mlx/bin/python video.py keyframes projects/猫咖啡店/storyboard.json        # 按 keyframe 描述批量出关键帧
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json            # 生成全部镜头
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --only S03  # 只重做 S03
  .venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --concat    # 拼接成片
  .venv-mlx/bin/python video.py init projects/猫咖啡店                             # 生成分镜模板
  .venv-mlx/bin/python video.py import-prompts 提示词.txt projects/猫咖啡店         # 每行一个镜头的文本 → 分镜
      文本格式：镜号 | 画面（出关键帧） | 动作（出视频） | 运镜 | 其他参数 key=value

局部修改已有视频（任何 video.py 生成的视频都可以，包括分镜里的镜头）：
  .venv-mlx/bin/python video.py retake 视频.mp4 --from 0.8 --to 1.6 -p "猫转头看向镜头"   # 只重做这一段
  .venv-mlx/bin/python video.py extend 视频.mp4 --seconds 1 -p "猫跳下窗台"            # 往后续写
  .venv-mlx/bin/python video.py extend 视频.mp4 --image 结尾.png                        # 续写到指定画面

参数助手（交互菜单，帮你熟悉各项参数，拼好提示词和命令行）：
  .venv-mlx/bin/python video.py assist

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
    for section in ("ltx", "cuda"):
        conf = cfg.get(section, {})
        for k in ("cli", "model", "gemma", "python"):
            if k in conf and not os.path.isabs(conf[k]) and ("/" in conf[k] or conf[k].startswith(".")):
                conf[k] = os.path.normpath(os.path.join(HERE, conf[k]))
    # 后端：Mac 默认 mlx（ltx-2-mlx），其他平台默认 cuda（video_cuda.py）
    import platform
    cfg["backend"] = cfg.get("backend") or ("mlx" if platform.system() == "Darwin" else "cuda")
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


def file_sha1(path):
    if not path or not os.path.exists(path):
        return None
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


# ---------------- 生成 ----------------

def cuda_command(cfg, job, prompt, seed, w, h, frames, out):
    """Linux + NVIDIA：调用 video_cuda.py（LTX-2，diffusers）。"""
    c = cfg.get("cuda", {})
    model = c.get("model", os.path.join(HERE, "models", "ltx-2"))
    if not os.path.isdir(model):
        raise SystemExit(f"找不到 LTX-2 模型目录：{model}。先运行 deploy/setup_cuda.sh 下载")
    if job.get("_end_image"):
        print("  注意：CUDA 后端暂不支持尾帧锁定，--end-image 会被忽略")
    cmd = [c.get("python", sys.executable), os.path.join(HERE, "video_cuda.py"), "-p", prompt, "-o", out,
           "-W", str(w), "-H", str(h), "-f", str(frames), "--fps", str(job.get("fps", DEFAULTS["fps"])),
           "--seed", str(seed), "--model", model, "--steps", str(c.get("steps", 30)),
           "--guidance", str(c.get("guidance", 3.0)), "--offload", c.get("offload", "auto")]
    if job.get("_image"):
        cmd += ["--image", job["_image"]]
    neg = NEGATIVES.get(job.get("neg"), job.get("neg"))
    if neg:
        cmd += ["--negative", neg]
    if job.get("audio", DEFAULTS["audio"]):
        cmd.append("--audio")
    return cmd


def api_command(cfg, job, prompt, seed, w, h, frames, out):
    """商用视频 API（阿里云百炼 Wan 图生视频）：调用 video_api.py。按秒计费。"""
    c = cfg.get("api", {})
    if not job.get("_image"):
        raise SystemExit("Wan 图生视频需要首帧：给镜头加 image / keyframe，或用 --image")
    fps = int(job.get("fps", DEFAULTS["fps"]))
    seconds = max(2, min(15, round(frames / fps)))
    cmd = [sys.executable, os.path.join(HERE, "video_api.py"), "-p", prompt, "-o", out, "--image", job["_image"],
           "--duration", str(seconds), "--seed", str(seed), "--resolution", c.get("resolution", "720P"),
           "--model", c.get("model", "wan2.7-i2v-2026-04-25"), "--base-url", c.get("base_url", "https://dashscope.aliyuncs.com"),
           "--key-env", c.get("api_key_env", "DASHSCOPE_API_KEY")]
    if job.get("_end_image"):
        cmd += ["--end-image", job["_end_image"]]
    neg = NEGATIVES.get(job.get("neg"), job.get("neg"))
    if neg:
        cmd += ["--negative", neg]
    return cmd


def ltx_command(cfg, job, prompt, seed, w, h, frames, out):
    if cfg.get("backend") == "api":
        return api_command(cfg, job, prompt, seed, w, h, frames, out)
    if cfg.get("backend") == "cuda":
        return cuda_command(cfg, job, prompt, seed, w, h, frames, out)
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
                  "image_strength": job.get("image_strength", DEFAULTS["image_strength"]),
                  # 记录关键帧内容指纹：同名图片被 /keyframe 覆盖后，要能识别出来重新生成
                  "image_sha1": file_sha1(job.get("image")), "end_image_sha1": file_sha1(job.get("end_image"))}
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


# ---------------- 局部修改：retake / extend ----------------

def probe(video):
    """返回 (帧数, 帧率, 宽, 高, 元数据)。优先读 video.py 写的同名 .json，没有就用 ffprobe。"""
    meta_path = video[:-4] + ".json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if {"frames", "fps", "size"} <= meta.keys():
        return meta["frames"], meta["fps"], meta["size"][0], meta["size"][1], meta
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SystemExit("读取视频信息需要 ffprobe：brew install ffmpeg")
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", video],
                         capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    return int(st["nb_read_frames"]), round(num / den), st["width"], st["height"], meta


def latent_index(frame):
    """画面帧 -> 潜空间帧：第 0 帧单独一个，之后每 8 帧压成一个。"""
    return 0 if frame <= 0 else (frame - 1) // 8 + 1


def latent_range_seconds(start, end, fps):
    """潜空间帧区间 [start, end) 实际覆盖的秒数范围。"""
    first = 0 if start == 0 else 8 * (start - 1) + 1
    last = 8 * (end - 1)
    return first / fps, last / fps


def edit_job(cfg, a, kind):
    if cfg.get("backend") in ("cuda", "api"):
        raise SystemExit(f"{kind} 目前只支持 Mac 上的 LTX-2.3（ltx-2-mlx）；CUDA 后端可以用 --only 重做整个镜头")
    video = a.video
    if not os.path.exists(video):
        raise SystemExit(f"找不到视频：{video}")
    frames, fps, w, h, meta = probe(video)
    job = {k: v for k, v in vars(a).items() if v is not None and k not in ("cmd", "dry_run", "video")}
    job["prompt"] = job.get("prompt") or meta.get("prompt")
    if not job["prompt"]:
        raise SystemExit("这个视频没有记录提示词，请用 -p 指定")
    prompt = compose(job)
    ltx = cfg.get("ltx", {})
    base = video[:-4]
    seeds = job.get("seeds") or DEFAULTS["seeds"]
    total_latent = latent_index(frames - 1) + 1

    if kind == "retake":
        start = latent_index(round(a.start * fps))
        end = min(total_latent, latent_index(round(a.end * fps)) + 1)
        if end <= start:
            raise SystemExit("--to 必须大于 --from")
        t0, t1 = latent_range_seconds(start, end, fps)
        print(f"  原视频 {frames} 帧 @ {fps}fps（{(frames - 1) / fps:.2f} 秒），重做 {t0:.2f}–{t1:.2f} 秒"
              f"（潜空间帧 {start}–{end - 1}）")
        extra = ["--start", str(start), "--end", str(end)]
        tag = f"_retake{t0:.1f}-{t1:.1f}"
    else:
        n = max(1, round(float(job.get("seconds", 2)) * fps / 8))
        print(f"  原视频 {frames} 帧（{(frames - 1) / fps:.2f} 秒），往后续写约 {8 * n / fps:.2f} 秒（{8 * n} 帧）")
        extra = ["--extend-frames", str(n), "--direction", "after"]
        if job.get("image"):
            extra += ["--image", prepare_image(job["image"], w, h, os.path.dirname(video) or "."), "-1", "1.0"]
        tag = f"_ext{8 * n / fps:.1f}s"
    print(f"  实际提示词：{prompt}")

    for seed in seeds:
        out = f"{base}{tag}_s{seed}.mp4"
        cmd = [ltx["cli"], kind, "--model", ltx["model"], "--gemma", ltx["gemma"], "--distilled",
               "--video", video, "-p", prompt, "--seed", str(seed), "-o", out] + extra
        if job.get("low_ram", DEFAULTS["low_ram"]):
            cmd.append("--low-ram")
        if a.dry_run:
            print("  [dry-run] " + " ".join(cmd))
            continue
        print(f"  生成中：种子 {seed}...", flush=True)
        t = time.time()
        log = out[:-4] + ".log"
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, HF_HUB_OFFLINE="1")).returncode
        if rc != 0 or not os.path.exists(out):
            print(f"  ❌ 失败（exit={rc}），日志：{log}")
            continue
        nf, nfps, nw, nh, _ = probe(out)
        with open(out[:-4] + ".json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "seed": seed, "size": [nw, nh], "frames": nf, "fps": nfps,
                       "from": video, "edit": kind}, f, ensure_ascii=False, indent=1)
        print(f"  -> {out}  ({time.time() - t:.0f}s，{nf} 帧 / {(nf - 1) / nfps:.2f} 秒)")


# ---------------- 分镜 ----------------

def last_frame(video_path, out_png):
    """取视频最后一帧存成图片（镜头接力用）。"""
    ffmpeg = ffmpeg_bin()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", video_path, "-update", "1",
                    "-frames:v", "1", out_png], check=True)
    return out_png


def picked_clip(base, job):
    """分镜里某个镜头被选用（pick）的视频路径。漫剧镜头（mode=still）没有种子。"""
    if job.get("mode") == "still":
        w, h = parse_size(job)
        return os.path.join(base, job["id"], f"{job['id']}_still_{w}x{h}.mp4")
    seed = job.get("pick", (job.get("seeds") or DEFAULTS["seeds"])[0])
    w, h = parse_size(job)
    tag = "_i2v" if job.get("image") or job.get("start_from") else ""
    return os.path.join(base, job["id"], f"{job['id']}_s{seed}_{frames_for(job)}f_{w}x{h}{tag}.mp4")


def render_still_shot(base, job):
    """漫剧镜头：关键帧推拉平移；有台词时自动延长到念完。"""
    import postprod
    if not job.get("image") or not os.path.exists(job["image"]):
        raise SystemExit(f"{job['id']} 是漫剧镜头，需要关键帧图片（image）")
    w, h = parse_size(job)
    fps = int(job.get("fps", DEFAULTS["fps"]))
    seconds = float(job.get("seconds", DEFAULTS["seconds"]))
    if job.get("line"):
        _, dur = postprod.tts(job["line"], os.path.join(base, "audio"), postprod.voice_cfg(job), job.get("voice"))
        seconds = max(seconds, dur + 0.5)
    out = picked_clip(base, job)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    postprod.render_still(job["image"], out, w, h, seconds, fps, job.get("camera"), job.get("shake"))
    print(f"  漫剧镜头：{seconds:.2f} 秒，运镜 {job.get('camera') or '固定'} -> {os.path.relpath(out, HERE)}"
          f"  ({time.time() - t0:.0f}s)")


def link_shots(base, jobs, i):
    """镜头衔接：start_from=prev 用上一镜最后一帧作首帧；end_to=next 用下一镜关键帧作尾帧。"""
    job = jobs[i]
    if job.get("start_from") == "prev":
        if i == 0:
            raise SystemExit(f"{job['id']} 是第一个镜头，没有上一镜可以接力")
        prev = picked_clip(base, jobs[i - 1])
        if not os.path.exists(prev):
            raise SystemExit(f"{job['id']} 要接力 {jobs[i - 1]['id']}，但还没有它的视频：{prev}（先生成上一镜）")
        os.makedirs(os.path.join(base, job["id"]), exist_ok=True)
        job["image"] = last_frame(prev, os.path.join(base, job["id"], f"from_{jobs[i - 1]['id']}.png"))
        print(f"  接力：首帧取自 {jobs[i - 1]['id']} 的最后一帧")
    if job.get("end_to") == "next":
        if i == len(jobs) - 1 or not jobs[i + 1].get("image"):
            raise SystemExit(f"{job['id']} 要对准下一镜，但下一镜没有关键帧")
        job["end_image"] = jobs[i + 1]["image"]
        print(f"  对准：尾帧取自 {jobs[i + 1]['id']} 的关键帧")

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


REF_KEYS = ("character", "scene", "refs", "cast", "ref_images", "max_refs", "keyframe_ref_model")


def fill_refs(text, settings):
    """把 {character}、{scene} 以及 settings.refs 里的 {名字} 替换成设定描述。"""
    refs = {k: settings[k] for k in ("character", "scene") if isinstance(settings.get(k), str)}
    refs.update(settings.get("refs") or {})
    for k, v in refs.items():
        text = text.replace("{" + k + "}", v)
    return text


def load_board(path):
    """读分镜，展开每个镜头：定义（cast）、换装（loadout）、场景变体（variant）、技能（skill + stage）。"""
    with open(path, encoding="utf-8") as f:
        board = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    settings = {k: v for k, v in board.get("settings", {}).items() if not k.startswith("_")}
    own_refs = dict(settings.get("refs") or {})  # 分镜里直接写的 refs 文字，优先于定义
    cast = settings.get("cast")
    assets = None
    if cast or any(s.get("skill") or s.get("blocking") or s.get("camera_from") for s in board["shots"]):
        sys.path.insert(0, HERE)
        import assets
    cache = {}

    def cast_for(loadouts=None, variants=None, phases=None):
        """按镜头的换装/变体/阶段解析 cast，相同组合只解析一次。"""
        key = tuple(json.dumps(x or {}, sort_keys=True) for x in (loadouts, variants, phases))
        if key not in cache:
            cache[key] = assets.resolve_cast(cast, True, loadouts, variants, phases) if cast else ({}, {}, {})
        return cache[key]

    per_asset = int(settings.get("ref_images", 1))
    max_refs = int(settings.get("max_refs", 3))  # 参考图太多会更慢、更占内存
    jobs = []
    prev_cam, axis_memory = None, {}
    for shot in board["shots"]:
        texts, cast_imgs, cast_voices = cast_for(shot.get("loadout"), shot.get("variant"), shot.get("phase"))
        shot_settings = dict(settings, refs={**texts, **own_refs})
        job = {k: v for k, v in settings.items() if k not in REF_KEYS}
        job.update({k: v for k, v in shot.items() if not k.startswith("_")})
        job["prompt"] = fill_refs(job.get("prompt", ""), shot_settings)  # 漫剧镜头可以不写
        if job.get("skill"):  # 技能：这一阶段的描述加进画面和动作，并带上技能的音效和剪辑方式
            sk = assets.find_skill(job["skill"])
            stxt = assets.skill_text(sk, job.get("stage"))
            job["prompt"] = "，".join(x for x in (job["prompt"], stxt) if x)
            if job.get("keyframe"):
                job["keyframe"] = job["keyframe"] + "，" + stxt
            for k, v in (sk.get("edit") or {}).items():
                job.setdefault(k, v) if k not in shot else None
            if sk.get("sfx") and "sfx" not in shot:
                job["sfx"] = sk["sfx"]
        if job.get("speaker"):  # 台词跟随角色：用角色定义里的音色，字幕可带上角色名
            sp = cast_voices.get(job["speaker"])
            if sp is None:
                raise SystemExit(f"{shot['id']} 的 speaker“{job['speaker']}”不在 cast 里：{'、'.join(cast_voices) or '无'}")
            job["_speaker_name"] = sp["name"]
            if sp.get("voice") and not shot.get("voice"):
                job["_voice_cfg"] = sp["voice"]
        cam = shot.get("camera_from") or settings.get("camera_from")
        if job.get("blocking") or (cam and any(v.get("layout") for v in cast_voices.values())):
            # 站位：东南西北 -> 画面左右、深处、前景；机位越过轴线时提醒（左右会反转）
            if not cam:
                raise SystemExit(f"{shot['id']} 写了 blocking，需要 camera_from（机位在地点的哪一侧）")
            blk = job.get("blocking") or {}
            if len(blk) >= 2:  # 轴线规则：两个角色的连线是轴线，机位要待在同一侧，否则左右反转
                (ka, pa), (kb, pb) = list(blk.items())[:2]
                pair = tuple(sorted((ka, kb)))
                side = assets.axis_side(blk[pair[0]], blk[pair[1]], cam)
                last = axis_memory.get(pair)
                if side and last and side != last[0]:
                    print(f"  提示：{shot['id']} 机位（{cam}侧）越过了 {pair[0]}–{pair[1]} 的轴线（上次在 {last[1]} 是另一侧），"
                          f"画面里两人的左右位置会反转；除非有意为之，建议保持在同一侧或加过渡镜头")
                if side:
                    axis_memory[pair] = (side, shot["id"])
            elif prev_cam and assets.opposite(prev_cam, cam):
                print(f"  提示：{shot['id']} 机位从{prev_cam}侧跳到对面的{cam}侧，画面左右会反转")
            prev_cam = cam
            place = [f"{cast_voices[k]['name'] if k in cast_voices else k}位于{assets.screen_pos(d, cam)}"
                     for k, d in (job.get("blocking") or {}).items()]
            for k, v in cast_voices.items():  # 地点布局也按机位换算
                if v.get("layout") and "{" + k + "}" in (job.get("keyframe") or ""):
                    place.append(assets.layout_text(v["layout"], cam))
            if place and job.get("keyframe"):
                job["keyframe"] += "，" + "，".join(p for p in place if p)
        if job.get("keyframe"):
            raw_kf = job["keyframe"]
            # 这个镜头出现了哪些定义（按在描述里出现的顺序），各取前 n 张参考图（本体在前、装备在后），总数不超过 max_refs
            used = sorted((k for k in cast_imgs if "{" + k + "}" in raw_kf), key=lambda k: raw_kf.index("{" + k + "}"))
            job["_cast"] = used
            n = int(shot.get("ref_images", per_asset))
            job["_refs"] = [img for k in used for img in cast_imgs[k][:n]][:int(shot.get("max_refs", max_refs))]
            job["keyframe"] = fill_refs(raw_kf, shot_settings)
        for key in ("image", "end_image"):
            if job.get(key) and not os.path.isabs(job[key]):
                job[key] = os.path.join(base, job[key])
        jobs.append(job)
    return base, jobs


def gen_keyframes(path, only=None, model=None, quantize=None, seeds=None, force=False, use_refs=True):
    """按分镜里每个镜头的 keyframe（静态画面描述）批量出关键帧，写回 image 字段。

    分镜用 cast 引用了定义、且定义有设定图时，用支持参考图的模型（默认 FLUX.2 klein）以设定图为参考出图。
    """
    base, jobs = load_board(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    settings = raw.get("settings", {})
    has_refs = use_refs and any(j.get("_refs") for j in jobs if j.get("keyframe") and (not only or j["id"] in only))
    model = model or (settings.get("keyframe_ref_model", "flux2-klein-4b") if has_refs
                      else settings.get("keyframe_model", "z-image-turbo"))
    todo = [j for j in jobs if j.get("keyframe") and (not only or j["id"] in only)]
    todo = [j for j in todo if force or only or not os.path.exists(os.path.join(base, f"{j['id']}.png"))]
    if not todo:
        print("没有需要生成的关键帧（已存在的会跳过；--only 指定镜头或加 --force 重新生成）")
        return
    sys.path.insert(0, HERE)
    from backends import BACKENDS
    print(f"加载 {BACKENDS[model].title} ...")
    backend = BACKENDS[model](quantize=quantize)
    if has_refs and not backend.supports_refs:
        print(f"  注意：{backend.title} 不支持参考图，这次只用文字描述（换 --model flux2-klein-4b 可以用设定图）")
    for job in todo:
        w, h = parse_size(job)
        kw, kh = map(int, str(job.get("keyframe_size") or f"{w * 3 // 2}x{h * 3 // 2}").split("x"))
        prompt = compose(dict(job, prompt=job["keyframe"], camera=None, motion=None, ai=None))
        shot_seeds = seeds or job.get("keyframe_seeds") or [job.get("keyframe_seed", 42)]
        refs = job.get("_refs") if use_refs and backend.supports_refs else []
        print(f"\n== {job['id']}  {kw}x{kh}  种子 {shot_seeds}\n  {prompt}")
        if refs:
            print("  参考设定图：" + "、".join(os.path.relpath(r, HERE) for r in refs))
        tag = "_ref" if refs else ""
        for seed in shot_seeds:
            t0 = time.time()
            if refs:
                img = backend.generate_refs(prompt + "，人物和怪物的外观与参考图保持一致", refs, kw, kh,
                                            backend.final_steps, seed)
            else:
                img, _ = backend.generate(prompt, "", kw, kh, backend.final_steps, seed, backend.cfg)
            out = os.path.join(base, f"{job['id']}_kf_s{seed}{tag}.png")
            img.save(out)
            backend.cleanup()
            print(f"  -> {os.path.relpath(out, HERE)}  ({time.time() - t0:.0f}s)")
        # 第一个种子作为当前关键帧；想换就把 image 改成其他 _kf_sN.png
        chosen = f"{job['id']}_kf_s{shot_seeds[0]}{tag}.png"
        shutil.copyfile(os.path.join(base, chosen), os.path.join(base, f"{job['id']}.png"))
        for shot in raw["shots"]:
            if shot.get("id") == job["id"]:
                shot["image"] = f"{job['id']}.png"
                shot["_keyframe_from"] = chosen
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)


def ffmpeg_bin():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit("没有找到 ffmpeg：Mac 用 brew install ffmpeg，Linux 用 apt install -y ffmpeg")
    return ffmpeg


EDIT_KEYS = ("speed", "trim", "in", "out")


def edit_concat(items, out, edit, settings=None, base=None):
    """带剪辑的拼接：每个镜头先单独处理（截取、变速、闪白/淡入淡出），再拼接，可选补帧。

    items: [(视频路径, 镜头参数)]，镜头参数里可以有：
      speed  变速倍数（>1 加快），trim [开始秒, 结束秒]（按原视频时间），
      in     开头转场：cut（默认）/ flash（闪白，适合打击瞬间）/ fade（从黑场淡入）
      out    结尾转场：cut（默认）/ fade（淡出到黑场）
    edit（分镜 settings.edit）：fps 输出帧率（默认 25），smooth 补帧到的帧率（如 50，不填不补帧），
      fade_in / fade_out 全片开头淡入、结尾淡出的秒数
    """
    ffmpeg = ffmpeg_bin()
    tmp = out[:-4] + "_parts"
    os.makedirs(tmp, exist_ok=True)
    fps = int(edit.get("fps", 25))
    tw, th = parse_size(items[0][1])  # 统一缩放到分镜尺寸：本地视频、API 视频、漫剧镜头可以混剪
    parts, total, timeline = [], 0.0, []
    for idx, (clip, job) in enumerate(items):
        frames, cfps, _, _, _ = probe(clip)
        dur = (frames - 1) / cfps
        a, b = job.get("trim") or [0, dur]
        b = min(b, dur)
        speed = float(job.get("speed", 1.0))
        length = (b - a) / speed
        vf = [f"trim=start={a}:end={b}", f"setpts=(PTS-STARTPTS)/{speed}", f"fps={fps}",
              f"scale={tw}:{th}:force_original_aspect_ratio=increase", f"crop={tw}:{th}", "setsar=1"]
        if job.get("line") and job.get("mode") != "still":
            # 台词比画面长：最后一帧定格到念完，避免台词被下一镜或片尾截断
            import postprod
            _, dur = postprod.tts(job["line"], os.path.join(base or tmp, "audio"), postprod.voice_cfg(job), job.get("voice"))
            hold = dur + 0.4 - length
            if hold > 0.05:
                vf.append(f"tpad=stop_mode=clone:stop_duration={hold:.2f}")
                length += hold
        fin = job.get("in", "cut")
        if idx == 0 and edit.get("fade_in"):
            fin = "fade"
        if fin == "flash":
            vf.append("fade=t=in:st=0:d=0.15:color=white")
        elif fin == "fade":
            vf.append(f"fade=t=in:st=0:d={edit.get('fade_in', 0.4)}")
        fout = job.get("out", "cut")
        if idx == len(items) - 1 and edit.get("fade_out"):
            fout = "fade"
        if fout == "fade":
            d = float(edit.get("fade_out", 0.6))
            vf.append(f"fade=t=out:st={max(0, length - d):.3f}:d={d}")
        part = os.path.join(tmp, f"{idx:02d}.mp4")
        rc = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", clip, "-an", "-vf", ",".join(vf),
                             "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", part]).returncode
        if rc != 0:
            raise SystemExit(f"处理 {clip} 失败")
        parts.append(part)
        timeline.append((total, length, job))
        total += length
        print(f"  {os.path.basename(clip)}：截取 {a:.2f}–{b:.2f} 秒，{speed:g} 倍速 → {length:.2f} 秒，"
              f"开头 {fin}，结尾 {fout}")
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    settings = settings or {}
    post = settings.get("audio") or any(j.get("line") or j.get("sfx") or j.get("in") == "flash" for _, _, j in timeline)
    joined = os.path.join(tmp, "joined.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined],
                   check=True)
    video_only = joined
    if edit.get("smooth"):  # 运动补帧，让动作更顺滑
        print(f"  补帧到 {edit['smooth']}fps（需要一两分钟）...", flush=True)
        video_only = os.path.join(tmp, "smooth.mp4")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", joined, "-vf",
                        f"minterpolate=fps={edit['smooth']}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", video_only], check=True)
    if post:
        import postprod
        w, h = parse_size(items[0][1])
        print("  加配音、字幕、音效和背景声...", flush=True)
        postprod.finish(video_only, out, timeline, dict(settings, _size=f"{w}x{h}"),
                        os.path.join(base or tmp, "audio"))
    else:
        shutil.copyfile(video_only, out)
    shutil.rmtree(tmp)
    print(f"成片 -> {out}（{total:.1f} 秒）")


def import_prompts(txt, project_dir, clean=True, replace=False):
    """把“每行一个镜头”的提示词文本导入成分镜（新建或按镜号合并）。"""
    from prompttools import clean_prompt, parse_shot_lines
    with open(txt, encoding="utf-8") as f:
        shots = parse_shot_lines(f.read())
    if not shots:
        raise SystemExit(f"{txt} 里没有解析到镜头。格式：镜号 | 画面 | 动作 | 运镜 | key=value")
    notes = []
    for shot in shots:
        line = shot.pop("_line")
        if clean:
            for key in ("keyframe", "prompt"):
                if shot.get(key):
                    shot[key], removed = clean_prompt(shot[key])
                    if removed:
                        notes.append(f"  第 {line} 行 {shot['id']} {key}：去掉了 {'、'.join(removed)}")
        cam = shot.get("camera")
        if cam and cam not in CAMERAS:  # 不是预设的运镜：并进动作描述里，避免生成时报错
            shot["prompt"] = "，".join(x for x in (shot.get("prompt"), cam) if x)
            shot.pop("camera")
            notes.append(f"  第 {line} 行 {shot['id']}：运镜“{cam}”不是预设（{'、'.join(CAMERAS)}），已并入动作描述")
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, "storyboard.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            board = json.load(f)
    else:
        board = {"settings": {"ratio": "横屏", "fps": 25, "seconds": 2, "seeds": [42],
                              "keyframe_model": "z-image-turbo", "keyframe_size": "1152x768"}, "shots": []}
    if replace:
        board["shots"] = []
    existing = {x["id"]: x for x in board["shots"]}
    added = updated = 0
    for shot in shots:
        if shot["id"] in existing:
            existing[shot["id"]].update(shot)  # 合并：保留文本里没写的已有字段（如 pick、剪辑参数）
            updated += 1
        else:
            board["shots"].append(shot)
            added += 1
    board["shots"].sort(key=lambda x: x.get("id", ""))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False, indent=2)
    print(f"分镜 -> {os.path.relpath(path, HERE)}：新增 {added} 个镜头，更新 {updated} 个，共 {len(board['shots'])} 个")
    if notes:
        print("清理与提示：\n" + "\n".join(notes))
    for shot in shots:
        kind = "漫剧" if shot.get("mode") == "still" else "视频"
        print(f"  {shot['id']} [{kind}] 画面：{(shot.get('keyframe') or '-')[:30]} | 动作：{(shot.get('prompt') or '-')[:20]}"
              f" | 运镜：{shot.get('camera') or '-'}")
    rel = os.path.relpath(path, HERE)
    print(f"下一步：\n  .venv-mlx/bin/python video.py keyframes {rel}     # 批量出关键帧\n"
          f"  .venv-mlx/bin/python video.py board {rel} --concat  # 生成视频并拼接")


def concat(clips, out):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:  # 云服务器上常没有系统 ffmpeg，用 imageio-ffmpeg 自带的版本
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit("没有找到 ffmpeg：Mac 用 brew install ffmpeg，Linux 用 apt install -y ffmpeg")
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


# ---------------- 参数助手 ----------------

ASSIST_COMMANDS = {
    "/image": "首帧关键帧（图生视频，画面最可控）",
    "/end-image": "尾帧关键帧（锁定结尾画面）",
    "/camera": "运镜",
    "/motion": "动作幅度",
    "/ratio": "视频画幅",
    "/seed": "种子（可多个，空格分隔）",
    "/seconds": "时长（秒）",
    "/fps": "帧率",
    "/style": "风格",
    "/shot": "景别构图",
    "/light": "光线",
    "/tone": "色调氛围",
    "/time": "时间天气",
    "/detail": "细节质感（可多选）",
    "/neg": "负面提示词",
    "/ai": "生成前用 AI 处理提示词",
    "/audio": "同时生成音频 on/off",
    "/show": "查看参数、实际提示词和等价命令行",
    "/tips": "视频提示词写法建议",
    "/save": "存进分镜：/save 片名 镜号",
    "/dry": "打印将要执行的 LTX 命令（不生成）",
    "/gen": "按当前参数生成（约 2–3 分钟）",
    "/reset": "清空所有参数",
    "/help": "命令列表",
    "/quit": "退出",
}

ASSIST_TIPS = """  视频提示词写法：
  · 写“怎么动”，而不是“长什么样”：画面外观交给关键帧（/image），提示词只写动作，如“猫慢慢抬起头”
  · 一个镜头只写一个简单动作；多个动作拆成多个镜头
  · 运镜越简单越稳：固定、推进、拉远成功率最高；环绕、跟拍容易出错
  · LTX 每段 2 秒左右最稳，超过约 4 秒成功率明显下降，长视频靠分镜拼接
  · 用图生视频时，风格、光线主要由关键帧决定；预设只起辅助作用，最好在 studio 出关键帧时定好
  · 同一个镜头多试几个种子（/seed 42 43 44），挑最好的一个
  · 角色要前后一致：在分镜 settings 里写好 character，每个镜头用 {character} 引用"""

NUMERIC_HINTS = {
    "/seconds": [("2", "默认，LTX 最稳"), ("1", "很短，适合补镜头"), ("3", "稍长，动作可以多一点"),
                 ("4", "接近上限，成功率下降")],
    "/fps": [("25", "默认"), ("24", "LTX-2.3 训练时的帧率"), ("16", "更省算力，动作略顿")],
    "/seed": [("42", "默认"), ("42 43 44", "一次出 3 个挑选")],
    "/audio": [("off", "默认不生成音频"), ("on", "同时生成环境音")],
}


def assist_choices(cmd):
    if cmd in ("/image", "/end-image"):
        pics = []  # 最近生成的图片，方便直接选
        for root in (os.path.join(HERE, "outputs"), os.path.join(HERE, "projects")):
            for dirpath, _, files in os.walk(root):
                pics += [os.path.join(dirpath, f) for f in files if f.endswith(".png") and not f.startswith("key_")]
        pics.sort(key=os.path.getmtime, reverse=True)
        return [(os.path.relpath(x, HERE), time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(x))))
                for x in pics[:10]] + [("off", "取消")]
    if cmd == "/camera":
        return list(CAMERAS.items()) + [("off", "取消")]
    if cmd == "/motion":
        return list(MOTIONS.items()) + [("off", "取消")]
    if cmd == "/ratio":
        return [(n, f"{w}x{h}") for n, (w, h) in VIDEO_RATIOS.items()]
    if cmd == "/neg":
        return list(NEGATIVES.items()) + [("off", "清除")]
    if cmd == "/ai":
        from ai import load_config
        return [(n, t.get("desc", "")) for n, t in load_config()["tools"].items()
                if t.get("target") == "prompt" and t.get("input") != "image"] + [("off", "取消")]
    for key, (c, _, table) in GROUPS.items():
        if c == cmd:
            return list(table.items()) + [("off", "取消")]
    return NUMERIC_HINTS.get(cmd, [])


class Assist:
    KEYS = {"/image": "image", "/end-image": "end_image", "/camera": "camera", "/motion": "motion",
            "/ratio": "ratio", "/neg": "neg", "/ai": "ai"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.job = {}

    def cli_line(self):
        """等价的 video.py gen 命令行，可以直接复制使用。"""
        import re
        import shlex

        def q(x):  # 只有含空格或 shell 特殊字符时才加引号，中文不加
            x = str(x)
            return x if re.fullmatch(r"[^\s'\"$`\\!*?&|;<>()\[\]{}#~]+", x) else shlex.quote(x)

        j = self.job
        parts = [".venv-mlx/bin/python", "video.py", "gen", q(j.get("prompt", "（提示词）"))]
        for key, flag in (("image", "--image"), ("end_image", "--end-image"), ("ratio", "--ratio"),
                          ("camera", "--camera"), ("motion", "--motion"), ("seconds", "--seconds"),
                          ("fps", "--fps"), ("neg", "--neg"), ("ai", "--ai")):
            if j.get(key) is not None:
                parts += [flag, q(j[key])]
        for key in GROUPS:
            v = j.get(key)
            if v:
                parts += [f"--{key}"] + [q(x) for x in (v if isinstance(v, list) else [v])]
        if j.get("seeds"):
            parts += ["--seed"] + [str(x) for x in j["seeds"]]
        if j.get("audio"):
            parts.append("--audio")
        return " ".join(parts)

    def show(self):
        j = self.job
        w, h = parse_size(j)
        frames = frames_for(j)
        fps = int(j.get("fps", DEFAULTS["fps"]))
        print(f"  画幅 {j.get('ratio') or f'{w}x{h}'} | {frames} 帧 @ {fps}fps（{frames / fps:.2f} 秒）| "
              f"种子 {j.get('seeds') or DEFAULTS['seeds']} | 音频 {'开' if j.get('audio') else '关'}")
        print(f"  首帧 {j.get('image') or '-'} | 尾帧 {j.get('end_image') or '-'}")
        presets = [f"{label} {'+'.join(j[k]) if isinstance(j.get(k), list) else j.get(k)}"
                   for k, (_, label, _) in GROUPS.items() if j.get(k)]
        presets += [f"运镜 {j['camera']}"] if j.get("camera") else []
        presets += [f"动作 {j['motion']}"] if j.get("motion") else []
        print(f"  预设：{' | '.join(presets) or '-'}")
        neg = NEGATIVES.get(j.get("neg"), j.get("neg"))
        print(f"  负面提示词：{neg or '-'}" + (f" | AI：{j['ai']}（生成时执行）" if j.get("ai") else ""))
        if j.get("prompt"):
            print(f"  实际提示词：{compose(dict(j, ai=None))}")
        else:
            print("  还没有提示词：直接输入一句描述动作的话")
        print(f"  等价命令行：\n    {self.cli_line()}")

    def save(self, arg):
        parts = arg.split()
        if len(parts) != 2 or not self.job.get("prompt"):
            print("  用法：/save 片名 镜号（先输入提示词），例如 /save 猫咖啡店 S03")
            return
        name, sid = parts
        proj = os.path.join(HERE, "projects", name)
        os.makedirs(proj, exist_ok=True)
        path = os.path.join(proj, "storyboard.json")
        board = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else \
            {"settings": {"character": "", "scene": "", "ratio": "横屏", "fps": 25, "seconds": 2, "seeds": [42]},
             "shots": []}
        shot = {k: v for k, v in self.job.items() if v not in (None, [], "")}
        shot["id"] = sid
        for key in ("image", "end_image"):  # 关键帧复制进片目录，分镜里用相对路径
            if shot.get(key):
                dst = os.path.join(proj, f"{sid}{'_end' if key == 'end_image' else ''}.png")
                shutil.copyfile(os.path.join(HERE, shot[key]) if not os.path.isabs(shot[key]) else shot[key], dst)
                shot[key] = os.path.basename(dst)
        board["shots"] = [x for x in board["shots"] if x.get("id") != sid] + [shot]
        board["shots"].sort(key=lambda x: x.get("id", ""))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)
        print(f"  已存为 {os.path.relpath(path, HERE)} 的 {sid}（共 {len(board['shots'])} 个镜头）")
        print(f"  生成：.venv-mlx/bin/python video.py board {os.path.relpath(path, HERE)} --only {sid}")

    def command(self, line):
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        j = self.job
        if cmd in self.KEYS or any(cmd == c for c, _, _ in GROUPS.values()):
            key = self.KEYS.get(cmd) or next(k for k, (c, _, _) in GROUPS.items() if c == cmd)
            if not arg:  # 列出选项，帮助熟悉
                for value, desc in assist_choices(cmd):
                    print(f"  {value:<8} {desc}")
                return
            if arg == "off":
                j.pop(key, None)
            elif key in MULTI:
                cur = j.get(key) or []
                j[key] = [x for x in cur if x != arg] if arg in cur else cur + [arg]
            else:
                if key in ("image", "end_image") and not os.path.exists(os.path.join(HERE, arg)) \
                        and not os.path.exists(arg):
                    print(f"  找不到图片：{arg}")
                    return
                valid = dict(assist_choices(cmd))
                if key not in ("image", "end_image", "neg") and arg not in valid:
                    print(f"  没有“{arg}”这个选项，输入 {cmd} 查看列表")
                    return
                j[key] = arg
            self.show()
        elif cmd == "/seconds":
            j["seconds"] = float(arg)
            self.show()
        elif cmd == "/fps":
            j["fps"] = int(arg)
            self.show()
        elif cmd == "/seed":
            j["seeds"] = [int(x) for x in arg.split()]
            self.show()
        elif cmd == "/audio":
            j["audio"] = arg == "on"
            self.show()
        elif cmd == "/show":
            self.show()
        elif cmd == "/tips":
            print(ASSIST_TIPS)
        elif cmd == "/save":
            self.save(arg)
        elif cmd in ("/dry", "/gen"):
            if not j.get("prompt"):
                print("  先输入提示词")
                return
            job = dict(j)  # 在副本上补全路径，不改动助手里的参数
            for key in ("image", "end_image"):
                if job.get(key) and not os.path.isabs(job[key]):
                    job[key] = os.path.join(HERE, job[key])
            run_job(self.cfg, job, OUT_DIR, dry=cmd == "/dry")
        elif cmd == "/reset":
            self.job = {}
            self.show()
        elif cmd in ("/help", "/?"):
            for c, d in ASSIST_COMMANDS.items():
                print(f"  {c:<11} {d}")
        else:
            print("  未知命令，/help 查看")

    def loop(self):
        from menu import make_prompt_session
        print("视频参数助手：帮你熟悉各项参数，拼出提示词和命令行。")
        print("  直接输入一句话 = 设为提示词（写动作）；输入 / 弹出命令菜单，每个选项都有说明；/tips 查看写法建议")
        read = make_prompt_session(ASSIST_COMMANDS, assist_choices, os.path.join(OUT_DIR, ".assist_history"), "video")
        read = read or (lambda: input("\nvideo> "))
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
                    self.job["prompt"] = line
                    self.show()
            except (SystemExit, Exception) as e:  # 参数校验失败也不退出助手
                print(f"  {e}")


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
    p.add_argument("--backend", choices=["mlx", "cuda", "api"], help="视频后端：mlx（Mac 本地）/ cuda（NVIDIA）/ api（百炼 Wan，按秒计费）")
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
    b.add_argument("--concat", action="store_true", help="生成后按镜头顺序拼接成片（用每个镜头的 pick 种子）；"
                   "镜头或 settings.edit 里有剪辑参数时按剪辑参数处理")
    b.add_argument("--output", help="成片文件名（默认 final.mp4）")
    b.add_argument("--backend", choices=["mlx", "cuda", "api"], help="视频后端（覆盖 video.json）")
    b.add_argument("--dry-run", action="store_true")
    def add_edit_args(p):
        p.add_argument("video")
        p.add_argument("-p", "--prompt", help="提示词（默认沿用原视频的）")
        p.add_argument("--seed", dest="seeds", type=int, nargs="+")
        p.add_argument("--camera", choices=list(CAMERAS))
        p.add_argument("--motion", choices=list(MOTIONS))
        p.add_argument("--style", choices=list(GROUPS["style"][2]))
        p.add_argument("--ai", help="先用 AI 工具处理提示词")
        p.add_argument("--no-low-ram", dest="low_ram", action="store_false", default=None)
        p.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("retake", help="只重做视频里的一段（按秒指定）")
    add_edit_args(r)
    r.add_argument("--from", dest="start", type=float, required=True, help="开始秒数")
    r.add_argument("--to", dest="end", type=float, required=True, help="结束秒数")
    e = sub.add_parser("extend", help="往后续写视频")
    add_edit_args(e)
    e.add_argument("--seconds", type=float, help="续写秒数（默认 2）")
    e.add_argument("--image", help="续写结尾要到达的画面")
    k = sub.add_parser("keyframes", help="按分镜批量生成关键帧（用生图模型，需在 .venv-mlx 里运行）")
    k.add_argument("storyboard")
    k.add_argument("--only", nargs="+", help="只生成这些镜头")
    k.add_argument("--model", help="生图模型（默认 settings.keyframe_model 或 z-image-turbo）")
    k.add_argument("--quantize", type=int)
    k.add_argument("--seed", dest="seeds", type=int, nargs="+", help="覆盖种子；多个种子都会出图，第一个作为当前关键帧")
    k.add_argument("--force", action="store_true", help="已有关键帧也重新生成")
    k.add_argument("--no-refs", dest="use_refs", action="store_false", help="不用定义里的设定图做参考，只用文字")
    ip = sub.add_parser("import-prompts", help="把每行一个镜头的提示词文本导入成分镜")
    ip.add_argument("textfile")
    ip.add_argument("project_dir")
    ip.add_argument("--no-clean", dest="clean", action="store_false",
                    help="不清理 Midjourney 参数、WebUI 权重括号、LoRA 标签等专用语法")
    ip.add_argument("--replace", action="store_true", help="清空原有镜头再导入（默认按镜号合并）")
    sub.add_parser("assist", help="参数助手：交互菜单熟悉各项参数")
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

    if a.cmd == "import-prompts":
        import_prompts(a.textfile, a.project_dir, a.clean, a.replace)
        return
    if a.cmd == "keyframes":
        gen_keyframes(a.storyboard, a.only, a.model, a.quantize, a.seeds, a.force, a.use_refs)
        return
    cfg = load_cfg()
    if getattr(a, "backend", None):
        cfg["backend"] = a.backend
    if cfg["backend"] == "api" and a.cmd in ("gen", "board") and not getattr(a, "dry_run", False):
        print("注意：api 后端按生成的视频秒数计费（百炼 Wan），失败不计费；价格以百炼控制台为准")
    if a.cmd == "assist":
        Assist(cfg).loop()
        return
    if a.cmd in ("retake", "extend"):
        edit_job(cfg, a, a.cmd)
        return
    if a.cmd == "gen":
        job = {k: v for k, v in vars(a).items() if v is not None and k not in ("cmd", "dry_run", "backend")}
        run_job(cfg, job, OUT_DIR, dry=a.dry_run)
        return

    base, jobs = load_board(a.storyboard)
    picked = []
    for i, job in enumerate(jobs):
        sid = job["id"]
        if a.only and sid not in a.only:
            if job.get("start_from") == "prev" and a.concat:
                job["image"] = os.path.join(base, sid, f"from_{jobs[i - 1]['id']}.png")  # 拼接时只需路径规则一致
            picked.append((sid, job))
            continue
        print(f"\n== {sid}")
        link_shots(base, jobs, i)
        if job.get("mode") == "still":
            if not a.dry_run:
                render_still_shot(base, job)
        else:
            run_job(cfg, job, os.path.join(base, sid), name_prefix=f"{sid}_", dry=a.dry_run)
        picked.append((sid, job))
    if a.concat and not a.dry_run:
        with open(a.storyboard, encoding="utf-8") as f:
            edit = json.load(f).get("settings", {}).get("edit")
        clips = []
        for sid, job in picked:
            clip = picked_clip(base, job)
            if job.get("mode") == "still" and not os.path.exists(clip):
                render_still_shot(base, job)
            if not os.path.exists(clip):
                raise SystemExit(f"{sid} 缺少选用种子的视频：{clip}")
            clips.append(clip)
        out = os.path.join(base, a.output or "final.mp4")
        with open(a.storyboard, encoding="utf-8") as f:
            settings = json.load(f).get("settings", {})
        post = settings.get("audio") or any(job.get("line") or job.get("sfx") for _, job in picked)
        if edit or post or any(any(k in job for k in EDIT_KEYS) for _, job in picked):
            edit_concat(list(zip(clips, [job for _, job in picked])), out, edit or {}, settings, base)
        else:
            concat(clips, out)


if __name__ == "__main__":
    sys.exit(main())
