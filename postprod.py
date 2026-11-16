"""后期：漫剧镜头（静态图推拉）、配音（TTS）、字幕、音效、背景音乐和混音。video.py 调用。

分镜镜头字段：
  mode   "still" = 漫剧镜头：用关键帧做推拉平移，不跑视频模型（几秒完成）；默认是视频镜头
  camera 漫剧镜头的运镜：推进 / 拉远 / 左移 / 右移 / 上摇 / 固定 / 手持（带晃动）
  shake  true = 加镜头震动（打击感）
  line   这个镜头的旁白或台词；会合成配音、生成字幕。漫剧镜头会自动延长到能念完
  speaker 说这句的角色（cast 里的键，如 hero）：用该角色定义里的 voice，字幕前加角色名
  voice  这句用的声音（覆盖角色和全局）
  sfx    音效："impact"（内置打击声）/ "whoosh"（内置风声）/ 音频文件路径；闪白转场默认带 impact

settings：
  "tts": {"engine": "say", "voice": "Tingting", "rate": 185} # macOS 自带语音（普通话女声）；以后可换其他引擎
  "audio": {"bgm": "rain" | 音乐文件路径, "bgm_volume": 0.25, "voice_volume": 1.0, "sfx_volume": 0.8,
            "subtitles": true}
"""
import hashlib
import json
import os
import shutil
import subprocess


def ffmpeg_bin():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("没有找到 ffmpeg：Mac 用 brew install ffmpeg，Linux 用 apt install -y ffmpeg")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败：" + r.stderr.strip()[-600:])


def media_duration(path):
    """用 ffmpeg 读时长（不依赖 ffprobe）。"""
    r = subprocess.run([ffmpeg_bin(), "-i", path], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise SystemExit(f"读不到时长：{path}")


# ---------------- 漫剧镜头：静态图推拉 ----------------

def render_still(image, out, w, h, seconds, fps=25, camera=None, shake=False):
    """关键帧做推拉平移（Ken Burns）。先把图放大再推拉，避免 zoompan 的抖动。"""
    n = max(2, round(seconds * fps))
    z0, z1, x, y = 1.0, 1.04, "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if camera == "推进":
        z0, z1 = 1.0, 1.18
    elif camera == "拉远":
        z0, z1 = 1.18, 1.0
    elif camera in ("左移", "右移"):
        z0 = z1 = 1.15
        x = f"(iw-iw/zoom)*(1-on/{n})" if camera == "左移" else f"(iw-iw/zoom)*on/{n}"
    elif camera == "上摇":
        z0 = z1 = 1.15
        y = f"(ih-ih/zoom)*(1-on/{n})"
    zoom = f"{z0}+({z1}-{z0})*on/{n}"
    big = w * 4
    vf = [f"scale={big}:-2", f"zoompan=z='{zoom}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={fps}"]
    if shake or camera == "手持":
        a = 10 if shake else 4
        vf += [f"crop={w - 2 * a}:{h - 2 * a}:x='{a}+{a}*sin(n*1.9)':y='{a}+{a}*cos(n*2.7)'", f"scale={w}:{h}"]
    vf.append("format=yuv420p")
    run([ffmpeg_bin(), "-y", "-loglevel", "error", "-loop", "1", "-i", image, "-vf", ",".join(vf),
         "-frames:v", str(n), "-c:v", "libx264", "-crf", "17", out])
    return out


# ---------------- 配音 ----------------

def voice_cfg(job):
    """这句台词用的音色：speaker 角色定义里的 voice 优先，否则用分镜 settings.tts（旁白）。"""
    return job.get("_voice_cfg") or job.get("tts") or {}


def tts(text, out_dir, cfg, voice=None):
    """合成一句配音，返回 (wav 路径, 时长秒)。按内容缓存，相同句子不重复合成。"""
    engine = cfg.get("engine", "say")
    voice = voice or cfg.get("voice", "Tingting")
    rate = str(cfg.get("rate", 185))
    key = hashlib.sha1(f"{engine}|{voice}|{rate}|{text}".encode()).hexdigest()[:12]
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, f"tts_{key}.wav")
    if not os.path.exists(wav):
        if engine != "say":
            raise SystemExit(f"暂不支持的配音引擎：{engine}（目前只有 macOS 的 say）")
        if not shutil.which("say"):
            raise SystemExit("say 只在 macOS 上有；Linux 上请先生成配音文件，或关掉 line")
        aiff = wav[:-4] + ".aiff"
        subprocess.run(["say", "-v", voice, "-r", rate, "-o", aiff, text], check=True)
        if media_duration(aiff) < 0.3:  # 有些音色在列表里但没有下载语音数据，会输出空音频
            os.remove(aiff)
            raise SystemExit(f"音色 {voice} 没有输出声音（可能没下载语音数据）。普通话可用 Tingting；"
                             f"更多音色在 系统设置 → 辅助功能 → 朗读内容 → 系统声音 里下载")
        run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", aiff, "-ar", "48000", "-ac", "2", wav])
        os.remove(aiff)
    return wav, media_duration(wav)


# ---------------- 字幕 ----------------

FONT_CANDIDATES = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]


def subtitle_png(text, w, h, out):
    """把一行字幕画成透明 PNG（白字黑边，底部居中），再用 overlay 叠到视频上。"""
    from PIL import Image, ImageDraw, ImageFont
    font_path = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
    size = max(18, h // 18)
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=font)
    d.text(((w - tw) / 2, h - size * 2.4), text, font=font, fill="white", stroke_width=max(2, size // 12),
           stroke_fill="black")
    img.save(out)
    return out


# ---------------- 音效与背景声 ----------------

SFX = {  # 程序生成的占位音效（不需要素材文件）
    "impact": "aevalsrc='0.9*sin(2*PI*48*t)*exp(-5*t)+0.5*(random(0)*2-1)*exp(-18*t)':d=0.9:s=48000",
    "whoosh": "anoisesrc=d=0.6:c=pink:a=0.6:r=48000,highpass=f=500,lowpass=f=4000,afade=t=in:d=0.25,afade=t=out:st=0.3:d=0.3",
}
BGM = {  # 程序生成的环境声
    "rain": "anoisesrc=c=pink:a=0.5:r=48000,lowpass=f=2800,highpass=f=150",
}


def finish(video, out, timeline, settings, work_dir):
    """给拼好的视频加字幕、配音、音效和背景声。timeline: [(开始秒, 时长秒, 镜头参数)]"""
    audio = settings.get("audio") or {}
    tts_cfg = settings.get("tts") or {}
    w, h = map(int, settings.get("_size", "768x512").split("x"))
    total = sum(length for _, length, _ in timeline)
    ff = ffmpeg_bin()
    inputs, filters, voice_tags, sfx_tags, overlays = ["-i", video], [], [], [], []

    def add_input(args):
        inputs.extend(args)
        return len([x for x in inputs if x == "-i"]) - 1

    for i, (start, length, job) in enumerate(timeline):
        if job.get("line"):
            wav, dur = tts(job["line"], work_dir, voice_cfg(job) or tts_cfg, job.get("voice"))
            k = add_input(["-i", wav])
            ms = int((start + 0.15) * 1000)
            filters.append(f"[{k}:a]adelay={ms}|{ms},volume={audio.get('voice_volume', 1.0)}[v{i}]")
            voice_tags.append(f"[v{i}]")
            if audio.get("subtitles", True):
                text = job["line"]
                if audio.get("subtitle_speaker", True) and job.get("_speaker_name"):
                    text = f"{job['_speaker_name']}：{text}"
                png = subtitle_png(text, w, h, os.path.join(work_dir, f"sub_{i}.png"))
                k = add_input(["-i", png])
                end = start + 0.15 + max(dur, 1.2)
                overlays.append((k, start + 0.1, end))
        sfx = job.get("sfx") or ("impact" if job.get("in") == "flash" else None)
        if sfx:
            if sfx in SFX:
                k = add_input(["-f", "lavfi", "-i", SFX[sfx]])
            else:
                k = add_input(["-i", sfx])
            ms = int(start * 1000)
            filters.append(f"[{k}:a]adelay={ms}|{ms},volume={audio.get('sfx_volume', 0.8)}[s{i}]")
            sfx_tags.append(f"[s{i}]")

    mix = []
    bgm = audio.get("bgm")
    if bgm:
        if bgm in BGM:
            k = add_input(["-f", "lavfi", "-i", BGM[bgm]])
        else:
            k = add_input(["-stream_loop", "-1", "-i", bgm])
        filters.append(f"[{k}:a]atrim=0:{total:.2f},volume={audio.get('bgm_volume', 0.25)},"
                       f"afade=t=in:d=0.5,afade=t=out:st={max(0, total - 1):.2f}:d=1[bgm]")
        if voice_tags:  # 旁白出现时自动压低背景声
            # 旁白补齐到全片长度，否则压低背景声的 sidechaincompress 会在旁白结束时把背景声一起截断
            filters.append(f"{''.join(voice_tags)}amix=inputs={len(voice_tags)}:normalize=0,"
                           f"apad=whole_dur={total:.2f},asplit=2[voice][vkey]")
            filters.append("[bgm][vkey]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300[bgmd]")
            mix += ["[bgmd]", "[voice]"]
        else:
            mix.append("[bgm]")
    elif voice_tags:
        filters.append(f"{''.join(voice_tags)}amix=inputs={len(voice_tags)}:normalize=0[voice]")
        mix.append("[voice]")
    mix += sfx_tags

    vlast = "0:v"
    for j, (k, t0, t1) in enumerate(overlays):
        filters.append(f"[{vlast}][{k}:v]overlay=0:0:enable='between(t,{t0:.2f},{t1:.2f})'[ov{j}]")
        vlast = f"ov{j}"
    maps = ["-map", f"[{vlast}]" if overlays else "0:v"]
    if mix:
        filters.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=longest,"
                       f"alimiter=limit=0.95,apad=whole_dur={total:.2f}[aout]")
        maps += ["-map", "[aout]"]
    if not filters:
        shutil.copyfile(video, out)
        return
    cmd = [ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters), *maps,
           "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{total:.2f}", out]
    run(cmd)
