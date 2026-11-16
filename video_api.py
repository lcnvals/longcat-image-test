"""商用视频 API 运行器：阿里云百炼 Wan（通义万相）图生视频。由 video.py 通过子进程调用，也可单独运行。

  export DASHSCOPE_API_KEY=你的密钥        # 百炼控制台创建，注意和地域一致
  python video_api.py -p "橘猫慢慢眨眼" --image cat.png --duration 5 -o out.mp4

流程：提交异步任务 → 每 15 秒查询一次状态 → 成功后立刻下载视频（官方视频链接只保留 24 小时）。
按输出视频的秒数计费，失败不计费；价格以百炼控制台为准。只用标准库。
文档：https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://dashscope.aliyuncs.com"   # 北京地域（文档说明旧域名仍可用）；新加坡等地域用 --base-url
DEFAULT_MODEL = "wan2.7-i2v-2026-04-25"


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def request(url, key, body=None, timeout=60):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if body is not None:
        headers["X-DashScope-Async"] = "enable"  # HTTP 调用必须声明异步
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"API 返回 HTTP {e.code}：{e.read().decode(errors='replace')[:400]}")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"连接 {url} 失败：{e}")


def main():
    ap = argparse.ArgumentParser(description="阿里云百炼 Wan 图生视频")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--image", required=True, help="首帧图片（Wan 图生视频必须有首帧）")
    ap.add_argument("--end-image", help="尾帧图片")
    ap.add_argument("--negative", help="负面提示词（最多 500 字）")
    ap.add_argument("--duration", type=int, default=5, help="秒数 2–15")
    ap.add_argument("--resolution", default="720P", choices=["720P", "1080P"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="地域域名，需和密钥、模型同一地域")
    ap.add_argument("--key-env", default="DASHSCOPE_API_KEY")
    ap.add_argument("--prompt-extend", action="store_true", help="让服务端用大模型改写提示词（默认关）")
    ap.add_argument("--timeout", type=int, default=900, help="最长等待秒数")
    a = ap.parse_args()

    key = os.environ.get(a.key_env)
    if not key:
        sys.exit(f"没有设置环境变量 {a.key_env}。先 export {a.key_env}=你的百炼 API Key")
    media = [{"type": "first_frame", "url": data_url(a.image)}]
    if a.end_image:
        media.append({"type": "last_frame", "url": data_url(a.end_image)})
    params = {"resolution": a.resolution, "duration": max(2, min(15, a.duration)),
              "prompt_extend": a.prompt_extend, "watermark": False}
    if a.seed is not None:
        params["seed"] = a.seed
    inp = {"prompt": a.prompt, "media": media}
    if a.negative:
        inp["negative_prompt"] = a.negative[:500]
    base = a.base_url.rstrip("/")

    t0 = time.time()
    r = request(f"{base}/api/v1/services/aigc/video-generation/video-synthesis", key,
                {"model": a.model, "input": inp, "parameters": params})
    task = (r.get("output") or {}).get("task_id")
    if not task:
        sys.exit(f"提交失败：{json.dumps(r, ensure_ascii=False)[:400]}")
    print(f"已提交任务 {task}（{a.model}，{params['duration']} 秒，{a.resolution}）", flush=True)

    while True:
        time.sleep(15)  # 文档建议约 15 秒查询一次
        r = request(f"{base}/api/v1/tasks/{task}", key)
        out = r.get("output") or {}
        status = out.get("task_status")
        print(f"  {time.time() - t0:.0f}s 状态 {status}", flush=True)
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            sys.exit(f"任务 {status}：{out.get('code', '')} {out.get('message', '')}")
        if time.time() - t0 > a.timeout:
            sys.exit(f"等待超过 {a.timeout} 秒，任务 {task} 仍未完成（24 小时内可以用任务号再查）")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    urllib.request.urlretrieve(out["video_url"], a.output)  # 视频链接 24 小时后失效，马上下载
    usage = r.get("usage") or {}
    print(f"-> {a.output}（用时 {time.time() - t0:.0f}s，计费时长 {usage.get('output_video_duration', '?')} 秒）")


if __name__ == "__main__":
    main()
