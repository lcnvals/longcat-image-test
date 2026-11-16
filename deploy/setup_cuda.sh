#!/usr/bin/env bash
# Linux + NVIDIA 一键部署：环境检查 → 安装依赖 → 下载模型 → 自检。
#
# 用法（在仓库根目录运行）：
#   bash deploy/setup_cuda.sh                                  # 默认：Z-Image-Turbo + LTX-2，魔搭下载
#   bash deploy/setup_cuda.sh --models z-image-turbo,flux2-klein-4b,longcat,ltx-2
#   bash deploy/setup_cuda.sh --source hf                       # 从 Hugging Face 下载（海外服务器）
#   bash deploy/setup_cuda.sh --pip-mirror                      # pip 用清华镜像（国内服务器）
#   bash deploy/setup_cuda.sh --skip-models                     # 只装环境，不下载模型
#   bash deploy/setup_cuda.sh --check                           # 只做环境检查和自检
#   DRY_RUN=1 bash deploy/setup_cuda.sh                         # 只打印将要执行的安装/下载命令
#
# 可重复运行：已装好的依赖和已下载完整的模型会自动跳过（下载工具支持断点续传）。
# 注意：作者没有 NVIDIA 机器，这个脚本尚未在真实服务器上跑过，遇到问题请把输出发回来。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
VENV="$ROOT/.venv-cuda"
MODELS="z-image-turbo,ltx-2"
SOURCE="modelscope"
PIP_MIRROR=""
SKIP_MODELS=0
CHECK_ONLY=0
DRY="${DRY_RUN:-0}"
run() { if [ "$DRY" = 1 ]; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi; }

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --pip-mirror) PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "未知参数：$1（--help 查看用法）"; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✘ %s\033[0m\n' "$*"; exit 1; }

# 各模型：目录、魔搭/HF 仓库、下载大小（GB，约）、只下载哪些文件
model_dir()  { case "$1" in longcat) echo "$ROOT/model-t2i" ;; *) echo "$ROOT/models/$1" ;; esac; }
model_repo() {
  case "$1" in
    z-image-turbo)  echo "Tongyi-MAI/Z-Image-Turbo" ;;
    flux2-klein-4b) echo "black-forest-labs/FLUX.2-klein-4B" ;;
    longcat)        echo "meituan-longcat/LongCat-Image" ;;
    ltx-2)          echo "Lightricks/LTX-2" ;;
    *) die "未知模型：$1（可选 z-image-turbo、flux2-klein-4b、longcat、ltx-2）" ;;
  esac
}
model_size() { case "$1" in z-image-turbo) echo 33 ;; flux2-klein-4b) echo 16 ;; longcat) echo 30 ;; ltx-2) echo 95 ;; esac; }
# LTX-2 仓库有 314GB（含多种单文件权重和重复的文本编码器），只下 diffusers 需要的部分
model_include() {
  case "$1" in
    ltx-2) echo "model_index.json scheduler/* transformer/* text_encoder/model* text_encoder/*.json tokenizer/* vae/* audio_vae/* vocoder/* connectors/*" ;;
    *) echo "" ;;
  esac
}
model_exclude() { case "$1" in flux2-klein-4b) echo "flux-2-klein-4b.safetensors" ;; *) echo "" ;; esac; }

# 先校验模型名，避免下载到一半才发现写错
for m in ${MODELS//,/ }; do
  case "$m" in z-image-turbo|flux2-klein-4b|longcat|ltx-2) ;; *) echo "未知模型：$m（可选 z-image-turbo、flux2-klein-4b、longcat、ltx-2）"; exit 1 ;; esac
done

# ---------------------------------------------------------------- 1. 环境检查
say "1/4 环境检查"
[ "$(uname -s)" = "Linux" ] || die "这个脚本用于 Linux + NVIDIA 服务器；Mac 请按 README 用 .venv / .venv-mlx"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi：没有 NVIDIA 驱动，或不是 GPU 实例"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$(((VRAM + 512) / 1024))   # MiB -> GB，四舍五入
ok "GPU：$GPU，显存 ${VRAM_GB}GB，驱动 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if   [ "$VRAM_GB" -ge 80 ]; then ok "视频（LTX-2）可以全部放进显存，最快"
elif [ "$VRAM_GB" -ge 40 ]; then ok "视频（LTX-2）会按组件在显存和内存之间搬运，需要约 64GB 内存"
else warn "显存 ${VRAM_GB}GB < 40GB：生图没问题；视频（LTX-2 19B）只能逐层搬运，会非常慢。做视频建议换 48GB 以上的卡（L40S / A6000 / A100 / H100）"
fi

RAM_GB=$( [ -r /proc/meminfo ] && awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo || echo 0)
if [ "$RAM_GB" -ge 64 ]; then ok "内存 ${RAM_GB}GB"; else warn "内存 ${RAM_GB}GB：视频按组件搬运时建议 64GB 以上"; fi

PY=$(command -v python3 || true)
[ -n "$PY" ] || die "找不到 python3"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "需要 Python 3.10 以上，当前 $("$PY" -V)"
ok "$("$PY" -V)"

NEED=0
if [ "$SKIP_MODELS" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
  for m in ${MODELS//,/ }; do [ -d "$(model_dir "$m")" ] || NEED=$((NEED + $(model_size "$m"))); done
fi
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt $((NEED + 20)) ]; then
  die "磁盘剩余 ${FREE_GB}GB，下载模型约需 ${NEED}GB（另留 20GB 给环境）。云服务器通常要把仓库放到数据盘上"
fi
ok "磁盘剩余 ${FREE_GB}GB（模型约需 ${NEED}GB）"
command -v ffmpeg >/dev/null && ok "ffmpeg 已安装" || warn "没有系统 ffmpeg：拼接视频会用 imageio-ffmpeg 自带的版本；也可以 apt install -y ffmpeg"

# ---------------------------------------------------------------- 2. 安装依赖
if [ "$CHECK_ONLY" -eq 0 ]; then
  say "2/4 安装依赖到 .venv-cuda"
  [ -x "$VENV/bin/python" ] || run "$PY" -m venv "$VENV"
  PIP=("$VENV/bin/pip" install -q)
  [ -n "$PIP_MIRROR" ] && PIP+=(-i "$PIP_MIRROR")
  run "${PIP[@]}" --upgrade pip
  # diffusers 从 GitHub 按 commit 安装；国内服务器访问 GitHub 慢时可以先配置代理
  run "${PIP[@]}" -r requirements-cuda.txt
  ok "依赖安装完成"
fi

# ---------------------------------------------------------------- 3. 下载模型
if [ "$CHECK_ONLY" -eq 0 ] && [ "$SKIP_MODELS" -eq 0 ]; then
  say "3/4 下载模型（来源：$SOURCE）"
  for m in ${MODELS//,/ }; do
    dir=$(model_dir "$m"); repo=$(model_repo "$m")
    read -r -a inc <<< "$(model_include "$m")"
    read -r -a exc <<< "$(model_exclude "$m")"
    echo "  -> $m（$repo，约 $(model_size "$m")GB）到 ${dir#"$ROOT"/}"
    if [ "$SOURCE" = "hf" ]; then
      args=("$VENV/bin/hf" download "$repo" --local-dir "$dir")
      # hf 的 --include/--exclude 每次只接一个模式（多余的值会被当成文件名），所以逐个加
      for p in ${inc[@]+"${inc[@]}"}; do args+=(--include "$p"); done
      for p in ${exc[@]+"${exc[@]}"}; do args+=(--exclude "$p"); done
    else
      args=("$VENV/bin/modelscope" download --model "$repo" --local-dir "$dir" --max-workers 8)
      [ ${#inc[@]} -gt 0 ] && args+=(--include "${inc[@]}")
      [ ${#exc[@]} -gt 0 ] && args+=(--exclude "${exc[@]}")
    fi
    run "${args[@]}" || die "$m 下载失败；重新运行本脚本会断点续传"
    [ "$DRY" = 1 ] || ok "$m 完成（$(du -sh "$dir" | cut -f1)）"
  done
fi

# ---------------------------------------------------------------- 4. 自检
say "4/4 自检"
[ "$DRY" = 1 ] && { echo "  [dry-run] 跳过自检"; exit 0; }
[ -x "$VENV/bin/python" ] || die "还没有安装环境：去掉 --check 重新运行"
"$VENV/bin/python" - <<'EOF'
import importlib, os, shutil
import torch
ok = lambda s: print(f"  \033[32m✔\033[0m {s}")
bad = lambda s: print(f"  \033[31m✘\033[0m {s}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    ok(f"PyTorch {torch.__version__}，CUDA {torch.version.cuda}，{p.name} {p.total_memory / 1e9:.0f}GB")
else:
    bad("torch.cuda 不可用：检查驱动版本和 torch 的 CUDA 版本是否匹配")
d = importlib.import_module("diffusers")
for name in ("ZImagePipeline", "Flux2KleinPipeline", "LongCatImagePipeline", "LTX2Pipeline", "LTX2ImageToVideoPipeline"):
    (ok if hasattr(d, name) else bad)(f"diffusers {d.__version__} · {name}")
for key, path in (("z-image-turbo", "models/z-image-turbo"), ("flux2-klein-4b", "models/flux2-klein-4b"),
                  ("longcat", "model-t2i"), ("ltx-2", "models/ltx-2")):
    if os.path.exists(os.path.join(path, "model_index.json")):
        ok(f"模型 {key}：{path}")
    else:
        print(f"  · 模型 {key}：未下载（需要时用 --models {key}）")
try:
    import imageio_ffmpeg
    ok(f"ffmpeg：{shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()}")
except ImportError:
    bad("没有 ffmpeg")
EOF

say "完成。快速试跑："
cat <<'EOF'
  # 生图（交互）
  .venv-cuda/bin/python studio.py --model z-image-turbo
  # 视频：小尺寸试一段，确认能跑通
  .venv-cuda/bin/python video.py gen "一只橘猫坐在窗台上，慢慢转头看向窗外" --ratio 小横屏
  # 分镜批量：本地 Mac 上准备好 projects/<片名>/（storyboard.json + 关键帧），上传到服务器后
  .venv-cuda/bin/python video.py board projects/<片名>/storyboard.json --concat
EOF
