# LongCat-Image on Apple Silicon

在 Apple Silicon Mac 上本地运行文生图模型的交互工作台。最初为美团 [LongCat-Image](https://github.com/meituan-longcat/LongCat-Image)（6B）编写，现在同时支持 MLX 原生的 Z-Image-Turbo 和 FLUX.2 klein 4B，并可接入大模型 API 处理提示词。

LongCat 的官方示例针对 CUDA，并用 `enable_model_cpu_offload()` 节省显存；Mac 是统一内存，这个办法省不了内存。本仓库改用 MPS，并通过分段加载 / int8 量化让模型在 32GB 内存的 Mac 上跑起来。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `studio.py` | 常驻交互模式：模型只加载一次，先低步数预览、再出正式图；带画幅、镜头、光线、风格等预设 | 已在 M1 Pro 32GB 上测试 |
| `t2i.py` | 单次文生图，文本编码器和 transformer 分段加载 | 已在 M1 Pro 32GB 上测试 |
| `edit.py` | LongCat-Image-Edit-Turbo 改图（照片 + 指令） | 未实测 |
| `presets.py` | `studio.py`、`video.py` 共用的提示词预设，可以自己加 | — |
| `menu.py` | 命令补全菜单（studio 和视频参数助手共用） | — |
| `assets.py` | 角色/怪物/场景定义与设定图 | 实测过 |
| `postprod.py` | 漫剧镜头、配音、字幕、音效、背景声与混音 | 实测过 |
| `prompttools.py` | 提示词清理与批量导入解析 | 实测过 |
| `video_api.py` | 商用视频 API（阿里云百炼 Wan） | 模拟接口测过，未用真实密钥 |
| `backends.py` | 模型后端：longcat（PyTorch）、z-image-turbo、flux2-klein-4b（mflux/MLX） | 三个都实测过 |
| `video.py`、`video.json` | 本地视频生成（LTX-2.3）：单镜头和分镜批量，图生视频、运镜预设、拼接成片 | 实测过（video 分支） |
| `deploy/setup_cuda.sh`、`video_cuda.py` | Linux + NVIDIA 一键部署与 CUDA 视频运行器 | 未在真实 GPU 上测试 |
| `ai.py`、`ai.json` | AI 提示词工具：调用 OpenAI 兼容接口扩写、翻译、看图反推等，可自定义 | 用模拟接口测过，真实接口未测 |

## 模型选择（M1 Pro 32GB 实测）

1024×1024，种子 43，mflux 模型 8-bit 量化。时间为单张出图耗时（不含首次加载）。

| 模型 | `--model` | 单张耗时 | 内存峰值 | 中文文字渲染 | 手部 | 许可 |
|---|---|---|---|---|---|---|
| FLUX.2 klein 4B | `flux2-klein-4b` | 约 57–61 秒（4 步）；512×512 约 18 秒 | 约 9GB（精修 13GB） | ❌ 乱码 | 少一根手指 | Apache 2.0 |
| Z-Image-Turbo | `z-image-turbo` | 约 190 秒（9 步） | 约 12GB（精修 19GB） | ✅ “深夜食堂”“营业中”完全正确 | ✅ 五指正确 | Apache 2.0 |
| LongCat-Image | `longcat`（默认） | 512×512、20 步约 3.5 分钟 | 约 21GB 常驻 | 未系统测试 | 8 步预览时常出错 | Apache 2.0 |

- **要中文文字（海报、招牌）或手部**：选 Z-Image-Turbo，最稳但最慢。
- **要快速出图、试构图**：选 FLUX.2 klein 4B，画质好，但中文文字不行。
- **24GB 的 Mac（如 M4 Mac mini）**：优先用两个 MLX 模型。Z-Image 精修时峰值 19GB 偏紧，可以加 `--quantize 4` 降低内存（未实测画质）。
- 蒸馏模型（FLUX.2 klein、Z-Image-Turbo）**不支持 CFG 和负面提示词**，`/cfg`、`/neg` 会提示不生效。
- 两个蒸馏模型本身就是满步数出图，`/refine` 能保留结构，但细节提升不明显。

## 硬件要求

模型权重共约 29GB：文本编码器（Qwen2.5-VL）16.6GB + transformer 12.5GB + VAE 0.17GB。

| 内存 | 建议 |
|---|---|
| 16GB | 不支持 |
| 24GB | `studio.py --quant-transformer`，或 `t2i.py --te-device cpu`（未实测） |
| 32GB | `studio.py` 或 `t2i.py` 默认参数 |
| 48GB 以上 | 都可以，余量充足 |

磁盘至少留 35GB（模型 29GB + Python 环境）。

## 安装

需要 Python 3.10 以上。

```bash
git clone git@github.com:lcnvals/longcat-image-test.git
cd longcat-image-test
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` 把 diffusers 锁定在一个包含 LongCat 管线的 git commit 上，这个版本需要 transformers 5.x。

使用 Z-Image-Turbo / FLUX.2 klein 时另建一个 MLX 环境（和上面的 PyTorch 环境依赖不兼容，分开装）：

```bash
python3 -m venv .venv-mlx
.venv-mlx/bin/pip install -r requirements-mlx.txt
```

## 下载模型

国内从魔搭（ModelScope）直连最快（实测总速度约 28MB/s，约 20 分钟下完）。从 Hugging Face 走代理时，大文件 CDN 只有几十 KB/s。

```bash
# 开着代理的话，先去掉代理变量
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  .venv/bin/modelscope download --model meituan-longcat/LongCat-Image \
  --local-dir ./model-t2i --max-workers 8
```

改图模型（`edit.py` 用）：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  .venv/bin/modelscope download --model meituan-longcat/LongCat-Image-Edit-Turbo \
  --local-dir ./model --max-workers 8
```

MLX 模型（`studio.py --model` 用，放在 `models/` 下）：

```bash
# zsh 里先清掉代理变量
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
# Z-Image-Turbo，约 33GB（transformer 是 fp32，加载时量化）
.venv-mlx/bin/modelscope download --model Tongyi-MAI/Z-Image-Turbo \
  --local-dir ./models/z-image-turbo --max-workers 8
# FLUX.2 klein 4B，约 16GB（排除重复的单文件权重）
.venv-mlx/bin/modelscope download --model black-forest-labs/FLUX.2-klein-4B \
  --local-dir ./models/flux2-klein-4b --exclude 'flux-2-klein-4b.safetensors' --max-workers 8
```

## 使用

### studio.py：常驻交互模式（推荐）

用 `--model` 选模型，不同模型要用对应的虚拟环境：

```bash
.venv/bin/python studio.py                             # LongCat-Image（默认）
.venv-mlx/bin/python studio.py --model z-image-turbo   # Z-Image-Turbo
.venv-mlx/bin/python studio.py --model flux2-klein-4b  # FLUX.2 klein 4B
```

每个模型的默认尺寸和步数不同：LongCat 768×768、预览 8 步、正式 25 步；MLX 模型 1024×1024，FLUX.2 klein 4 步，Z-Image 9 步。各模型的图片和历史分开保存：LongCat 在 `outputs/`，其他在 `outputs/<模型名>/`。

```bash
.venv/bin/python studio.py --size 768x768
```

启动时加载一次模型（约 1 分钟），出现 `longcat>` 提示符后。

输入时有**命令菜单**：

- 输入 `/` 会弹出全部命令和说明；继续输入可以筛选，比如 `/st` 只剩 `/style`、`/steps`。
- 选中命令后会接着弹出参数菜单，比如 `/style ` 后面列出全部风格和对应的描述。
- 数值类命令（`/final`、`/steps`、`/cfg`、`/size`、`/batch`、`/seed`）会列出常用值和说明，也可以直接输入其他数字。
- 用 ↑↓ 选择，按回车或 Tab 确认。选中命令时回车只填入命令；选中参数时回车直接执行。
- ↑↓ 在没有菜单时用来翻历史输入，历史保存在 `outputs/.history`。

基本流程：


```
longcat> 一位中国年轻女性在厨房做饭，写实摄影     ← 直接输入提示词：8 步预览，自动打开
longcat> /batch 4                              ← 同一提示词换 4 个种子（43–46）各出一张预览
longcat> /seed 45                              ← 选中 45 号
longcat> /refine                               ← 以 45 号预览图为底稿精修（推荐，结构不变）
longcat> /final                                ← 或者：同种子从头重新生成 25 步（结构可能变）
longcat> /quit
```

| 命令 | 作用 |
|---|---|
| 直接输入文字 | 用预览步数出图 |
| `/final [步数]` | 用上一条提示词、同种子从头重新生成正式图（默认 25 步）；手指等细碎结构可能和预览不同 |
| `/refine [强度] [步数] [文件名]` | **以预览图为底稿精修**：保留构图和手指结构，只补细节（强度默认 0.5） |
| `/batch N` | 上一条提示词连换 N 个种子出预览 |
| `/seed N`、`/seed +` | 设种子 / 种子加 1 |
| `/size WxH` | 设尺寸 |
| `/steps N` | 设预览步数 |
| `/cfg X` | 设 guidance_scale（默认 4.0） |
| `/ratio [名称]` | 画幅预设，会改尺寸 |
| `/shot` `/light` `/tone` `/time` `/detail` `/style` | 镜头、光线、色调、时间天气、细节质感、风格预设（见下文） |
| `/neg [预设名\|文本\|off]` | 负面提示词：用预设（通用/人像/手部/风景/插画）或自定义文本；不带参数列出预设；`off` 清除 |
| `/reset` | 清空所有追加预设和负面提示词 |
| `/open on\|off` | 出图后是否自动用“预览”打开 |
| `/show`、`/help` | 查看当前设置和实际提示词 / 帮助 |

#### 提示词预设

模型没有风格、构图之类的参数，这些都靠提示词描述来控制。`studio.py` 把常用描述做成了预设，选中的预设会按“镜头，光线，色调，时间天气，细节，风格”的顺序自动接到提示词后面，你只需要写画面内容：

```
longcat> /ratio 竖图
longcat> /shot 半身
longcat> /light 窗光
longcat> /style 人像
longcat> /neg 人像
longcat> 一位中国年轻女性在咖啡馆看书
         → 实际提示词：一位中国年轻女性在咖啡馆看书，半身构图，窗边自然光，柔和侧光，专业人像摄影，背景虚化，皮肤质感细腻
```

每类命令不带参数会列出全部选项和对应的描述，带 `off` 就是取消这一类。`/show` 可以随时查看实际拼接出来的提示词。

| 命令 | 类别 | 可选项 |
|---|---|---|
| `/ratio` | 画幅 | 方图 768×768、竖图 672×896、壁纸 576×1024、横图 896×672、宽屏 1024×576、小方图 512×512 |
| `/shot` | 镜头构图 | 特写、半身、全身、远景、俯拍、仰拍、航拍、微距、广角、对称、三分法 |
| `/light` | 光线 | 黄金时刻、蓝调时刻、逆光、剪影、棚拍、窗光、霓虹、烛光、阴天、侧光 |
| `/tone` | 色调氛围 | 暖色、冷色、黑白、莫兰迪、鲜艳、复古、温馨、静谧、神秘、热闹、孤独 |
| `/time` | 时间天气 | 清晨、正午、黄昏、夜晚、晴天、雨天、雪天、雾天、樱花、秋叶 |
| `/detail` | 细节质感（**可多选**） | 毛发、皮肤、布料、材质、锐利、虚化、光影、高清、相机、微观 |
| `/style` | 风格 | 写实、电影、胶片、人像、动漫、治愈动画、3D卡通、水彩、油画、水墨、国潮、素描、扁平插画、像素、赛博朋克、蒸汽波 |
| `/neg` | 负面提示词 | 通用、人像、手部、风景、插画 |

注意事项：

- 画幅预设的面积都在 0.4–0.6MP 之间，速度和 768×768 差不多；`小方图` 最快，适合试提示词。
- `/detail` 可多选：`/detail 毛发`、`/detail 锐利` 会同时生效，再选一次同一项就是取消；`/detail off` 清空。描述要和画面主体对应（画猫选毛发，画人选皮肤），无关的质感模型可能硬加进画面。
- 细节描述只是让模型往这个方向画，能画多细还取决于分辨率和步数：512×512、8 步很难画出根根分明的毛发，正式图建议 1024×1024 以上、25–30 步。
- 提示词里自己写了某类描述（比如“黄昏”），就把对应预设关掉，免得两种描述冲突。
- 负面提示词靠 CFG 起作用（`/cfg 1` 时无效）。各个预设的实际效果还没逐一实测。
- 预设都在 `presets.py` 里，想加自己的，在对应字典里加一项就行。

启动时也能直接指定：

```bash
.venv/bin/python studio.py --ratio 竖图 --shot 半身 --light 窗光 --style 人像 --neg 人像
```

启动参数：`--model`、`--size`（默认按模型）、`--ratio`（会覆盖 `--size`）、`--preview`（默认 8）、`--final`（默认 25）、`--seed`（默认 43）、`--shot`、`--light`、`--tone`、`--time`、`--detail`（可多个，如 `--detail 毛发 锐利`）、`--style`、`--neg`、`--quant-transformer`（LongCat 的 transformer 也量化为 int8，24GB 机器用，未实测）、`--quantize`（MLX 模型量化位数，默认 8）、`--ai-provider`、`--ai-auto`。

图片保存在 `outputs/`，文件名带编号、种子、步数、尺寸，比如 `003_s45_25st_768x768.png`。每张图的提示词记录在 `outputs/log.txt`。

**内存方案**：文本编码器用 optimum-quanto 量化为 int8（16.6GB 降到约 8.5GB），和 bf16 的 transformer 一起常驻 MPS，合计约 21GB。实测量化后的出图和 bf16 几乎一致。

#### AI 提示词工具

用大模型 API 处理提示词。服务商和工具都在配置文件里，可以自己加。

```
/ai                    列出所有工具和当前服务商
/ai 扩写                扩写上一条提示词（也可以 /ai 扩写 一只猫）
/use                   采用 AI 结果作为新提示词并出预览
/ai 负面                根据画面生成针对性的负面提示词，直接生效
/ai 检查                检查提示词里的冲突和歧义，给出修改建议
/ai 看图 照片.png        看图反推提示词（需要支持视觉的模型）
/ai 精简                去掉重复和冲突的描述
/ai auto 英译           之后每次出图前自动翻译成英文（给英文更强的模型用）
/ai provider deepseek  切换服务商
```

内置服务商（都走 OpenAI 兼容的 `/chat/completions` 接口）：

| 服务商 | 模型 | 密钥环境变量 |
|---|---|---|
| `longcat`（默认） | LongCat-2.5，支持看图 | `LONGCAT_API_KEY`（在 https://longcat.chat/platform/api_keys 申请） |
| `deepseek` | deepseek-chat | `DEEPSEEK_API_KEY` |
| `qwen` | qwen-plus | `DASHSCOPE_API_KEY` |
| `ollama` | 本地模型，不需要密钥 | — |

密钥只从环境变量读取，不写进任何文件：

```bash
export LONGCAT_API_KEY=你的密钥
.venv-mlx/bin/python studio.py --model z-image-turbo --ai-auto 扩写
```

LongCat 官方 API 目前只提供 LongCat-2.5（对话和看图），没有图像生成接口，所以出图仍在本地。

**自定义**：新建 `ai.local.json`（不进仓库），同名项覆盖 `ai.json`。比如加一个服务商和一个工具：

```json
{
  "providers": {
    "my-llm": {"base_url": "https://example.com/v1", "model": "模型名", "api_key_env": "MY_LLM_KEY"}
  },
  "tools": {
    "小红书": {
      "desc": "改写成小红书封面风格的画面描述",
      "target": "prompt",
      "system": "把用户描述改写成适合小红书封面的画面：明亮清新、主体居中、留出标题空间。只输出提示词。"
    }
  }
}
```

工具字段：`target` 决定结果去向（`prompt` 新提示词、`neg` 负面提示词、`show` 只显示）；`input: "image"` 表示要传图片；`provider` 可以给单个工具指定服务商；`user` 是用户消息模板，`{text}` 会替换成提示词。

### t2i.py：单次出图

```bash
.venv/bin/python t2i.py '一只橘猫坐在窗台上，写实摄影风格' -o out.png \
  --width 768 --height 768 --steps 25 --seed 43
```

参数：`--negative`、`--cfg`（默认 4.0）、`--rewrite`（先用文本编码器扩写提示词，较慢）、`--te-device cpu`（文本编码器放 CPU，24GB 机器用）。

每次运行都会先加载文本编码器、编码完释放，再加载 transformer 出图，峰值内存约 17GB。适合偶尔出一张，或者写进批处理脚本。

### edit.py：改图（未实测）

```bash
.venv/bin/python edit.py 照片.jpg '把这张照片转换成日系动漫风格，保持人物发型、服装和姿势不变' -o anime.png
```

Edit-Turbo 是蒸馏模型，默认 8 步，不使用 CFG。

## 视频生成（video.py，实验中）

用 LTX-2.3 蒸馏版（MLX 4-bit）在本地生成短视频。和 `studio.py` 共用预设和 AI 提示词工具，但**单独运行**：视频模型约占 15GB 内存，不和生图模型同时加载。推荐流程是先用 studio 出关键帧，退出后再用 `video.py` 让它动起来。

### 安装

LTX 装在单独的目录和虚拟环境里（依赖和生图环境不兼容），路径写在 `video.json`；本机路径不同时，新建 `video.local.json` 覆盖（不进仓库）。

```bash
mkdir -p video-lab && cd video-lab
git clone https://github.com/dgrauet/ltx-2-mlx
git -C ltx-2-mlx checkout 895201df379b079c2154738bcf92005b7a371923   # 24GB 实测用的版本
python3 -m venv ltx-mlx
ltx-mlx/bin/pip install -e ltx-2-mlx/packages/ltx-core-mlx -e ltx-2-mlx/packages/ltx-pipelines-mlx av imageio-ffmpeg psutil
# 模型约 20.5GB（HF），文本编码器 Gemma 约 8GB（魔搭）
# hf 的 --include 每次只接一个模式，多个模式要各写一个（否则多余的会被当成文件名，.json 会漏下）
# audio_vae / vocoder 是 retake、extend 必需的音频解码器
hf download dgrauet/ltx-2.3-mlx-q4 --include "*.json" --include "connector.safetensors" \
  --include "transformer-distilled-1.1.safetensors" --include "spatial_upscaler_x2_v1_1.safetensors" \
  --include "vae_decoder.safetensors" --include "vae_encoder.safetensors" \
  --include "audio_vae.safetensors" --include "vocoder.safetensors" --local-dir ltx-model
modelscope download --model mlx-community/gemma-3-12b-it-4bit --local-dir gemma
```

拼接成片需要 ffmpeg：`brew install ffmpeg`。

### 参数助手（熟悉参数用）

不确定有哪些参数、每个选项会往提示词里加什么，可以先用助手：

```bash
.venv-mlx/bin/python video.py assist
```

```
video> /camera              ← 不带参数：列出所有运镜和对应的描述
video> /camera 推进          ← 选择后自动显示当前参数、实际提示词和等价命令行
video> /style 电影
video> 橘猫在窗台上伸懒腰      ← 直接输入一句话 = 设为提示词（写动作）
video> /image                ← 列出最近生成的 10 张图，可直接选作首帧
video> /show                 ← 查看实际提示词和等价命令行（可以复制下来直接用）
video> /tips                 ← 视频提示词写法建议
video> /save 猫咖啡店 S03     ← 存进分镜（关键帧会复制进片目录）
video> /dry                  ← 打印将要执行的 LTX 命令
video> /gen                  ← 按当前参数生成
```

和 studio 一样，输入 `/` 会弹出命令菜单，选中命令后自动弹出参数选项，每一项都附带说明。助手主要用来熟悉参数、拼出命令，熟悉之后直接用下面的命令行或分镜文件更方便。

### 单个镜头

```bash
.venv-mlx/bin/python video.py gen "橘猫慢慢眨眼，尾巴轻轻摆动" --image cat.png --camera 推进 --motion 轻微
.venv-mlx/bin/python video.py gen "..." --seed 42 43 44            # 一次出多个种子挑选
.venv-mlx/bin/python video.py gen "..." --image a.png --end-image b.png   # 锁定首尾帧
.venv-mlx/bin/python video.py gen "..." --dry-run                  # 只打印命令
```

| 参数 | 说明 |
|---|---|
| `--image` / `--end-image` | 首帧 / 尾帧关键帧（图生视频），会自动居中裁成视频比例 |
| `--image-strength` | 首帧锁定强度 0–1，默认 1.0 |
| `--seed 42 43 …` | 一个或多个种子 |
| `--seconds` / `--frames` / `--fps` | 时长（默认 2 秒、25fps）；帧数自动调整为 LTX 要求的 8n+1 |
| `--ratio` / `--size` | 画幅：横屏 768×512（默认）、宽屏 896×512、竖屏 512×768、方形 640×640、高清横屏 1152×768、小横屏 512×320；自定义尺寸须为 64 的倍数 |
| `--camera` | 运镜：固定、推进、拉远、左移、右移、上摇、环绕、跟拍、手持 |
| `--motion` | 动作幅度：轻微、自然、明显 |
| `--style` `--shot` `--light` `--tone` `--time` `--detail` | 和 studio 相同的提示词预设 |
| `--neg` | 负面提示词（预设名或文本，通过 NAG 生效） |
| `--ai 扩写` / `--ai 英译` | 生成前用 AI 工具处理提示词 |
| `--enhance` | 用 LTX 自带的 Gemma 扩写提示词 |
| `--audio` | 同时生成音频（默认关闭） |
| `--no-low-ram` | 关闭低内存模式（内存充足时更快） |

输出在 `outputs/video/`，每个视频旁边有同名 `.json`（完整参数）和 `.log`。参数完全相同的视频会直接跳过，不重复生成。

### 分镜（批量）

```bash
.venv-mlx/bin/python video.py init projects/猫咖啡店                          # 生成分镜模板
.venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json          # 生成全部镜头
.venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --only S03  # 只重做 S03
.venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --concat    # 拼接成片 final.mp4
```

`storyboard.json` 由全片设置 `settings` 和镜头列表 `shots` 组成：镜头里的字段会覆盖全片设置；`{character}`、`{scene}` 会替换成设定里的角色和场景描述（保证每个镜头的描述一字不差）；`image` 是关键帧路径（相对分镜文件）；`pick` 指定拼接时用哪个种子。每个镜头输出到 `projects/<片名>/<镜号>/`。改了某个镜头只需要重跑它，没改的镜头会自动跳过。

### 定义：角色、怪物、场景（assets.py）

角色、怪物、场景定义一次，所有镜头和项目都引用。每个定义 = 外观描述 + 设定图 + 音色：

```bash
.venv-mlx/bin/python assets.py new 红带武士 --kind character --style 游戏CG \
  --desc "一名年轻武士，黑色短发，额头系着红色发带，身穿白色内衬和深蓝色轻甲，手持一把长太刀" \
  --must 红色发带 深蓝色轻甲 --voice Tingting
.venv-mlx/bin/python assets.py sheet 红带武士      # 按定义生成设定图：角色是正面/侧面/背面/战斗姿态
.venv-mlx/bin/python assets.py list                 # 列出所有定义
.venv-mlx/bin/python assets.py show 红带武士        # 查看描述、必须特征、音色、设定图
```

定义存在 `assets/<类别>/<名字>/definition.json`（类别：characters、monsters、scenes、props）：

| 字段 | 作用 |
|---|---|
| `desc` | 外观描述，每个镜头原样引用 |
| `must` | 必须保持的特征，会追加到描述后面 |
| `avoid` | 不该出现的特征；**只在支持负面提示词的模型里生效**。在正面提示词里写“不要出现四肢”反而会画出四肢，所以不支持负面提示词的模型请在 `desc` 里用正面描述（如“蛇形身躯盘绕在地面上”） |
| `voice` | 这个角色说台词用的音色（见下文“声音”） |
| `display_name` | 字幕里显示的名字 |
| `sheets` | 要生成哪些设定图、各自的画面要求和尺寸 |

分镜里用 `cast` 引用，镜头描述里写 `{hero}` 等占位符：

```json
"settings": {"cast": {"hero": "红带武士", "monster": "八岐大蛇", "scene": "火山峡谷"}, "ref_images": 1},
"shots": [{"id": "S04", "keyframe": "{scene}，{hero}挥刀砍在{monster}的鳞片上", "prompt": "太刀砍在鳞片上迸出火花"}]
```

`video.py keyframes` 出关键帧时，会自动把这个镜头里出场的定义的设定图作为参考（每个定义取 `ref_images` 张），用支持参考图的 FLUX.2 klein 生成，外观更一致；`--no-refs` 关闭。

### 从网页版或其他工具导入提示词

已经有现成的提示词（网页版、其他 AI、自己写的），可以直接用：

- **studio**：直接粘贴回车即可。会自动去掉本工具不认识的语法：Midjourney 参数（`--ar 16:9 --v 6`）、WebUI 权重写法（`(词:1.2)`、`((词))`、`[词]`）、LoRA 标签（`<lora:xx:0.8>`）、`BREAK`，并提示去掉了什么。注意已选的预设会追加在后面，粘贴完整提示词前可以先 `/reset`。
- **批量导入成分镜**：把提示词整理成文本，每行一个镜头：

```
# 镜号 | 画面（出关键帧） | 动作（出视频） | 运镜 | 其他参数 key=value
S01 | 暴雨夜的火山峡谷，八岐大蛇与武士对峙 | 蛇头缓缓摆动，雨水落下 | 推进
S02 | 蛇头张口扑来，武士准备闪避 | 武士向右翻滚躲开 | 固定 | seconds=3 seeds=42,43
| 胸口发光的逆鳞特写 | | 推进 | mode=still line=唯有胸口那片逆鳞……
```

```bash
.venv-mlx/bin/python video.py import-prompts 提示词.txt projects/八岐大蛇
```

- 镜号可以省略（自动编号），后面几列可以省略；空行和 `#` 开头的行忽略；全角竖线 `｜` 也可以。
- 已有分镜时按镜号**合并**（保留文本里没写的字段，如 `pick`、剪辑参数）；`--replace` 清空后重新导入。
- 运镜不是预设名称时，会并入动作描述并提示。`--no-clean` 关闭语法清理。
- 提示词本身没有格式要求，自然语言即可；想让画面出现文字时用引号括起来；不要写否定（“没有腿”反而容易画出腿）。

### 批量关键帧

镜头里写 `keyframe`（静态画面描述）后，可以批量出关键帧，不用在 studio 里一张张出：

```bash
.venv-mlx/bin/python video.py keyframes projects/八岐大蛇/storyboard.json                       # 缺的才出
.venv-mlx/bin/python video.py keyframes projects/八岐大蛇/storyboard.json --only S04 --seed 42 43  # 重出，多个种子
```

多个种子都会保存为 `<镜号>_kf_s<种子>.png`，第一个作为当前关键帧；想换就把 `<镜号>_kf_s43.png` 复制成 `<镜号>.png`。`keyframe` 写静态画面（长什么样），`prompt` 写动作（怎么动）。

### 镜头衔接

| 镜头字段 | 作用 | 适合 |
|---|---|---|
| `"start_from": "prev"` | 用上一镜（选用种子）的**最后一帧**作首帧 | 同一空间里的连续动作，如跃起 → 刺下 |
| `"end_to": "next"` | 用下一镜的关键帧作尾帧 | 让这一镜“走到”下一镜开头的画面 |

上一镜重新生成后，接力的镜头会被识别出来自动重新生成。

### 剪辑：节奏、转场、补帧

拼接成片时可以设置剪辑参数（只用 ffmpeg，十几秒完成，不用重新生成）：

| 镜头字段 | 作用 |
|---|---|
| `speed` | 变速倍数，几乎不动的镜头加速（如 1.6） |
| `trim` | `[开始秒, 结束秒]`，剪掉有瑕疵的部分 |
| `in` | 开头转场：`cut`（默认）、`flash`（闪白，适合打击瞬间，自动配打击音效）、`fade`（从黑场淡入） |
| `out` | 结尾转场：`cut`、`fade`（淡出到黑场） |

`settings.edit`：`fps` 输出帧率、`smooth` 补帧到的帧率（如 50，让动作更顺滑）、`fade_in` / `fade_out` 全片开头和结尾的淡入淡出秒数。成片文件名用 `--output` 指定。

### 漫剧镜头：静态图推拉

不是每个镜头都需要视频模型。镜头写 `"mode": "still"`，就用关键帧做推拉平移（几秒完成，视频镜头要几分钟），适合开场、空镜、对话、特写：

```json
{"id": "S01", "mode": "still", "camera": "推进", "line": "传说中，火山深处沉睡着八岐大蛇。"}
```

运镜：推进、拉远、左移、右移、上摇、固定、手持（轻微晃动）；`"shake": true` 加打击震动。有台词时镜头自动延长到念完。视频镜头、漫剧镜头、API 生成的视频可以混剪，拼接时统一缩放到分镜尺寸。

### 声音：配音、字幕、音效、背景声

| 镜头字段 | 作用 |
|---|---|
| `line` | 旁白或台词：合成配音并生成字幕 |
| `speaker` | 说话的角色（`cast` 里的键）：用该角色定义里的 `voice`，字幕前加角色名，如“武士：就是现在！” |
| `voice` | 单独指定这句的音色 |
| `sfx` | 音效：`impact`（打击）、`whoosh`（风声）或音频文件路径；闪白转场默认带 `impact` |

```json
"settings": {
  "tts": {"engine": "say", "voice": "Tingting", "rate": 185},
  "audio": {"bgm": "rain", "bgm_volume": 0.3, "subtitles": true, "subtitle_speaker": true}
}
```

- 配音目前用 macOS 自带的 `say`，普通话可用 `Tingting`。列表里的 `Eddy`、`Flo` 等音色如果没下载语音数据会输出空音频，程序会报错提示。更多音色在“系统设置 → 辅助功能 → 朗读内容 → 系统声音”里下载。
- `bgm` 可以是内置的 `rain`（程序生成的雨声），也可以是音乐文件路径（自动循环）。旁白出现时背景声会自动压低。
- 打击、风声和雨声都是程序生成的占位音效，换成真实素材效果更好。

### 商用视频 API（阿里云百炼 Wan）

需要量产或更强的动作效果时，可以把视频镜头交给云端的通义万相（Wan 2.7）图生视频：

```bash
export DASHSCOPE_API_KEY=你的百炼 API Key       # 百炼控制台创建；密钥、地域、模型要一致
.venv-mlx/bin/python video.py board projects/八岐大蛇/storyboard.json --only S08 --backend api
```

- 支持首帧、尾帧、负面提示词、种子；时长 2–15 秒（按镜头的 `seconds` 取整），分辨率在 `video.json` 的 `api` 段设置（默认 720P）。
- 提交任务后每 15 秒查询一次，完成后立刻下载（官方视频链接只保留 24 小时）。
- **按生成的视频秒数计费，失败不计费**，价格以百炼控制台为准。
- 其他地域（如新加坡）修改 `api.base_url`。可灵、海螺、即梦等暂未接入（官方文档未能读取到）。
- 以上在本地用模拟接口测试过，尚未用真实密钥调用。

### 局部修改：只重做一段 / 往后续写

任何 `video.py` 生成的视频（包括分镜里的镜头）都可以局部修改，原视频保留，结果另存为新文件：

```bash
# 只重做 0.8–1.6 秒这一段，其余部分不变
.venv-mlx/bin/python video.py retake 视频.mp4 --from 0.8 --to 1.6 -p "猫转头看向镜头"
# 往后续写 1 秒；不写 -p 就沿用原视频的提示词
.venv-mlx/bin/python video.py extend 视频.mp4 --seconds 1 -p "猫站起来，跳下窗台"
# 续写到指定的结尾画面
.venv-mlx/bin/python video.py extend 视频.mp4 --image 结尾.png
```

- retake 按 8 帧一组重做，实际范围会对齐到组边界，命令会打印实际覆盖的秒数（例如 0.8–1.6 秒实际是 0.68–1.60 秒）。
- 实测（512×320，约 2 秒）：retake 约 142 秒，区间外画面不变；extend 续写 1 秒约 117 秒，原有帧完整保留，接缝处看不出拼接。
- retake 改动动作的幅度比较保守；extend 沿用弱动作的提示词时续写段几乎静止，**续写时用 `-p` 写一个新的、明确的动作**。
- 只能往后续写（蒸馏模型不支持往前）。

### 从 studio 存关键帧到分镜

在 studio 里出到满意的图后，输入：

```
/keyframe 猫咖啡店 S02
```

会把最近一张图复制成 `projects/猫咖啡店/S02.png`，并写进 `projects/猫咖啡店/storyboard.json` 的 S02 镜头（分镜文件不存在就新建，同一镜号会覆盖关键帧）。镜头的 `prompt` 先复制图片提示词，**需要改成描述动作的句子**，然后退出 studio，运行：

```bash
.venv-mlx/bin/python video.py board projects/猫咖啡店/storyboard.json --only S02
```

关键帧图片被覆盖后，`video.py` 会根据图片内容识别出来并重新生成，不会误跳过。

### 实测（M1 Pro 32GB）

| 场景 | 耗时 | 内存峰值 |
|---|---|---|
| 768×512，49 帧（约 2 秒），文生视频 | 约 194 秒 | 约 15.5GB |
| 768×512，49 帧，图生视频（Z-Image 关键帧） | 约 197 秒 | 约 15.5GB |
| 512×320，49 帧，分镜里的单个镜头 | 约 110 秒 | — |

- 中文提示词能正确理解（橘猫、窗台、看窗外）。
- **图生视频的一致性很好**：首帧和关键帧完全一致，猫的花纹、窗户全程不变。文生视频的 2 秒内动作幅度很小。
- 资料显示 LTX-2.3 超过约 2 秒（97 帧以上）成功率明显下降，所以建议每个镜头 2 秒左右，靠分镜拼接成长视频。
- 24GB M4 Pro 上的公开实测：768×512、49 帧约 110–130 秒，峰值 14.6GB。基础版 M4 没有实测数据。
- 许可：LTX-2 社区许可，年收入 1000 万美元以下可免费商用。

## 在 NVIDIA 云服务器上部署（实验中）

Mac 上生成视频很慢（M1 Pro 上 2 秒视频约 3 分钟）。可以在 Mac 上调好分镜和关键帧，把视频这类批量任务交给云上的 NVIDIA GPU。

> 作者没有 NVIDIA 机器，以下脚本和 CUDA 代码**尚未在真实服务器上测试**。遇到问题请把脚本输出发回来。

### 选什么卡

| 用途 | 显存 | 说明 |
|---|---|---|
| 只生图（Z-Image、FLUX.2、LongCat） | 24GB（如 RTX 4090）就够 | |
| 生成视频（LTX-2，19B） | **建议 48GB 以上**（L40S、A6000、A100、H100） | transformer 光权重就要约 38GB；24GB 卡只能逐层搬运，会非常慢 |

内存建议 64GB 以上；磁盘按模型大小准备：Z-Image 约 33GB、FLUX.2 klein 约 16GB、LongCat 约 30GB、LTX-2 约 95GB。

### 一键部署

```bash
git clone https://github.com/lcnvals/longcat-image-test.git && cd longcat-image-test
bash deploy/setup_cuda.sh                    # 默认装 Z-Image-Turbo + LTX-2，从魔搭下载
bash deploy/setup_cuda.sh --pip-mirror       # 国内服务器：pip 用清华镜像
bash deploy/setup_cuda.sh --source hf        # 海外服务器：从 Hugging Face 下载
bash deploy/setup_cuda.sh --models z-image-turbo,flux2-klein-4b,longcat,ltx-2   # 自选模型
bash deploy/setup_cuda.sh --check            # 只做环境检查和自检
```

脚本分四步：环境检查（GPU、显存、内存、磁盘、Python）→ 安装依赖到 `.venv-cuda` → 下载模型 → 自检（CUDA、diffusers 管线、模型、ffmpeg）。可以重复运行，已完成的部分会跳过，下载支持断点续传。

### 使用

命令和 Mac 上一样，只是换成 `.venv-cuda` 里的 Python，模型名不变（在 Linux 上自动选用 CUDA 实现）：

```bash
.venv-cuda/bin/python studio.py --model z-image-turbo
.venv-cuda/bin/python video.py gen "一只橘猫坐在窗台上，慢慢转头看向窗外" --ratio 小横屏
.venv-cuda/bin/python video.py board projects/<片名>/storyboard.json --concat
```

推荐流程：在 Mac 上用 studio 和 `video.py keyframes` 准备好 `projects/<片名>/`（分镜 + 关键帧），整个目录上传到服务器，跑 `video.py board ... --concat`，再把成片下载回来。

### 和 Mac 版的区别

| | Mac（LTX-2.3 蒸馏版，MLX） | NVIDIA（LTX-2，diffusers） |
|---|---|---|
| 模型 | LTX-2.3 22B 蒸馏，4-bit | LTX-2 19B，bf16，30 步，带 CFG |
| 负面提示词 | 通过 NAG 生效 | 原生支持 |
| 尾帧锁定 `--end-image` | 支持 | 暂不支持（会忽略） |
| `retake` / `extend` | 支持 | 暂不支持，可以用 `--only` 重做整个镜头 |
| 显存策略 | 低内存模式 | 自动：≥80GB 全部放显存，≥40GB 按组件搬运，更小逐层搬运 |

后端在 `video.json` 的 `cuda` 段配置（模型路径、步数、guidance、显存策略），也可以设置 `"backend": "mlx"` 或 `"cuda"` 强制指定。

## 实测速度（M1 Pro 32GB）

| 场景 | 耗时 |
|---|---|
| `studio.py` 启动加载 | 约 56 秒（只需一次） |
| 512×512，8 步预览 | 约 1.5–2.5 分钟 |
| 512×512，20 步 | 约 3.5 分钟 |

去噪每步约 10 秒（512×512，开着 CFG），分辨率和步数越大越慢。768×768 的计算量约是 512 的 2.3 倍。

注意：MPS 是异步执行的，diffusers 的进度条只反映任务提交速度，进度条跑满不代表算完了，要以脚本打印的耗时为准。

## 使用技巧

- **`/refine` 和 `/final` 怎么选**：改了步数后，调度器的去噪路线会变，`/final` 即使用同一个种子，手指数量、姿势这类细碎结构也可能和预览图不一样（实测种子 45 的 8 步预览手指正确，25 步重新生成后拇指错位）。`/refine` 把预览图编码回潜空间，按强度加噪后只跑后半程去噪，所以结构由预览图决定，只补细节。
  - 强度 0.4：最忠于预览图；0.5（默认）：结构不变、细节明显提升；0.6–0.7：细节更多，结构可能有小改动。
  - 实际只跑约“强度 × 步数”步，比 `/final` 更快。
  - 重启后仍能精修：启动时会从 `outputs/log.txt` 恢复历史；也可以直接指定文件，比如 `/refine 0.5 011_s45_8st_512x512.png`（种子和尺寸从文件名读取，提示词要先输入）。

- **先预览，再出正式图**：用 8 步调提示词，满意后提示词、种子、尺寸都不变，只加步数。预览能确认场景、人物、风格；动作和表情在正式图里可能会变。
- **预览和正式图的尺寸必须一致**：尺寸不同，初始噪声不同，就是两张完全不同的图。
- **关键信息写具体**：比如写“一位中国年轻女性”，而不是“一个女人”，否则人物特征不稳定。
- **图中文字用引号括起来**：比如 `招牌上写着"深夜食堂"`，模型会逐字渲染。
- **预览和正式出图都不要开 `--rewrite`**：改写结果每次可能不同，两次生成的提示词就对不上了。
- **出图前关掉占内存的程序**：内存紧张时系统会大量换页，速度会成倍下降。用久了交换空间涨得很高，重启一次电脑就好。

## 已知问题

- diffusers 的 `LongCatImagePipeline.encode_prompt` 在只传 `prompt_embeds`、不传 `prompt` 时会执行 `len(None)` 报错。脚本里用缓存的 embeds 替换了这个方法。
- `LongCatImagePipeline.__call__` 不支持 `callback_on_step_end`。

## 许可

模型和官方代码为 Apache-2.0，见 [meituan-longcat/LongCat-Image](https://github.com/meituan-longcat/LongCat-Image)。本地运行没有内容过滤，生成和传播内容的法律责任由使用者自己承担。
