# LongCat-Image on Apple Silicon

在 Apple Silicon Mac 上本地运行美团 [LongCat-Image](https://github.com/meituan-longcat/LongCat-Image)（6B 文生图模型）的推理脚本。

官方示例针对 CUDA，并用 `enable_model_cpu_offload()` 节省显存；Mac 是统一内存，这个办法省不了内存。本仓库改用 MPS，并通过分段加载 / int8 量化让模型在 32GB 内存的 Mac 上跑起来。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `studio.py` | 常驻交互模式：模型只加载一次，先低步数预览、再出正式图；带画幅、镜头、光线、风格等预设 | 已在 M1 Pro 32GB 上测试 |
| `t2i.py` | 单次文生图，文本编码器和 transformer 分段加载 | 已在 M1 Pro 32GB 上测试 |
| `edit.py` | LongCat-Image-Edit-Turbo 改图（照片 + 指令） | 未实测 |
| `presets.py` | `studio.py` 用的提示词预设，可以自己加 | — |

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

## 使用

### studio.py：常驻交互模式（推荐）

```bash
.venv/bin/python studio.py --size 768x768
```

启动时加载一次模型（约 1 分钟），出现 `longcat>` 提示符后。

输入时有**命令菜单**：

- 输入 `/` 会弹出全部命令和说明；继续输入可以筛选，比如 `/st` 只剩 `/style`、`/steps`。
- 选中命令后会接着弹出参数菜单，比如 `/style ` 后面列出全部风格和对应的描述。
- 用 ↑↓ 选择，按回车或 Tab 确认。选中命令时回车只填入命令；选中参数时回车直接执行。
- ↑↓ 在没有菜单时用来翻历史输入，历史保存在 `outputs/.history`。

基本流程：


```
longcat> 一位中国年轻女性在厨房做饭，写实摄影     ← 直接输入提示词：8 步预览，自动打开
longcat> /batch 4                              ← 同一提示词换 4 个种子（43–46）各出一张预览
longcat> /seed 45                              ← 选中 45 号
longcat> /final                                ← 同提示词、同种子、同尺寸出 25 步正式图
longcat> /quit
```

| 命令 | 作用 |
|---|---|
| 直接输入文字 | 用预览步数出图 |
| `/final [步数]` | 用上一条提示词出正式图（默认 25 步） |
| `/batch N` | 上一条提示词连换 N 个种子出预览 |
| `/seed N`、`/seed +` | 设种子 / 种子加 1 |
| `/size WxH` | 设尺寸 |
| `/steps N` | 设预览步数 |
| `/cfg X` | 设 guidance_scale（默认 4.0） |
| `/ratio [名称]` | 画幅预设，会改尺寸 |
| `/shot` `/light` `/tone` `/time` `/detail` `/style` | 镜头、光线、色调、时间天气、细节质感、风格预设（见下文） |
| `/neg [预设名\|文本\|off]` | 负面提示词：用预设或自定义文本；不带参数列出预设；`off` 清除 |
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
| `/neg` | 负面提示词 | 通用、人像、风景、插画 |

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

启动参数：`--size`（默认 768x768）、`--ratio`（会覆盖 `--size`）、`--preview`（默认 8）、`--final`（默认 25）、`--seed`（默认 43）、`--shot`、`--light`、`--tone`、`--time`、`--detail`（可多个，如 `--detail 毛发 锐利`）、`--style`、`--neg`、`--quant-transformer`（transformer 也量化为 int8，24GB 机器用，未实测）。

图片保存在 `outputs/`，文件名带编号、种子、步数、尺寸，比如 `003_s45_25st_768x768.png`。每张图的提示词记录在 `outputs/log.txt`。

**内存方案**：文本编码器用 optimum-quanto 量化为 int8（16.6GB 降到约 8.5GB），和 bf16 的 transformer 一起常驻 MPS，合计约 21GB。实测量化后的出图和 bf16 几乎一致。

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

## 实测速度（M1 Pro 32GB）

| 场景 | 耗时 |
|---|---|
| `studio.py` 启动加载 | 约 56 秒（只需一次） |
| 512×512，8 步预览 | 约 1.5–2.5 分钟 |
| 512×512，20 步 | 约 3.5 分钟 |

去噪每步约 10 秒（512×512，开着 CFG），分辨率和步数越大越慢。768×768 的计算量约是 512 的 2.3 倍。

注意：MPS 是异步执行的，diffusers 的进度条只反映任务提交速度，进度条跑满不代表算完了，要以脚本打印的耗时为准。

## 使用技巧

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
