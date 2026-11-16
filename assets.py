"""角色 / 怪物 / 场景“定义”（资产）：文字描述 + 设定图，定一次，所有镜头和项目都引用。

目录：assets/<类别>/<名字>/definition.json + 设定图
  类别：characters（角色）、monsters（怪物）、scenes（场景）、props（道具）

  .venv-mlx/bin/python assets.py new 红带武士 --kind character --desc "一名年轻武士，……" --style 游戏CG
  .venv-mlx/bin/python assets.py sheet 红带武士             # 按定义生成设定图（三视图等）
  .venv-mlx/bin/python assets.py sheet 红带武士 --only front --seed 43   # 只重出某一张
  .venv-mlx/bin/python assets.py import 八岐大蛇 ~/Downloads/大蛇.png --as full   # 导入网页版等外部生成的图
  .venv-mlx/bin/python assets.py list
  .venv-mlx/bin/python assets.py show 红带武士

分镜里用 "cast": {"hero": "红带武士", "monster": "八岐大蛇"} 引用，镜头描述里写 {hero}、{monster}。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "assets")
KINDS = {"character": "characters", "monster": "monsters", "scene": "scenes", "prop": "props"}
KIND_LABEL = {"character": "角色", "monster": "怪物", "scene": "场景", "prop": "道具"}

# 各类别默认生成的设定图：文件名 -> (画面要求, 尺寸)
DEFAULT_SHEETS = {
    "character": {
        "front": ("角色设定图，正面全身站姿，面向镜头，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "side": ("角色设定图，侧面全身站姿，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "back": ("角色设定图，背面全身站姿，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "action": ("手持武器摆出战斗姿势，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
    },
    "monster": {
        "full": ("怪物设定图，全身完整入镜，展示整体体型和全部特征，纯深灰色背景，没有其他角色", "1152x768"),
        "head": ("怪物设定图，头部特写，展示五官和纹理细节，纯深灰色背景", "1152x768"),
        "detail": ("怪物设定图，身体表面材质细节特写，纯深灰色背景", "1152x768"),
    },
    "scene": {
        "wide": ("场景设定图，空镜远景，画面中没有任何人物和怪物", "1152x768"),
        "mid": ("场景设定图，中景，画面中没有任何人物和怪物", "1152x768"),
    },
    "prop": {
        "front": ("道具设定图，正面完整展示，纯浅灰色背景", "1024x1024"),
    },
}


# ---------------- 读取 ----------------

def asset_dir(kind, name):
    return os.path.join(ROOT, KINDS[kind], name)


def find(name):
    """按名字在所有类别里查找定义，返回 (定义 dict, 目录)；找不到返回 (None, None)。"""
    for kind, sub in KINDS.items():
        d = os.path.join(ROOT, sub, name)
        path = os.path.join(d, "definition.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("kind", kind)
            data.setdefault("name", name)
            return data, d
    return None, None


def load(name):
    data, d = find(name)
    if data is None:
        known = "、".join(n for _, n, _ in all_assets()) or "（还没有任何定义）"
        raise SystemExit(f"找不到定义“{name}”。已有：{known}。用 assets.py new 创建")
    return data, d


def all_assets():
    out = []
    for kind, sub in KINDS.items():
        base = os.path.join(ROOT, sub)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if os.path.exists(os.path.join(base, name, "definition.json")):
                    out.append((kind, name, os.path.join(base, name)))
    return out


def describe(data):
    """用于提示词的文字：描述 + 必须保持的特征。"""
    text = data.get("desc", "")
    must = [m for m in data.get("must", []) if m not in text]
    if must:
        text += "，" + "，".join(must)
    return text


def images(data, d):
    """已经生成的设定图（绝对路径），按定义里的顺序。"""
    return [os.path.join(d, f) for f in data.get("images", []) if os.path.exists(os.path.join(d, f))]


def resolve_cast(cast, with_voice=False):
    """分镜的 cast：{"hero": "红带武士", ...} -> ({"hero": 描述文字}, {"hero": [设定图路径]})
    with_voice=True 时再返回 {"hero": {"name": 显示名, "voice": 音色配置}}，配音跟随角色定义。"""
    texts, refs, voices = {}, {}, {}
    for key, name in (cast or {}).items():
        data, d = load(name)
        texts[key] = describe(data)
        refs[key] = images(data, d)
        voices[key] = {"name": data.get("display_name") or data["name"], "voice": data.get("voice")}
    return (texts, refs, voices) if with_voice else (texts, refs)


# ---------------- 创建与生成设定图 ----------------

def cmd_new(a):
    d = asset_dir(a.kind, a.name)
    path = os.path.join(d, "definition.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} 已存在。直接编辑这个文件即可")
    os.makedirs(d, exist_ok=True)
    voice = {"engine": "say", "voice": a.voice, "rate": a.rate} if a.voice else None
    data = {"name": a.name, "kind": a.kind, "desc": a.desc, "must": a.must or [], "avoid": a.avoid or [],
            "voice": voice,
            "style": a.style, "seed": a.seed, "images": [],
            "sheets": {k: {"prompt": p, "size": s} for k, (p, s) in DEFAULT_SHEETS[a.kind].items()},
            "_说明": "desc 是每个镜头都会原样引用的外观描述；must 是必须保持的特征，avoid 是不该出现的；"
                     "voice 是这个角色说台词时用的音色（镜头里写 speaker 引用）；"
                     "sheets 是要生成的设定图，可以增删改；images 由 assets.py sheet 自动填写"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已创建{KIND_LABEL[a.kind]}定义 -> {os.path.relpath(path, HERE)}")
    print(f"生成设定图：.venv-mlx/bin/python assets.py sheet {a.name}")


def cmd_sheet(a):
    data, d = load(a.name)
    path = os.path.join(d, "definition.json")
    sheets = data.get("sheets") or {k: {"prompt": p, "size": s} for k, (p, s) in DEFAULT_SHEETS[data["kind"]].items()}
    todo = {k: v for k, v in sheets.items() if not a.only or k in a.only}
    if not todo:
        raise SystemExit(f"没有要生成的设定图。可选：{'、'.join(sheets)}")
    sys.path.insert(0, HERE)
    from backends import BACKENDS
    from presets import STYLES
    backend = BACKENDS[a.model](quantize=a.quantize)
    style = STYLES.get(data.get("style") or "", "")
    # avoid 不能写进正面提示词：“不要出现四肢”反而会让模型画出四肢。
    # 支持负面提示词的模型放进 neg；不支持的只提醒，请在 desc 里用正面描述（如“蛇形身躯盘绕在地面上”）
    avoid = "，".join(data.get("avoid") or [])
    neg = avoid if backend.supports_neg else ""
    if avoid and not backend.supports_neg:
        print(f"  提醒：{backend.title} 不支持负面提示词，avoid（{avoid}）不会生效；"
              f"请在 desc 里用正面描述代替否定")
    seed = a.seed if a.seed is not None else data.get("seed", 42)
    made = []
    for key, sheet in todo.items():
        w, h = map(int, sheet["size"].split("x"))
        prompt = "，".join(x for x in (describe(data), sheet["prompt"], style) if x)
        print(f"\n== {key}  {w}x{h}  种子 {seed}\n  {prompt}", flush=True)
        t0 = time.time()
        img, _ = backend.generate(prompt, neg, w, h, backend.final_steps, seed, backend.cfg)
        fname = f"{key}.png"
        img.save(os.path.join(d, fname))
        img.save(os.path.join(d, f"{key}_s{seed}.png"))  # 保留每个种子的版本，方便回退
        backend.cleanup()
        made.append(fname)
        print(f"  -> {os.path.relpath(os.path.join(d, fname), HERE)}  ({time.time() - t0:.0f}s)", flush=True)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["images"] = list(dict.fromkeys((raw.get("images") or []) + made))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def cmd_import(a):
    """把外部生成的图片（网页版、其他工具、手绘）导入为设定图，并登记到定义里。"""
    import shutil
    data, d = load(a.name)
    if not os.path.exists(a.file):
        raise SystemExit(f"找不到图片：{a.file}")
    key = a.as_name or os.path.splitext(os.path.basename(a.file))[0]
    dst = os.path.join(d, f"{key}.png")
    if os.path.exists(dst):  # 旧图留一个备份，方便回退
        shutil.copyfile(dst, os.path.join(d, f"{key}_before_import_{time.strftime('%m%d%H%M%S')}.png"))
    from PIL import Image
    Image.open(a.file).convert("RGB").save(dst)
    path = os.path.join(d, "definition.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    images = [x for x in (raw.get("images") or []) if x != f"{key}.png"]
    raw["images"] = [f"{key}.png"] + images if a.first else images + [f"{key}.png"]
    raw.setdefault("_来源", {})[f"{key}.png"] = f"外部导入：{os.path.abspath(a.file)}（{time.strftime('%Y-%m-%d %H:%M')}）"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    order = "、".join(raw["images"])
    print(f"已导入 -> {os.path.relpath(dst, HERE)}；当前设定图顺序：{order}")
    print("出关键帧时按这个顺序取参考图（每个定义取 ref_images 张）；--first 可以把导入的图排到最前")


def cmd_list(_):
    items = all_assets()
    if not items:
        print("还没有任何定义。用 assets.py new 名字 --kind character --desc \"……\" 创建")
    for kind, name, d in items:
        data, _ = find(name)
        print(f"  [{KIND_LABEL[kind]}] {name:<10} 设定图 {len(images(data, d))} 张  {data.get('desc', '')[:40]}")


def cmd_show(a):
    data, d = load(a.name)
    print(f"[{KIND_LABEL[data['kind']]}] {data['name']}  （{os.path.relpath(d, HERE)}）")
    print(f"  描述：{data.get('desc')}")
    print(f"  必须：{'、'.join(data.get('must', [])) or '-'}   避免：{'、'.join(data.get('avoid', [])) or '-'}")
    print(f"  风格：{data.get('style') or '-'}   种子：{data.get('seed', 42)}")
    v = data.get("voice") or {}
    print(f"  音色：{v.get('voice', '-')}（{v.get('engine', '-')}，语速 {v.get('rate', '-')}）" if v else "  音色：-（用分镜的旁白音色）")
    print(f"  设定图：{'、'.join(os.path.basename(x) for x in images(data, d)) or '（还没生成）'}")
    print(f"  可生成：{'、'.join((data.get('sheets') or DEFAULT_SHEETS[data['kind']]).keys())}")


def main():
    ap = argparse.ArgumentParser(description="角色 / 怪物 / 场景定义", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="创建定义")
    n.add_argument("name")
    n.add_argument("--kind", choices=list(KINDS), required=True)
    n.add_argument("--desc", required=True, help="外观描述，每个镜头都会原样引用")
    n.add_argument("--must", nargs="+", help="必须保持的特征")
    n.add_argument("--avoid", nargs="+", help="不该出现的特征")
    n.add_argument("--style", help="生成设定图用的风格预设，如 游戏CG")
    n.add_argument("--seed", type=int, default=42)
    n.add_argument("--voice", help="台词音色（macOS say 的音色名，如 Tingting）")
    n.add_argument("--rate", type=int, default=185, help="语速")
    s = sub.add_parser("sheet", help="按定义生成设定图")
    s.add_argument("name")
    s.add_argument("--only", nargs="+", help="只生成这几张，如 front side")
    s.add_argument("--seed", type=int, help="覆盖定义里的种子")
    s.add_argument("--model", default="z-image-turbo")
    s.add_argument("--quantize", type=int)
    im = sub.add_parser("import", help="导入外部生成的图片作为设定图")
    im.add_argument("name")
    im.add_argument("file")
    im.add_argument("--as", dest="as_name", help="保存成哪个名字（如 full、front），默认用原文件名")
    im.add_argument("--first", action="store_true", help="排到设定图最前面，出关键帧时优先作为参考")
    sub.add_parser("list", help="列出所有定义")
    sh = sub.add_parser("show", help="查看一个定义")
    sh.add_argument("name")
    a = ap.parse_args()
    {"new": cmd_new, "sheet": cmd_sheet, "import": cmd_import, "list": cmd_list, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
