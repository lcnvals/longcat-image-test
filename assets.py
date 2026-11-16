"""定义（资产）：人物、怪物、装备服饰、技能、场景。定一次，所有镜头和项目都引用。

参考换装类游戏和网游的设计：
  人物 = 本体（脸、体型、发色）+ 栏位（发型、上衣、外套、鞋子、饰品……主手武器、护甲）
       + 套装（便装、战斗装……覆盖部分栏位）+ 技能 + 音色
  单品（武器、护甲、服饰、饰品）各自是定义，有自己的设定图，可以给不同人物穿戴
  技能（人物和怪物都有）= 前摇 / 释放 / 命中三段 + 固定色调 + 音效 + 剪辑方式
  场景 = 世界观 → 区域 → 地点，地点有时间、天气变体

目录：assets/<类别>/<名字>/definition.json + 设定图；栏位可在 assets/slots.json 里增改

常用命令：
  assets.py new 红带武士 --kind character --base "年轻男性，黑色短发，剑眉，身形精瘦"
  assets.py new 断罪太刀 --kind weapon --desc "一把长太刀，刀身有暗纹，黑色刀柄缠红绳"
  assets.py wear 红带武士 主手 断罪太刀              # 穿戴：栏位填单品名字，或直接写一段描述
  assets.py loadout 红带武士 便装 护甲=- 上衣=素色和服 主手=木刀   # 套装：覆盖部分栏位，- 表示脱下
  assets.py describe 红带武士 --loadout 便装          # 预览拼出来的完整描述
  assets.py sheet 红带武士                            # 生成设定图
  assets.py import 八岐大蛇 ~/Downloads/大蛇.png --as full --first   # 导入外部生成的图
  assets.py list / show 名字 / slots                  # 列出、查看、查看全部栏位

分镜里："cast": {"hero": "红带武士", "monster": "八岐大蛇", "scene": "火山峡谷"}，描述里写 {hero}；
镜头里可写 "loadout": {"hero": "便装"}、"variant": {"scene": "夜·暴雨"}、"skill": "八岐大蛇.烈焰吐息", "stage": "释放"。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "assets")

KINDS = {  # 类别 -> 目录
    "character": "characters", "monster": "monsters",
    "weapon": "weapons", "armor": "armors", "outfit": "outfits", "accessory": "accessories",
    "mount": "mounts", "skill": "skills", "prop": "props",
    "world": "worlds", "region": "regions", "scene": "scenes",
}
KIND_LABEL = {"character": "人物", "monster": "怪物", "weapon": "武器", "armor": "护甲", "outfit": "服饰",
              "accessory": "饰品", "mount": "坐骑", "skill": "技能", "prop": "道具",
              "world": "世界观", "region": "区域", "scene": "地点"}
ITEM_KINDS = ("weapon", "armor", "outfit", "accessory", "prop")
DEFAULT_ITEM_SLOT = {"weapon": "主手", "armor": "护甲", "outfit": "上衣", "accessory": "头饰", "prop": "副手"}

# 默认栏位（按描述拼接的顺序）。template 里的 {} 换成单品描述；excludes 是互斥的栏位。
DEFAULT_SLOTS = [
    {"slot": "发型", "group": "发型妆容", "template": "{}"},
    {"slot": "妆容", "group": "发型妆容", "template": "{}"},
    {"slot": "头饰", "group": "饰品", "template": "头戴{}"},
    {"slot": "耳饰", "group": "饰品", "template": "{}"},
    {"slot": "面饰", "group": "饰品", "template": "脸上{}"},
    {"slot": "颈饰", "group": "饰品", "template": "颈间{}"},
    {"slot": "连衣裙", "group": "服装", "template": "穿着{}", "excludes": ["上衣", "下装"]},
    {"slot": "上衣", "group": "服装", "template": "上身穿{}"},
    {"slot": "下装", "group": "服装", "template": "下身穿{}"},
    {"slot": "护甲", "group": "战斗", "template": "身披{}"},
    {"slot": "外套", "group": "服装", "template": "外披{}"},
    {"slot": "手饰", "group": "饰品", "template": "{}"},
    {"slot": "腰饰", "group": "饰品", "template": "腰间{}"},
    {"slot": "袜子", "group": "服装", "template": "{}"},
    {"slot": "鞋子", "group": "服装", "template": "脚穿{}"},
    {"slot": "背饰", "group": "饰品", "template": "背后{}"},
    {"slot": "主手", "group": "战斗", "template": "手持{}"},
    {"slot": "副手", "group": "战斗", "template": "另一只手拿着{}"},
    {"slot": "坐骑", "group": "战斗", "template": "骑着{}"},
]

# 各类别默认生成的设定图：文件名 -> (画面要求, 尺寸)
DEFAULT_SHEETS = {
    "character": {
        "front": ("角色设定图，正面全身站姿，面向镜头，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "side": ("角色设定图，侧面全身站姿，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "back": ("角色设定图，背面全身站姿，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
        "action": ("摆出战斗姿势，全身完整入镜，纯浅灰色背景，没有其他人物", "768x1152"),
    },
    "monster": {
        "full": ("怪物设定图，全身完整入镜，展示整体体型和全部特征，纯深灰色背景，没有其他角色", "1152x768"),
        "head": ("怪物设定图，头部特写，展示五官和纹理细节，纯深灰色背景", "1152x768"),
        "detail": ("怪物设定图，身体表面材质细节特写，纯深灰色背景", "1152x768"),
    },
    "weapon": {
        "full": ("武器设定图，横向完整展示整把武器，纯浅灰色背景，没有人物", "1152x768"),
        "detail": ("武器设定图，刃部和握柄的细节特写，纯浅灰色背景", "1152x768"),
    },
    "armor": {"front": ("装备设定图，穿在无脸人台上的正面完整展示，纯浅灰色背景", "768x1152")},
    "outfit": {"front": ("服装设定图，穿在无脸人台上的正面完整展示，纯浅灰色背景", "768x1152")},
    "accessory": {"front": ("饰品设定图，正面特写完整展示，纯浅灰色背景", "1024x1024")},
    "mount": {"side": ("坐骑设定图，侧面全身完整入镜，纯浅灰色背景", "1152x768")},
    "skill": {"effect": ("技能特效概念图，深色背景上展示特效的形态和颜色，没有人物", "1152x768")},
    "prop": {"front": ("道具设定图，正面完整展示，纯浅灰色背景", "1024x1024")},
    "world": {"key": ("世界观概念图，宏大的远景，展示整体风格与氛围，没有人物", "1152x768")},
    "region": {"map": ("区域俯视地图，古卷轴地图风格，标出山脉、河流、城镇和地标", "1152x768"),
               "wide": ("区域远景概念图，展示整体地貌与建筑风格，没有人物", "1152x768")},
    "scene": {
        "wide": ("场景设定图，空镜远景，画面中没有任何人物和怪物", "1152x768"),
        "mid": ("场景设定图，中景，画面中没有任何人物和怪物", "1152x768"),
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


def save(data, d):
    with open(os.path.join(d, "definition.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def all_assets():
    out = []
    for kind, sub in KINDS.items():
        base = os.path.join(ROOT, sub)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if os.path.exists(os.path.join(base, name, "definition.json")):
                    out.append((kind, name, os.path.join(base, name)))
    return out


def slot_table():
    """栏位表：默认栏位 + assets/slots.json 里的自定义（同名覆盖，新增的按文件里的位置插入末尾）。"""
    slots = [dict(s) for s in DEFAULT_SLOTS]
    path = os.path.join(ROOT, "slots.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            custom = json.load(f)
        index = {s["slot"]: i for i, s in enumerate(slots)}
        for s in custom:
            if s["slot"] in index:
                slots[index[s["slot"]]].update(s)
            else:
                slots.append(s)
    return slots


def images(data, d, variant=None):
    """已经生成的设定图（绝对路径），按定义里的顺序；地点有变体图时优先用变体图。"""
    if variant:
        vimgs = [os.path.join(d, f) for f in (data.get("variant_images") or {}).get(variant, [])
                 if os.path.exists(os.path.join(d, f))]
        if vimgs:
            return vimgs
    return [os.path.join(d, f) for f in data.get("images", []) if os.path.exists(os.path.join(d, f))]


# ---------------- 拼描述 ----------------

def _plain(data):
    text = data.get("base") or data.get("desc", "")
    must = [m for m in data.get("must", []) if m not in text]
    return "，".join([text] + must) if must else text


def equipped(data, loadout=None):
    """人物当前穿戴：{栏位: 单品名或描述}。套装覆盖默认栏位，值为 "-" 或空表示脱下；处理互斥栏位。"""
    slots = dict(data.get("slots") or {})
    if loadout:
        sets = data.get("loadouts") or {}
        if loadout not in sets:
            raise SystemExit(f"{data['name']} 没有套装“{loadout}”，已有：{'、'.join(sets) or '无'}")
        slots.update(sets[loadout])
    slots = {k: v for k, v in slots.items() if v and v != "-"}
    for s in slot_table():
        if s["slot"] in slots:
            for ex in s.get("excludes", []):
                slots.pop(ex, None)  # 例：穿了连衣裙就不再拼上衣、下装
    return slots


def item_text(value):
    """栏位的值：已有定义就用定义的描述，否则当作直接写的描述。"""
    data, _ = find(value)
    return _plain(data) if data else value


def describe(data, loadout=None, variant=None):
    """用于提示词的完整文字。人物：本体 + 各栏位；地点：世界观 + 区域 + 地点 + 变体；其他：描述 + 必须特征。"""
    kind = data.get("kind")
    if kind in ("character", "monster") and data.get("slots"):
        parts = [_plain(data)]
        worn = equipped(data, loadout)
        for s in slot_table():
            if s["slot"] in worn:
                parts.append(s.get("template", "{}").format(item_text(worn[s["slot"]])))
        return "，".join(p for p in parts if p)
    if kind == "scene":
        chain, cur = [], data
        while cur and cur.get("parent"):  # 向上找区域、世界观
            cur, _ = find(cur["parent"])
            if cur:
                chain.insert(0, _plain(cur))
        parts = chain + [_plain(data)]
        if variant:
            vs = data.get("variants") or {}
            parts.append(vs.get(variant, variant))
        return "，".join(p for p in parts if p)
    return _plain(data)


def gear_images(data, loadout=None):
    """人物穿戴的单品里，已有设定图的（每件取第一张），按栏位顺序。"""
    out = []
    worn = equipped(data, loadout)
    for s in slot_table():
        name = worn.get(s["slot"])
        if name:
            item, d = find(name)
            if item:
                imgs = images(item, d)
                if imgs:
                    out.append(imgs[0])
    return out


def resolve_entry(name, loadout=None, variant=None):
    """单个 cast 条目 -> (描述文字, 参考图[本体在前、装备在后], 音色信息)"""
    data, d = load(name)
    text = describe(data, loadout, variant)
    own = images(data, d, variant)
    if loadout:  # 换装时优先用这套装的设定图（assets.py sheet 名字 --loadout 套装 生成），避免被默认造型带偏
        lo_imgs = [os.path.join(d, f) for f in (data.get("loadout_images") or {}).get(loadout, [])
                   if os.path.exists(os.path.join(d, f))]
        if lo_imgs:
            own = lo_imgs
        else:
            own = []  # 没有套装设定图时不用默认造型的图做参考，只靠文字描述 + 单品图
    refs = own + (gear_images(data, loadout) if data.get("slots") else [])
    voice = {"name": data.get("display_name") or data["name"], "voice": data.get("voice")}
    return text, refs, voice


def resolve_cast(cast, with_voice=False, loadouts=None, variants=None):
    """分镜的 cast -> ({键: 描述}, {键: 参考图}[, {键: 音色}])。

    cast 的值可以是名字，或 {"name": 名字, "loadout": 套装, "variant": 变体}；
    loadouts / variants 是镜头级覆盖：{"hero": "便装"}、{"scene": "夜·暴雨"}。
    """
    texts, refs, voices = {}, {}, {}
    for key, entry in (cast or {}).items():
        if isinstance(entry, dict):
            name, lo, va = entry["name"], entry.get("loadout"), entry.get("variant")
        else:
            name, lo, va = entry, None, None
        lo = (loadouts or {}).get(key, lo)
        va = (variants or {}).get(key, va)
        texts[key], refs[key], voices[key] = resolve_entry(name, lo, va)
    return (texts, refs, voices) if with_voice else (texts, refs)


def find_skill(ref):
    """技能引用：'八岐大蛇.烈焰吐息' 或 '烈焰吐息' -> 技能定义"""
    owner, _, skill = ref.rpartition(".")
    data, _ = find(skill)
    if data is None or data.get("kind") != "skill":
        raise SystemExit(f"找不到技能“{ref}”。用 assets.py new {skill} --kind skill --owner {owner or '所属'} 创建")
    if owner and data.get("owner") and data["owner"] != owner:
        raise SystemExit(f"技能“{skill}”属于 {data['owner']}，不是 {owner}")
    return data


def skill_text(data, stage=None):
    """技能某一阶段的描述 + 固定色调。stage：前摇 / 释放 / 命中；不写就是释放。"""
    stages = data.get("stages") or {}
    stage = stage or "释放"
    if stage not in stages:
        raise SystemExit(f"技能“{data['name']}”没有阶段“{stage}”，已有：{'、'.join(stages)}")
    color = data.get("color")
    return "，".join(x for x in (stages[stage], f"{color}色调的技能特效" if color else "") if x)


# ---------------- 命令 ----------------

def cmd_new(a):
    d = asset_dir(a.kind, a.name)
    path = os.path.join(d, "definition.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} 已存在。直接编辑这个文件，或用 wear / loadout 修改")
    if a.kind in ("character",) and not (a.base or a.desc):
        raise SystemExit("人物需要 --base（本体：脸、体型、发色等，不含服装）")
    os.makedirs(d, exist_ok=True)
    data = {"name": a.name, "kind": a.kind}
    if a.kind in ("character", "monster"):
        data["base" if a.base else "desc"] = a.base or a.desc
        data.update({"slots": {}, "loadouts": {}, "skills": [], "display_name": a.name,
                     "voice": {"engine": "say", "voice": a.voice, "rate": a.rate} if a.voice else None})
    else:
        data["desc"] = a.desc or ""
    if a.kind in ITEM_KINDS:
        data["slot"] = a.slot or DEFAULT_ITEM_SLOT[a.kind]
    if a.kind == "skill":
        data.update({"owner": a.owner, "type": "近战", "color": "",
                     "stages": {"前摇": "", "释放": a.desc or "", "命中": ""},
                     "sfx": "impact", "edit": {"in": "flash", "shake": True}})
    if a.kind in ("region", "scene"):
        data["parent"] = a.parent
    if a.kind == "scene":
        data["variants"] = {}
    data.update({"must": a.must or [], "avoid": a.avoid or [], "style": a.style, "seed": a.seed, "images": [],
                 "sheets": {k: {"prompt": p, "size": s} for k, (p, s) in DEFAULT_SHEETS[a.kind].items()}})
    save(data, d)
    print(f"已创建{KIND_LABEL[a.kind]}定义 -> {os.path.relpath(path, HERE)}")
    if a.kind == "character":
        print(f"穿戴：assets.py wear {a.name} 主手 单品名或描述    预览：assets.py describe {a.name}")
    elif a.kind == "skill":
        print(f"补全技能的前摇/释放/命中描述和 color：编辑 {os.path.relpath(path, HERE)}")
    else:
        print(f"生成设定图：assets.py sheet {a.name}")


def cmd_wear(a):
    data, d = load(a.name)
    names = [s["slot"] for s in slot_table()]
    if a.slot not in names:
        raise SystemExit(f"没有栏位“{a.slot}”。可用：{'、'.join(names)}（可在 assets/slots.json 里增加）")
    value = " ".join(a.value)
    slots = data.setdefault("slots", {})
    if value in ("-", ""):
        slots.pop(a.slot, None)
        print(f"{a.name} 脱下了 {a.slot}")
    else:
        slots[a.slot] = value
        item, _ = find(value)
        hint = f"（单品定义：{KIND_LABEL[item['kind']]}）" if item else "（直接描述）"
        print(f"{a.name} 的 {a.slot}：{value} {hint}")
    save(data, d)
    print("当前完整描述：" + describe(data))


def cmd_loadout(a):
    data, d = load(a.name)
    names = [s["slot"] for s in slot_table()]
    changes = {}
    for kv in a.changes:
        slot, _, value = kv.partition("=")
        if slot not in names:
            raise SystemExit(f"没有栏位“{slot}”。可用：{'、'.join(names)}")
        changes[slot] = value or "-"
    data.setdefault("loadouts", {})[a.set_name] = changes
    save(data, d)
    print(f"{a.name} 的套装“{a.set_name}”：{changes}")
    print("套装完整描述：" + describe(data, a.set_name))


def cmd_describe(a):
    data, _ = load(a.name)
    print(describe(data, a.loadout, a.variant))


def cmd_slots(_):
    groups = {}
    for s in slot_table():
        groups.setdefault(s.get("group", "其他"), []).append(s["slot"] + ("（与" + "、".join(s["excludes"]) + "互斥）"
                                                                          if s.get("excludes") else ""))
    for g, items in groups.items():
        print(f"  {g}：{'、'.join(items)}")
    print("自定义：在 assets/slots.json 里写 [{\"slot\": \"发饰\", \"group\": \"饰品\", \"template\": \"头发上别着{}\"}]")


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
    avoid = "，".join(data.get("avoid") or [])
    neg = avoid if backend.supports_neg else ""
    if avoid and not backend.supports_neg:
        print(f"  提醒：{backend.title} 不支持负面提示词，avoid（{avoid}）不会生效；请在描述里用正面写法代替否定")
    seed = a.seed if a.seed is not None else data.get("seed", 42)
    desc = describe(data, a.loadout, a.variant)
    made = []
    for key, sheet in todo.items():
        w, h = map(int, sheet["size"].split("x"))
        prompt = "，".join(x for x in (desc, sheet["prompt"], style) if x)
        tag = f"{key}@{a.variant}" if a.variant else (f"{key}@{a.loadout}" if a.loadout else key)
        print(f"\n== {tag}  {w}x{h}  种子 {seed}\n  {prompt}", flush=True)
        t0 = time.time()
        img, _ = backend.generate(prompt, neg, w, h, backend.final_steps, seed, backend.cfg)
        fname = f"{tag}.png"
        img.save(os.path.join(d, fname))
        img.save(os.path.join(d, f"{tag}_{a.model}_s{seed}.png"))  # 按模型+种子保留每个版本
        backend.cleanup()
        made.append(fname)
        print(f"  -> {os.path.relpath(os.path.join(d, fname), HERE)}  ({time.time() - t0:.0f}s)", flush=True)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if a.variant:
        vi = raw.setdefault("variant_images", {})
        vi[a.variant] = list(dict.fromkeys((vi.get(a.variant) or []) + made))
    elif a.loadout:
        li = raw.setdefault("loadout_images", {})
        li[a.loadout] = list(dict.fromkeys((li.get(a.loadout) or []) + made))
    else:
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
    imgs = [x for x in (raw.get("images") or []) if x != f"{key}.png"]
    raw["images"] = [f"{key}.png"] + imgs if a.first else imgs + [f"{key}.png"]
    raw.setdefault("_来源", {})[f"{key}.png"] = f"外部导入：{os.path.abspath(a.file)}（{time.strftime('%Y-%m-%d %H:%M')}）"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"已导入 -> {os.path.relpath(dst, HERE)}；当前设定图顺序：{'、'.join(raw['images'])}")
    print("出关键帧时按这个顺序取参考图；--first 可以把导入的图排到最前")


def cmd_list(a):
    items = all_assets()
    if not items:
        print("还没有任何定义。用 assets.py new 名字 --kind character --base \"……\" 创建")
    for kind, name, d in items:
        if a.kind and kind != a.kind:
            continue
        data, _ = find(name)
        extra = f"栏位 {len(data.get('slots') or {})}，套装 {len(data.get('loadouts') or {})}" \
            if kind in ("character", "monster") else (f"栏位 {data.get('slot')}" if kind in ITEM_KINDS else "")
        print(f"  [{KIND_LABEL[kind]}] {name:<10} 设定图 {len(images(data, d))} 张  {extra}  "
              f"{(data.get('base') or data.get('desc') or '')[:30]}")


def cmd_show(a):
    data, d = load(a.name)
    kind = data["kind"]
    print(f"[{KIND_LABEL[kind]}] {data['name']}  （{os.path.relpath(d, HERE)}）")
    if data.get("base"):
        print(f"  本体：{data['base']}")
    elif data.get("desc"):
        print(f"  描述：{data['desc']}")
    if data.get("slots"):
        for s in slot_table():
            v = data["slots"].get(s["slot"])
            if v:
                item, _ = find(v)
                print(f"    {s['slot']}：{v}{'（' + KIND_LABEL[item['kind']] + '）' if item else ''}")
    for name, ch in (data.get("loadouts") or {}).items():
        print(f"  套装 {name}：{ch}")
    if data.get("skills"):
        print(f"  技能：{'、'.join(data['skills'])}")
    if kind in ITEM_KINDS:
        print(f"  栏位：{data.get('slot')}")
    if kind == "skill":
        print(f"  所属：{data.get('owner') or '-'}  类型：{data.get('type')}  色调：{data.get('color') or '-'}  "
              f"音效：{data.get('sfx') or '-'}  剪辑：{data.get('edit') or '-'}")
        for st, txt in (data.get("stages") or {}).items():
            print(f"    {st}：{txt or '（未填写）'}")
    if kind in ("region", "scene") and data.get("parent"):
        print(f"  上级：{data['parent']}")
    if data.get("variants"):
        print(f"  变体：{'、'.join(data['variants'])}")
    print(f"  必须：{'、'.join(data.get('must', [])) or '-'}   避免：{'、'.join(data.get('avoid', [])) or '-'}")
    v = data.get("voice") or {}
    if kind in ("character", "monster"):
        print(f"  音色：{v.get('voice', '-')}（语速 {v.get('rate', '-')}）" if v else "  音色：-（用分镜的旁白音色）")
    print(f"  设定图：{'、'.join(os.path.basename(x) for x in images(data, d)) or '（还没生成）'}")
    if kind != "skill":
        print(f"  完整描述：{describe(data)}")


def main():
    ap = argparse.ArgumentParser(description="定义：人物、怪物、装备服饰、技能、场景",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("常用命令：", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="创建定义")
    n.add_argument("name")
    n.add_argument("--kind", choices=list(KINDS), required=True,
                   help="、".join(f"{k}={v}" for k, v in KIND_LABEL.items()))
    n.add_argument("--base", help="人物/怪物本体：脸、体型、发色等（不含服装和武器）")
    n.add_argument("--desc", help="外观描述（单品、场景等）；技能则是“释放”阶段的描述")
    n.add_argument("--slot", help="单品默认穿戴在哪个栏位")
    n.add_argument("--owner", help="技能所属的人物或怪物")
    n.add_argument("--parent", help="区域所属的世界观 / 地点所属的区域")
    n.add_argument("--must", nargs="+", help="必须保持的特征")
    n.add_argument("--avoid", nargs="+", help="不该出现的特征（只在支持负面提示词的模型里生效）")
    n.add_argument("--style", help="生成设定图用的风格预设，如 游戏CG")
    n.add_argument("--seed", type=int, default=42)
    n.add_argument("--voice", help="台词音色（macOS say 的音色名，如 Tingting）")
    n.add_argument("--rate", type=int, default=185, help="语速")
    w = sub.add_parser("wear", help="穿戴：给人物的某个栏位设置单品或描述（- 表示脱下）")
    w.add_argument("name")
    w.add_argument("slot")
    w.add_argument("value", nargs="+")
    lo = sub.add_parser("loadout", help="定义套装：覆盖部分栏位，如 护甲=- 上衣=素色和服")
    lo.add_argument("name")
    lo.add_argument("set_name")
    lo.add_argument("changes", nargs="+")
    de = sub.add_parser("describe", help="预览拼出来的完整描述")
    de.add_argument("name")
    de.add_argument("--loadout")
    de.add_argument("--variant")
    sub.add_parser("slots", help="查看全部栏位")
    s = sub.add_parser("sheet", help="按定义生成设定图")
    s.add_argument("name")
    s.add_argument("--only", nargs="+", help="只生成这几张，如 front side")
    s.add_argument("--seed", type=int, help="覆盖定义里的种子")
    s.add_argument("--loadout", help="按某个套装生成（人物）")
    s.add_argument("--variant", help="按某个变体生成（地点）")
    s.add_argument("--model", default="z-image-turbo")
    s.add_argument("--quantize", type=int)
    im = sub.add_parser("import", help="导入外部生成的图片作为设定图")
    im.add_argument("name")
    im.add_argument("file")
    im.add_argument("--as", dest="as_name", help="保存成哪个名字（如 full、front），默认用原文件名")
    im.add_argument("--first", action="store_true", help="排到设定图最前面，出关键帧时优先作为参考")
    li = sub.add_parser("list", help="列出所有定义")
    li.add_argument("--kind", choices=list(KINDS))
    sh = sub.add_parser("show", help="查看一个定义")
    sh.add_argument("name")
    a = ap.parse_args()
    {"new": cmd_new, "wear": cmd_wear, "loadout": cmd_loadout, "describe": cmd_describe, "slots": cmd_slots,
     "sheet": cmd_sheet, "import": cmd_import, "list": cmd_list, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
