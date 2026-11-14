"""studio.py 的 AI 提示词工具：调用 OpenAI 兼容的大模型接口处理提示词。

配置：
  ai.json         内置的服务商和工具（随仓库提交，不含任何密钥）
  ai.local.json   你自己的服务商和工具（不进仓库），同名项覆盖 ai.json

密钥只从环境变量读取（服务商配置里的 api_key_env），不写进任何文件。
只用标准库发请求，PyTorch 和 MLX 两个环境里都能用。
"""
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    cfg = {"default_provider": None, "providers": {}, "tools": {}}
    for name in ("ai.json", "ai.local.json"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg["default_provider"] = data.get("default_provider") or cfg["default_provider"]
        cfg["providers"].update(data.get("providers", {}))
        cfg["tools"].update(data.get("tools", {}))
    return cfg


class AIError(Exception):
    pass


class AI:
    def __init__(self, provider=None):
        self.cfg = load_config()
        self.provider_name = provider or self.cfg["default_provider"]
        self.cache = {}

    @property
    def tools(self):
        return self.cfg["tools"]

    @property
    def providers(self):
        return self.cfg["providers"]

    def provider(self, name=None):
        name = name or self.provider_name
        if name not in self.providers:
            raise AIError(f"没有配置服务商“{name}”，可选：{', '.join(self.providers) or '无'}")
        return name, self.providers[name]

    def status(self, name=None):
        """返回 (服务商名, 是否已配置密钥, 密钥环境变量名)"""
        name, p = self.provider(name)
        env = p.get("api_key_env", "")
        return name, bool(os.environ.get(env)) or not env, env

    def run(self, tool_name, text="", image=None):
        """执行一个工具，返回模型输出的文本。相同输入走缓存。"""
        if tool_name not in self.tools:
            raise AIError(f"没有“{tool_name}”这个工具，/ai 查看列表")
        tool = self.tools[tool_name]
        if tool.get("input") == "image" and not image:
            raise AIError(f"“{tool_name}”需要一张图片：/ai {tool_name} 图片路径")
        provider_name, p = self.provider(tool.get("provider"))
        key = (tool_name, provider_name, text, image)
        if key in self.cache:
            return self.cache[key]

        user = tool.get("user", "{text}").replace("{text}", text)
        content = user
        if image:
            if not os.path.exists(image):
                raise AIError(f"找不到图片：{image}")
            mime = mimetypes.guess_type(image)[0] or "image/png"
            with open(image, "rb") as f:
                url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
            content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": user}]

        messages = [{"role": "system", "content": tool["system"]}] if tool.get("system") else []
        messages.append({"role": "user", "content": content})
        out = self._chat(p, messages, tool.get("temperature", 0.3))
        self.cache[key] = out
        return out

    def _chat(self, p, messages, temperature):
        env = p.get("api_key_env")
        api_key = os.environ.get(env, "") if env else ""
        if env and not api_key:
            raise AIError(f"没有设置环境变量 {env}。先在终端执行：export {env}=你的密钥，再启动 studio")
        body = json.dumps({"model": p["model"], "messages": messages, "temperature": temperature,
                           "stream": False}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = p["base_url"].rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=p.get("timeout", 60)) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise AIError(f"接口返回 HTTP {e.code}：{detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise AIError(f"连接 {url} 失败：{e}") from None
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise AIError(f"接口返回格式不对：{str(data)[:300]}") from None
        return strip_wrapping(text)


def strip_wrapping(text):
    """去掉模型常加的代码块、引号和“提示词：”前缀。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1] if "\n" in t else t.strip("`")
        t = t.rsplit("```", 1)[0]
    for prefix in ("提示词：", "提示词:", "Prompt:", "prompt:"):
        if t.startswith(prefix):
            t = t[len(prefix):]
    t = t.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1]
    return t.strip()
