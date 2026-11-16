"""studio.py 和 video.py 共用的命令补全菜单（prompt_toolkit）。"""
import os
import sys


def make_prompt_session(commands, arg_choices, history_path, label):
    """有 TTY 且装了 prompt_toolkit 时返回带补全菜单的输入函数，否则返回 None。

    commands: {命令: 说明}；arg_choices(命令) -> [(参数值, 说明)]；history_path: 输入历史文件。
    输入 / 弹出命令菜单；选中命令后自动弹出参数菜单；选中命令回车只填入，选中参数回车直接执行。
    """
    if not sys.stdin.isatty():
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return None

    class MenuCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            if " " not in text:  # 补命令
                for cmd, desc in commands.items():
                    if cmd.startswith(text):
                        yield Completion(cmd + " ", start_position=-len(text), display=cmd, display_meta=desc)
                return
            cmd, _, arg = text.partition(" ")
            for value, desc in arg_choices(cmd):
                if value.startswith(arg.strip()):
                    yield Completion(value, start_position=-len(arg), display_meta=desc)

    from prompt_toolkit.filters import completion_is_selected
    from prompt_toolkit.application import get_app
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    from prompt_toolkit.filters import Condition, has_completions

    @Condition
    def partial_command():
        text = get_app().current_buffer.text
        return text.startswith("/") and " " not in text and text not in commands

    @kb.add("enter", filter=has_completions & ~completion_is_selected & partial_command)
    def _(event):
        # 输入 "/ne" 回车：补成第一个匹配的命令（/neg ），不提交半截命令
        buf = event.current_buffer
        buf.apply_completion(buf.complete_state.completions[0])

    @kb.add("enter", filter=completion_is_selected)
    def _(event):
        # 菜单里选中的是命令（以空格结尾）：只填入，等参数；选中的是参数：填入并执行
        buf = event.current_buffer
        completion = buf.complete_state.current_completion
        buf.apply_completion(completion)
        if not completion.text.endswith(" "):
            buf.validate_and_handle()

    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    session = PromptSession(
        completer=MenuCompleter(),
        complete_while_typing=True,
        key_bindings=kb,
        history=FileHistory(history_path),
    )

    # 选中 "/style " 这类命令后，立刻弹出它的参数菜单
    def on_change(buf):
        cmd, sep, arg = buf.text.partition(" ")
        if sep and not arg and arg_choices(cmd) and not buf.complete_state:
            buf.start_completion(select_first=False)

    session.default_buffer.on_text_changed += on_change
    return lambda: session.prompt(f"\n{label}> ")
