import inspect
import os
import re
from rich.text import Text
from rich.console import Console
from rich.markup import escape
from rich.theme import Theme
from enum import Enum
from datetime import datetime


class LogLevel(Enum):
    ALL = "all"
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    NONE = "none"

    def __le__(self, other):
        levels = list(LogLevel)
        return levels.index(self) <= levels.index(other)


LOGGING_LEVEL = LogLevel.DEBUG

# THEME
my_theme = Theme(
    {
        "h1": "bold white on blue",
        "h2": "bold blue",
        "h3": "blue",
        "debug": "italic",
        "debug_warning": "italic yellow",
        "error": "bold red",
        "warning": "bold yellow",
        "info": "green",
        "success": "bold green",
    }
)


console = Console(theme=my_theme)
print = console.print

def _markdown_to_rich(text: str) -> str:
    """
    Convert markdown-like syntax to Rich markup.
    Supported:
      *italic*          -> [italic]...[/italic]
      _italic_          -> [italic]...[/italic]
      **bold**          -> [bold]...[/bold]
      ***bold+italic*** -> [bold][italic]...[/italic][/bold]
      __underline__     -> [underline]...[/underline]
    """
    # 1) bold+italic (***text***) - prima, così non viene "consumato" da ** o *
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"[bold][italic]\1[/italic][/bold]", text)

    # 2) bold (**text**)
    text = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", text)

    # 3) underline (__text__) - prima del singolo underscore per evitare conflitti
    text = re.sub(r"__(.+?)__", r"[underline]\1[/underline]", text)

    # 4) italic con underscore singolo (_text_)
    #    Lookaround per evitare __underline__ o _ dentro parole tipo snake_case
    text = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"[italic]\1[/italic]", text)

    # 5) italic con asterisco singolo (*text*)
    #    Lookaround per non catturare **bold**
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"[italic]\1[/italic]", text)

    return text

def _base_print(prefix, base_style, *args, **kwargs):
    kw = dict(kwargs)
    user_style = kw.pop("style", "")
    merged_style = f"{base_style} {user_style}".strip() if user_style else base_style
    kw["style"] = merged_style

    frame = inspect.currentframe().f_back.f_back
    abs_filename = os.path.abspath(frame.f_code.co_filename)
    rel_filename = os.path.relpath(abs_filename, os.path.abspath(os.path.dirname(__file__)))[3:]
    line_number = frame.f_lineno

    timestamp = datetime.now().strftime("%H:%M:%S.%f")

    prefix = Text(f"[{prefix} {timestamp}]")
    text = _markdown_to_rich(" ".join(str(arg) for arg in args))
    file_line = Text(f"({rel_filename}:{line_number})")

    console.print(prefix, end=" ", **kw)
    console.print(text, end=" ", **kw)
    console.print(file_line, style="dim " + kw.get("style", "")) #style="dim italic"

def debug(*args, **kwargs):
    if LOGGING_LEVEL <= LogLevel.DEBUG:
        _base_print("DEBUG", "italic", *args, **kwargs)

def info(*args, **kwargs):
    if LOGGING_LEVEL <= LogLevel.INFO:
        _base_print("INFO", "green", *args, **kwargs)

def success(*args, **kwargs):
    if LOGGING_LEVEL <= LogLevel.SUCCESS:
        _base_print("SUCCESS", "bold green", *args, **kwargs)

def warning(*args, **kwargs):
    if LOGGING_LEVEL <= LogLevel.WARNING:
        _base_print("WARNING", "bold yellow", *args, **kwargs)

def error(*args, **kwargs):
    if LOGGING_LEVEL <= LogLevel.ERROR:
        _base_print("ERROR", "bold red", *args, **kwargs)

