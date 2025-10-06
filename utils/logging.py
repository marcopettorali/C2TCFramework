import inspect
import os
import re
from rich.text import Text
from rich.console import Console
from rich.theme import Theme
from enum import Enum
from datetime import datetime

### LEGACY CODE
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

###


class LogLevel(Enum):
    ALL = "all"
    DEBUG = "debug"
    INFO = "info"
    FOCUS = "focus"
    WARNING = "warning"
    ERROR = "error"
    NONE = "none"

    def __le__(self, other):
        levels = list(LogLevel)
        return levels.index(self) <= levels.index(other)


_LOGGING_LEVEL = LogLevel.DEBUG


def set_logging_level(level: LogLevel):
    global _LOGGING_LEVEL

    if isinstance(level, str):
        level = LogLevel[level.upper()]
    _LOGGING_LEVEL = level


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
    # 1) bold+italic (***text***)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"[bold][italic]\1[/italic][/bold]", text)
    # 2) bold (**text**)
    text = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", text)
    # 3) underline (__text__) - prima del singolo underscore
    text = re.sub(r"__(.+?)__", r"[underline]\1[/underline]", text)
    # 4) italic con underscore singolo (_text_)
    text = re.sub(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)", r"[italic]\1[/italic]", text)
    # 5) italic con asterisco singolo (*text*)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"[italic]\1[/italic]", text)
    return text


def _merge_style(base_style: str, kw: dict) -> dict:
    """Pop 'style' e mergia con base_style; evita collisioni."""
    kw = dict(kw)  # copia
    user_style = kw.pop("style", "")
    kw["style"] = f"{base_style} {user_style}".strip() if user_style else base_style
    return kw


def _print_arg(arg, kw: dict, end: str, user_markup: bool | None):
    """
    Stampa un singolo argomento:
    - Se è stringa, applica markdown->rich e decide markup=True solo se il testo è stato trasformato.
    - Se non è stringa, stampa l'oggetto così com'è (nessun markup necessario).
    """
    # Non propagare 'end' in kw di base: lo passiamo esplicitamente
    base_kw = {k: v for k, v in kw.items() if k != "end" and k != "markup"}

    if isinstance(arg, Text):
        console.print(arg, end=end, **base_kw)
        return

    if isinstance(arg, str):
        original = arg
        transformed = _markdown_to_rich(original)
        if user_markup is None:
            # Attiva markup solo se la trasformazione ha cambiato la stringa
            use_markup = transformed != original
        else:
            # L'utente ha forzato markup: rispetta la scelta
            use_markup = bool(user_markup)
        console.print(transformed if use_markup else original, end=end, markup=use_markup, **base_kw)
        return

    # Oggetto non-stringa: stampalo direttamente; niente markup necessario
    console.print(arg, end=end, **base_kw)


def _base_print(prefix, base_style, *args, **kwargs):
    # Riga vuota se nessun argomento
    if len(args) == 0:
        console.print()
        return

    # Merge stili ed estrai eventuali controlli espliciti
    kw = _merge_style(base_style, kwargs)
    user_end = kw.get("end", None)
    user_markup = kw.get("markup", None)  # può essere True/False; se None decidiamo noi

    # Prefisso con timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    prefix_text = Text(f"[{timestamp} {prefix}]")
    console.print(
        prefix_text,
        end=" ",
        style=("dim " + kw.get("style", "")).strip(),
        **{k: v for k, v in kw.items() if k not in ("end", "markup", "style")},
    )

    # Stampa gli argomenti in sequenza, separati da spazio
    n = len(args)
    for i, arg in enumerate(args):
        is_last = i == n - 1
        end = user_end if user_end is not None else (" " if not is_last else " ")
        _print_arg(arg, kw, end=end, user_markup=user_markup)

    # Info del chiamante (due frame indietro: wrapper -> chiamante)
    frame = inspect.currentframe()
    if frame and frame.f_back:
        frame = frame.f_back
        if frame and frame.f_back:
            frame = frame.f_back
    abs_filename = os.path.abspath(frame.f_code.co_filename) if frame else ""
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        base_dir = os.getcwd()
    rel_filename = os.path.relpath(abs_filename, base_dir).lstrip("./")
    try:
        rel_filename = rel_filename  # mantieni il tuo slicing, se serve
    except Exception:
        pass
    line_number = frame.f_lineno if frame else 0
    function_name = frame.f_code.co_name if frame else "?"

    caller_info = Text(f"({rel_filename}:{line_number} - {function_name}())")
    console.print(caller_info, style=("dim " + kw.get("style", "")).strip())


# ===== Wrapper di livello =====
def debug(*args, **kwargs):
    if _LOGGING_LEVEL <= LogLevel.DEBUG:
        _base_print("DEBUG", "italic", *args, **kwargs)


def info(*args, **kwargs):
    if _LOGGING_LEVEL <= LogLevel.INFO:
        _base_print("INFO", "blue", *args, **kwargs)


def focus(*args, **kwargs):
    if _LOGGING_LEVEL <= LogLevel.FOCUS:
        _base_print("FOCUS", "green", *args, **kwargs)


def warning(*args, **kwargs):
    if _LOGGING_LEVEL <= LogLevel.WARNING:
        _base_print("WARNING", "bold yellow", *args, **kwargs)


def error(*args, **kwargs):
    if _LOGGING_LEVEL <= LogLevel.ERROR:
        _base_print("ERROR", "bold red", *args, **kwargs)
