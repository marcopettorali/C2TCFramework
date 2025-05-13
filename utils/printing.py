from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

# This is a simple wrapper around the rich library to maintain a consistent print interface

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
