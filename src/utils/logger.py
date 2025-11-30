"""
Logger utility using Rich for clean console output
"""

from rich import print as rprint
from rich.console import Console

console = Console()


def info(msg):
    """Print info message"""
    rprint(f"[bold cyan][INFO][/bold cyan] {msg}")


def success(msg):
    """Print success message"""
    rprint(f"[bold green][OK][/bold green] {msg}")


def error(msg):
    """Print error message"""
    rprint(f"[bold red][ERROR][/bold red] {msg}")


def warning(msg):
    """Print warning message"""
    rprint(f"[bold yellow][WARNING][/bold yellow] {msg}")


def debug(msg):
    """Print debug message"""
    rprint(f"[dim][DEBUG][/dim] {msg}")

