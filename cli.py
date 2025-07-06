import subprocess
import questionary
from rich.console import Console
from rich.panel import Panel
import typer
from pathlib import Path
import os

app = typer.Typer()
console = Console()

TASKS = [
    {
        "name": "lint",
        "desc": "Run Ruff linter and auto-fix common issues."
    },
    {
        "name": "test",
        "desc": "Run pytest with PYTHONPATH set to project root."
    },
    {
        "name": "exit",
        "desc": "Exit the CLI tool."
    }
]

PROJECT_ROOT = Path(__file__).resolve().parent

def run_ruff():
    console.print(Panel.fit("[cyan]Running: ruff check . --fix[/cyan]"))
    try:
        subprocess.run(["ruff", "check", ".", "--fix"], cwd=PROJECT_ROOT, check=True)
        console.print("[green]Ruff linting completed successfully.[/green]")
    except FileNotFoundError:
        console.print("[red]ruff not installed. Please install with `pip install ruff`.[/red]")
    except subprocess.CalledProcessError:
        console.print("[red]Ruff found issues that may need manual review.[/red]")

def run_pytest():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    console.print(Panel.fit("[cyan]Running: pytest with PYTHONPATH set automatically[/cyan]"))
    try:
        subprocess.run(["pytest"], cwd=PROJECT_ROOT, env=env, check=True)
        console.print("[green]Pytest completed successfully.[/green]")
    except subprocess.CalledProcessError:
        console.print("[red]Pytest failed.[/red]")

def main_menu():
    while True:
        options = [
            questionary.Choice(
                title=f"{task['name']} - {task['desc']}",
                value=task["name"]
            ) for task in TASKS
        ]

        selected = questionary.select(
            "Choose a task to perform:",
            choices=options
        ).ask()

        if selected == "exit":
            console.print("[bold blue]Exiting. Goodbye![/bold blue]")
            break

        confirm = questionary.confirm(f"Proceed with `{selected}`?").ask()

        if confirm:
            if selected == "lint":
                run_ruff()
            elif selected == "test":
                run_pytest()
        else:
            console.print("[yellow]Cancelled. Returning to menu.[/yellow]")

@app.command()
def interactive():
    """Start the interactive CLI."""
    console.print("[bold green]Welcome to the Project CLI Tool![/bold green]")
    main_menu()

if __name__ == "__main__":
    app()
