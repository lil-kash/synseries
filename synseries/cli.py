"""
synseries.cli
Command-line interface for synthetic time series generation, spec coding, and validation.
"""

import sys
import os
import argparse
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from synseries.nlp.prompt_compiler import PromptCompiler
from synseries.core.engine import TimeSeriesGeneratorEngine
from synseries.core.spec import TimeSeriesDatasetSpec
from synseries.validation.authenticity import DomainAuthenticityValidator
from synseries.validation.reporter import ValidationReporter

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="SynSeries: Domain-Valid Synthetic Time-Series Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate a synthetic time-series dataset")
    gen_parser.add_argument("--prompt", "-p", type=str, help="Natural language prompt (e.g. '12 hours of ICU patient data with sepsis')")
    gen_parser.add_argument("--spec", "-s", type=str, help="Path to YAML or JSON specification file")
    gen_parser.add_argument("--domain", "-d", type=str, choices=["medical", "finance", "iot", "energy", "web", "climate", "custom"], help="Domain preset")
    gen_parser.add_argument("--type", "-t", type=str, help="Sub-type preset (e.g. ecg, icu, cgm, ohlcv, portfolio, vibration, rul)")
    gen_parser.add_argument("--duration", type=str, help="Duration (e.g. 24h, 7d, 60s)")
    gen_parser.add_argument("--steps", "-n", type=int, help="Number of time steps")
    gen_parser.add_argument("--output", "-o", type=str, default="synthetic_dataset.csv", help="Output file path (.csv, .json, .parquet)")
    gen_parser.add_argument("--validate", action="store_true", help="Automatically run domain validity tests after generation")
    gen_parser.add_argument("--plot", action="store_true", help="Save visual plot of generated series")

    # Command: spec (gives the spec coding)
    spec_parser = subparsers.add_parser("spec", help="Generate declarative YAML/JSON specification code from prompt")
    spec_parser.add_argument("prompt", type=str, help="Natural language description of desired dataset")
    spec_parser.add_argument("--format", "-f", choices=["yaml", "json"], default="yaml", help="Output format")
    spec_parser.add_argument("--save", type=str, help="Save spec directly to a file")

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate an existing dataset against domain invariants")
    val_parser.add_argument("file", type=str, help="Path to CSV dataset")
    val_parser.add_argument("--domain", "-d", required=True, help="Domain to test against (e.g. medical, finance, iot, energy)")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Launch interactive web dashboard")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "spec":
        handle_spec(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "validate":
        handle_validate(args)
    elif args.command == "ui":
        handle_ui()


def handle_spec(args):
    prompt = args.prompt
    console.print(f"[bold cyan]Compiling specification for prompt:[/bold cyan] '{prompt}'")
    spec = PromptCompiler.compile(prompt)

    if args.format == "yaml":
        out_str = spec.to_yaml()
        syntax = Syntax(out_str, "yaml", theme="monokai", line_numbers=True)
    else:
        out_str = spec.to_json()
        syntax = Syntax(out_str, "json", theme="monokai", line_numbers=True)

    console.print(Panel(syntax, title=f"Compiled Specification ({spec.domain.upper()})", border_style="green"))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(out_str)
        console.print(f"[bold green]Saved specification to:[/bold green] {args.save}")


def handle_generate(args):
    spec = None
    if args.spec:
        console.print(f"[cyan]Loading specification from file:[/cyan] {args.spec}")
        spec = TimeSeriesDatasetSpec.from_file(args.spec)
    elif args.prompt:
        console.print(f"[cyan]Compiling prompt:[/cyan] '{args.prompt}'")
        spec = PromptCompiler.compile(args.prompt)
    elif args.domain:
        prompt_fallback = f"Generate {args.domain} dataset"
        if args.type:
            prompt_fallback += f" with type {args.type}"
        spec = PromptCompiler.compile(prompt_fallback)
    else:
        console.print("[red]Error: Please provide --prompt, --spec, or --domain[/red]")
        sys.exit(1)

    if args.duration:
        spec.duration = args.duration
    if args.steps:
        spec.n_steps = args.steps

    console.print(f"[bold green]Generating time series for domain:[/bold green] [yellow]{spec.domain.upper()}[/yellow]")
    df = TimeSeriesGeneratorEngine.generate_from_spec(spec)

    # Save output
    output_path = args.output
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path)
    elif output_path.endswith(".json"):
        df.to_json(output_path, orient="records", date_format="iso", indent=2)
    else:
        df.to_csv(output_path)

    console.print(f"[bold green]Dataset generated successfully:[/bold green] [bold underline]{output_path}[/bold underline]")

    # Display preview table
    preview_table = Table(title=f"Sample Preview: {spec.name} ({len(df)} rows, {len(df.columns)} columns)")
    preview_table.add_column("Timestamp", style="cyan")
    for col in df.columns[:6]:
        preview_table.add_column(col, style="magenta")

    for idx, row in df.head(5).iterrows():
        t_str = str(idx)
        vals = [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in row.iloc[:6]]
        preview_table.add_row(t_str, *vals)

    console.print(preview_table)

    # Optional validation
    if args.validate:
        run_validation_display(df, spec.domain)

    # Optional plot
    if args.plot:
        save_plot(df, spec.name)


def handle_validate(args):
    console.print(f"[cyan]Validating dataset file:[/cyan] {args.file} against domain [yellow]{args.domain}[/yellow]")
    df = pd.read_csv(args.file, index_col=0, parse_dates=True)
    run_validation_display(df, args.domain)


def run_validation_display(df: pd.DataFrame, domain: str):
    results = DomainAuthenticityValidator.validate(df, domain)
    table = Table(title=f"Domain Validity Matrix - {domain.upper()} (Score: {results['score_pct']}%)")
    table.add_column("Check Name", style="bold")
    table.add_column("Category")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    for c in results["checks"]:
        st = "[bold green]PASS[/bold green]" if "PASS" in c["status"] else "[bold red]FAIL[/bold red]"
        table.add_row(c["name"], c["category"], st, c["details"])

    console.print(table)
    overall_color = "green" if results["overall_status"] == "PASS" else "red"
    console.print(f"[{overall_color}]Overall Verdict: {results['overall_status']} ({results['score_pct']}% compliance)[/{overall_color}]")


def save_plot(df: pd.DataFrame, title: str):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    numeric_cols = df.select_dtypes(include=[float, int]).columns
    n_cols_to_plot = min(4, len(numeric_cols))
    if n_cols_to_plot == 0:
        return

    fig, axes = plt.subplots(n_cols_to_plot, 1, figsize=(10, 2.5 * n_cols_to_plot), sharex=True)
    if n_cols_to_plot == 1:
        axes = [axes]

    for i in range(n_cols_to_plot):
        col = numeric_cols[i]
        axes[i].plot(df.index, df[col], label=col, color=f"C{i}", linewidth=1.2)
        axes[i].set_ylabel(col)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"Synthetic Time Series: {title}", fontsize=14, y=0.99)
    plt.tight_layout()
    plot_file = "dataset_plot.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    console.print(f"[bold green]Visual plot saved to:[/bold green] {plot_file}")


def handle_ui():
    console.print("[bold cyan]Launching Streamlit dashboard...[/bold cyan]")
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    os.system(f"streamlit run {app_path}")


if __name__ == "__main__":
    main()
