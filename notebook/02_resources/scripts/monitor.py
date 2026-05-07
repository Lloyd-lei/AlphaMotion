"""monitor.py — live multi-variant training dashboard.

Run in a SEPARATE terminal alongside training:
    cd notebook
    python 02_resources/scripts/monitor.py

Reads:
    - notebook/02_resources/checkpoints/<variant>/status.json   (per-variant heartbeat)
    - notebook/02_resources/checkpoints/<variant>/tensorboard/  (TB event files)
    - nvitop API                                                 (live GPU + per-PID stats)

Displays (rich.live):
    - Per-variant table: status / step / loss / step_ms / ETA / GPU mem / PID utilization
    - GPU summary header: device / total mem / total util
    - Footer: TB launch hint + scan interval

Independent of trainers. Safe to start before, during, or after training.
Trainers heartbeat status.json; monitor never writes anything.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import nvitop
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------- TB scalar reader (latest value only) ----------------------------

def latest_tb_scalar(tb_dir: Path, tag: str) -> Optional[float]:
    """Return the most recent value for `tag` from any TB event file in tb_dir,
    or None. Imports lazily so monitor still runs if TB isn't installed."""
    if not tb_dir.exists():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 1024})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        events = ea.Scalars(tag)
        return float(events[-1].value) if events else None
    except Exception:
        return None


# ---------- per-variant status -----------------------------------------------

def read_variant_status(ckpt_dir: Path, variant: str) -> Dict:
    """Combine status.json + recent loss_history into a flat dict."""
    out = {"variant": variant, "status": "(no status)", "loss_recent_history": []}
    sp = ckpt_dir / variant / "status.json"
    if sp.exists():
        try:
            out.update(json.loads(sp.read_text()))
        except Exception as e:
            out["status"] = f"(json error: {e})"
    # Sparkline source: last N entries of loss_history.json
    hp = ckpt_dir / variant / "loss_history.json"
    if hp.exists():
        try:
            hist = json.loads(hp.read_text()).get("entries", [])
            out["loss_recent_history"] = [e["loss"] for e in hist[-30:]]
        except Exception:
            pass
    return out


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 25) -> str:
    """Unicode bar chart of last `width` values."""
    if not values:
        return "─" * width
    vs = values[-width:]
    lo, hi = min(vs), max(vs)
    span = max(hi - lo, 1e-12)
    return "".join(_SPARK_BLOCKS[min(7, int((v - lo) / span * 7))] for v in vs)


# ---------- nvitop GPU snapshot ---------------------------------------------

def gpu_summary():
    """Return [(name, total_gb, used_gb, util_pct), ...]"""
    devs = nvitop.Device.all()
    return [(d.name(), d.memory_total() / 1024**3,
             d.memory_used() / 1024**3, d.gpu_utilization())
            for d in devs]


def gpu_processes_by_pid():
    """Return {pid: (mem_gb, gpu_util_pct, command_short)}"""
    out = {}
    for dev in nvitop.Device.all():
        for proc in dev.processes().values():
            try:
                out[proc.pid] = (
                    proc.gpu_memory() / 1024**3,
                    proc.gpu_sm_utilization() if hasattr(proc, "gpu_sm_utilization") else 0,
                    (proc.command() or "")[:40],
                )
            except Exception:
                pass
    return out


# ---------- rendering --------------------------------------------------------

def render_dashboard(ckpt_root: Path, variants: list, refresh_s: float = 2.0):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="table"),
        Layout(name="footer", size=3),
    )

    gpu = gpu_summary()
    head_lines = []
    for name, tot, used, util in gpu:
        head_lines.append(f"[bold cyan]{name}[/]   "
                          f"mem [yellow]{used:.1f} / {tot:.1f} GB[/]   "
                          f"util [green]{util:>3d}%[/]")
    layout["header"].update(Panel(Text.from_markup("\n".join(head_lines)),
                                   title="GPU"))

    procs = gpu_processes_by_pid()

    # Table
    table = Table(expand=True, show_lines=False)
    table.add_column("variant",  style="bold")
    table.add_column("status",   justify="center")
    table.add_column("step",     justify="right")
    table.add_column("loss",     justify="right")
    table.add_column("recent loss curve", justify="left", no_wrap=True)
    table.add_column("ms/step",  justify="right")
    table.add_column("ETA",      justify="right")
    table.add_column("GPU mem",  justify="right")
    table.add_column("PID",      justify="right")

    for v in variants:
        s = read_variant_status(ckpt_root, v)
        status_str = s.get("status", "?")
        status_styled = {
            "running":      "[bold yellow]running[/]",
            "done":         "[bold green]done[/]",
            "interrupted":  "[bold red]interrupt[/]",
            "already_done": "[dim green]already[/]",
        }.get(status_str, f"[dim]{status_str}[/]")

        step_str = f"{s.get('step', '-')} / {s.get('n_steps', '-')}"
        loss_str = (f"{s['loss_recent']:.4f}" if isinstance(s.get('loss_recent'), (int, float)) else "-")
        ms_str   = (f"{s['step_ms']:.0f}"     if isinstance(s.get('step_ms'),     (int, float)) else "-")
        eta = s.get("eta_s")
        eta_str = (f"{eta/60:.1f} min" if isinstance(eta, (int, float)) and eta > 60
                    else (f"{eta:.0f} s" if isinstance(eta, (int, float)) else "-"))
        mem_str  = (f"{s['gpu_mem_gb']:.1f} GB" if isinstance(s.get('gpu_mem_gb'), (int, float)) else "-")
        pid = s.get("pid", "-")

        # Loss sparkline: last 25 points from loss_history.json
        spark_values = s.get("loss_recent_history", [])
        spark_str = sparkline(spark_values, width=25) if spark_values else "─" * 25
        # Color-code: green if descending overall, yellow if flat, red if rising
        if len(spark_values) >= 5:
            head, tail = spark_values[0], spark_values[-1]
            color = "green" if tail < head * 0.95 else ("red" if tail > head * 1.05 else "yellow")
            spark_styled = f"[{color}]{spark_str}[/]"
        else:
            spark_styled = f"[dim]{spark_str}[/]"

        table.add_row(v, status_styled, step_str, loss_str, spark_styled,
                      ms_str, eta_str, mem_str, str(pid))

    layout["table"].update(Panel(table, title="Variants"))

    layout["footer"].update(Panel(Text.from_markup(
        f"[dim]TB:[/] tensorboard --logdir notebook/02_resources/checkpoints/   "
        f"[dim]| refresh:[/] {refresh_s}s   "
        f"[dim]| ckpt root:[/] {ckpt_root}"
    )))

    return layout


# ---------- main loop --------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=Path,
                   default=Path("notebook/02_resources/checkpoints"))
    p.add_argument("--variants", nargs="*", default=None,
                   help="explicit variant list; default = config variants")
    p.add_argument("--refresh", type=float, default=2.0,
                   help="seconds between updates")
    args = p.parse_args()

    if not args.ckpt_root.exists():
        print(f"warning: {args.ckpt_root} doesn't exist yet — will retry as it appears")

    if args.variants is None:
        cfg = Path("notebook/01_resources/configs/variants.json")
        if cfg.exists():
            args.variants = list(json.loads(cfg.read_text()).keys())
        else:
            args.variants = ["full", "opm_only", "triangle_only", "pair_static",
                              "axial_only", "baseline"]

    console = Console()
    console.print(f"[bold]monitor.py[/]  watching {len(args.variants)} variants under {args.ckpt_root}/")
    console.print("Ctrl-C to exit (does NOT affect trainers)\n")

    try:
        with Live(render_dashboard(args.ckpt_root, args.variants, args.refresh),
                   refresh_per_second=1/args.refresh, console=console) as live:
            while True:
                time.sleep(args.refresh)
                live.update(render_dashboard(args.ckpt_root, args.variants, args.refresh))
    except KeyboardInterrupt:
        console.print("\n[bold]monitor exiting[/]")


if __name__ == "__main__":
    main()
