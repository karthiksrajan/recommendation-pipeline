"""Command-line entry point:  python -m src.pipeline.run --base . --source-csv "Goodreadss Books.csv" [--resume] [--dry-run]

Cron example (every night at 02:00):  0 2 * * *  cd /path/to/recommendation-pipeline && python -m src.pipeline.run --base . --resume
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DEFAULT_CFG, Paths, load_config, save_config  # noqa: E402
from src.pipeline.orchestrator import render_run_graph  # noqa: E402
from src.pipeline.tasks import build_workflow  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end recommendation data pipeline")
    ap.add_argument("--base", default=".", help="project directory (data/, models/, reports/ are created inside)")
    ap.add_argument("--source-csv", help="Goodreads items CSV (stored in the config on first use)")
    ap.add_argument("--resume", action="store_true", help="skip tasks that already succeeded with the same config")
    ap.add_argument("--dry-run", action="store_true", help="print the execution plan only")
    a = ap.parse_args(argv)
    base = Path(a.base).resolve()
    P = Paths(base).ensure()
    cfg_path = P.configs / "pipeline_config.json"
    cfg = load_config(base) if cfg_path.exists() else dict(DEFAULT_CFG)
    if a.source_csv:
        cfg["source_csv"] = str(Path(a.source_csv).resolve())
    save_config(base, cfg)
    wf = build_workflow(base, cfg)
    if a.dry_run:
        for n, lvl, deps in wf.dry_run():
            print(f"level {lvl:>2} | {n:18s} <- {deps}")
        return 0
    rec = wf.run(resume=a.resume, mode="cli")
    render_run_graph(rec, P.reports / "orchestration_run.png")
    print(json.dumps({k: rec[k] for k in ("run_id", "status", "seconds")}))
    return 0 if rec["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
