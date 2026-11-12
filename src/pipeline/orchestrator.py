"""A small, dependency-aware workflow engine (the 'clearly structured scheduled Python workflow' option of the assignment).

Features
* tasks declare their upstream dependencies -> executed in topological order (cycles / unknown deps are rejected)
* retries with back-off, per-task timing, status and error capture, file + console logging
* quality gate: a task may raise QualityGateFailed -> every downstream task is marked `skipped` (run status `stopped_by_gate`)
* upstream failure -> downstream tasks are `upstream_failed` (never run on bad inputs)
* resume / idempotence: with resume=True, tasks that succeeded in the previous run with the SAME config are `cached`
* every run is persisted as JSON (reports/orchestration/) and can be drawn as a DAG-run picture (see render_run_graph)
* run_scheduled() repeats the workflow on a fixed interval (cron / Airflow are the production alternatives)
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.config import Paths, config_hash

log = logging.getLogger("recsys.orchestrator")


class QualityGateFailed(Exception):
    """Raised by a gate task: stops the pipeline cleanly (not a crash)."""


@dataclass
class Task:
    name: str
    fn: Callable
    deps: list = field(default_factory=list)
    retries: int = 1
    description: str = ""


class Workflow:
    def __init__(self, name: str, base, cfg: dict):
        self.name, self.base, self.cfg = name, Path(base), cfg
        self.P = Paths(base).ensure()
        self.tasks: dict[str, Task] = {}

    # ------------------------------------------------------------------ definition
    def add(self, name, fn, deps=(), retries=1, description=""):
        if name in self.tasks:
            raise ValueError(f"duplicate task {name}")
        self.tasks[name] = Task(name, fn, list(deps), retries, description)
        return self

    def topological_order(self):
        """Kahn's algorithm (stable: keeps declaration order among independent tasks)."""
        for t in self.tasks.values():
            for d in t.deps:
                if d not in self.tasks:
                    raise ValueError(f"task '{t.name}' depends on unknown task '{d}'")
        indeg = {n: len(t.deps) for n, t in self.tasks.items()}
        order, ready = [], [n for n in self.tasks if indeg[n] == 0]
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m, t in self.tasks.items():
                if n in t.deps:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        ready.append(m)
        if len(order) != len(self.tasks):
            raise ValueError("cycle detected in the workflow graph")
        return order

    def levels(self):
        lvl = {}
        for n in self.topological_order():
            lvl[n] = 1 + max([lvl[d] for d in self.tasks[n].deps], default=-1)
        return lvl

    # ------------------------------------------------------------------ execution
    def _previous_run(self):
        p = self.P.orchestration / "latest_run.json"
        return json.loads(p.read_text()) if p.exists() else None

    def run(self, resume: bool = False, mode: str = "orchestrated", fail_fast: bool = True) -> dict:
        order = self.topological_order()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        chash = config_hash(self.cfg)
        os.environ["RECSYS_RUN_ID"], os.environ["RECSYS_RUN_MODE"] = run_id, mode
        fh = logging.FileHandler(self.P.orchestration / "pipeline.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s"))
        logging.getLogger("recsys").addHandler(fh)
        prev = self._previous_run() if resume else None
        prev_ok = {t["name"] for t in prev["tasks"] if t["status"] in ("success", "cached")} if prev and prev.get("config_hash") == chash else set()

        rec = {"run_id": run_id, "workflow": self.name, "mode": mode, "config_hash": chash, "started": datetime.now().isoformat(timespec="seconds"),
               "status": "running", "tasks": []}
        status, dirty = {}, set()
        stopped_by_gate = failed = False
        t_run = time.time()
        log.info(f"=== run {run_id} | workflow '{self.name}' | {len(order)} tasks | resume={resume} ===")
        for n in order:
            t = self.tasks[n]
            entry = {"name": n, "deps": t.deps, "description": t.description, "status": None, "attempts": 0, "seconds": 0.0, "start": None, "end": None,
                     "message": "", "result": None}
            blocked = [d for d in t.deps if status[d] in ("failed", "upstream_failed", "skipped", "gate_failed")]
            if blocked:
                # blocked by a quality gate -> `skipped` (a controlled stop); blocked by a crash -> `upstream_failed`
                crashed = any(status[d] in ("failed", "upstream_failed") for d in blocked)
                entry["status"] = "upstream_failed" if crashed else "skipped"
                entry["message"] = f"not run: upstream '{blocked[0]}' is {status[blocked[0]]}"
            elif resume and n in prev_ok and not any(d in dirty for d in t.deps):
                entry["status"], entry["message"] = "cached", "succeeded in the previous run with identical config - skipped"
            else:
                entry["start"] = datetime.now().isoformat(timespec="seconds")
                t0 = time.time()
                for attempt in range(1, t.retries + 2):
                    entry["attempts"] = attempt
                    try:
                        log.info(f"> {n} (attempt {attempt})")
                        entry["result"] = t.fn(self.base, self.cfg)
                        entry["status"] = "success"
                        break
                    except QualityGateFailed as e:
                        entry["status"], entry["message"] = "gate_failed", str(e)
                        stopped_by_gate = True
                        log.warning(f"QUALITY GATE FAILED in {n}: {e}")
                        break
                    except Exception as e:                      # noqa: BLE001
                        entry["message"] = f"{type(e).__name__}: {e}"
                        entry["traceback"] = traceback.format_exc(limit=6)
                        log.error(f"{n} failed (attempt {attempt}): {entry['message']}")
                        if attempt <= t.retries:
                            time.sleep(min(2 * attempt, 5))
                        else:
                            entry["status"] = "failed"
                            failed = True
                entry["seconds"] = round(time.time() - t0, 2)
                entry["end"] = datetime.now().isoformat(timespec="seconds")
                dirty.add(n)
            status[n] = entry["status"]
            rec["tasks"].append(entry)
            if fail_fast and entry["status"] == "failed":
                pass                                            # downstream tasks are marked upstream_failed by the loop
        rec["seconds"] = round(time.time() - t_run, 2)
        rec["ended"] = datetime.now().isoformat(timespec="seconds")
        rec["status"] = "failed" if failed else ("stopped_by_gate" if stopped_by_gate else "success")
        (self.P.orchestration / f"run_{run_id}.json").write_text(json.dumps(rec, indent=2, default=str))
        (self.P.orchestration / "latest_run.json").write_text(json.dumps(rec, indent=2, default=str))
        logging.getLogger("recsys").removeHandler(fh); fh.close()
        log.info(f"=== run {run_id} finished: {rec['status']} in {rec['seconds']}s ===")
        return rec

    def run_scheduled(self, every_seconds: float, max_runs: int = 1, resume: bool = True):
        """Repeat the workflow on a fixed interval (demo of a scheduled run; use cron / Airflow in production)."""
        out = []
        for i in range(max_runs):
            out.append(self.run(resume=resume if i else False, mode="scheduled"))
            if i < max_runs - 1:
                time.sleep(every_seconds)
        return out

    def dry_run(self):
        """Execution plan without running anything."""
        lvl = self.levels()
        return [(n, lvl[n], self.tasks[n].deps) for n in self.topological_order()]


STATUS_COLORS = {"success": "#2e9e5b", "cached": "#6c9bd1", "failed": "#d64545", "upstream_failed": "#e8a0a0",
                 "skipped": "#c9c9c9", "gate_failed": "#f0a030"}


def render_run_graph(rec: dict, path):
    """Draw the DAG of a run (left -> right) with status colours + a duration panel: the 'successful run' screenshot."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    tasks = {t["name"]: t for t in rec["tasks"]}
    lvl = {}
    for t in rec["tasks"]:                                      # tasks are stored in topological order
        lvl[t["name"]] = 1 + max([lvl[d] for d in t["deps"]], default=-1)
    cols = {}
    for n, l in lvl.items():
        cols.setdefault(l, []).append(n)
    max_rows = max(len(v) for v in cols.values())
    fig = plt.figure(figsize=(19, 9.2))
    ax = fig.add_axes([0.01, 0.38, 0.98, 0.54]); ax.axis("off")
    ax.set_xlim(-0.65, len(cols) - 0.35); ax.set_ylim(-0.7, max_rows - 0.3)
    pos = {}
    for l, names in cols.items():
        for r, n in enumerate(names):
            pos[n] = (l, (max_rows - 1) / 2 + (len(names) - 1) / 2 - r)
    w, h = 0.86, 0.5
    for n, t in tasks.items():
        for d in t["deps"]:
            (x1, y1), (x2, y2) = pos[d], pos[n]
            ax.add_patch(FancyArrowPatch((x1 + w / 2, y1), (x2 - w / 2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#777", alpha=0.8,
                                         connectionstyle="arc3,rad=0.0"))
    for n, t in tasks.items():
        x, y = pos[n]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=STATUS_COLORS.get(t["status"], "#fff"), ec="#333", lw=1))
        ax.text(x, y + 0.08, n, ha="center", va="center", fontsize=7.6, fontweight="bold", color="white" if t["status"] in ("success", "failed") else "#111")
        ax.text(x, y - 0.13, f"{t['status']} | {t['seconds']:.1f}s", ha="center", va="center", fontsize=6.6, color="white" if t["status"] in ("success", "failed") else "#111")
    ax.set_title(f"Workflow '{rec['workflow']}' - run {rec['run_id']} ({rec['mode']}) - status: {rec['status'].upper()} in {rec['seconds']:.1f}s - "
                 f"{sum(t['status'] == 'success' for t in rec['tasks'])} success, {sum(t['status'] == 'cached' for t in rec['tasks'])} cached, "
                 f"{sum(t['status'] in ('skipped', 'upstream_failed', 'gate_failed', 'failed') for t in rec['tasks'])} not-successful tasks", fontsize=11, fontweight="bold")
    # Gantt-style duration panel
    ax2 = fig.add_axes([0.07, 0.06, 0.9, 0.25])
    ran = [t for t in rec["tasks"] if t["start"]]
    t_zero = datetime.fromisoformat(rec["started"])
    for i, t in enumerate(ran):
        s = (datetime.fromisoformat(t["start"]) - t_zero).total_seconds()
        ax2.barh(i, max(t["seconds"], 0.15), left=s, color=STATUS_COLORS.get(t["status"], "#999"), edgecolor="#333", lw=0.5)
        ax2.text(s + max(t["seconds"], 0.15) + 0.3, i, f"{t['seconds']:.1f}s", va="center", fontsize=6.5)
    ax2.set_yticks(range(len(ran))); ax2.set_yticklabels([t["name"] for t in ran], fontsize=6.5); ax2.invert_yaxis()
    ax2.set_xlabel("seconds since start of run"); ax2.set_title("Task timeline (executed tasks)", fontsize=9, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#333") for c in STATUS_COLORS.values()]
    fig.legend(handles, list(STATUS_COLORS), loc="upper center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.375))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig
