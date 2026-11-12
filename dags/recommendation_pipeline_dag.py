"""Apache Airflow 2.x DAG for the recommendation pipeline.

It is generated from the SAME task registry (src/pipeline/tasks.py) that drives the Python workflow engine, so the two
orchestrators cannot drift apart. Copy the project folder next to Airflow and drop this file into $AIRFLOW_HOME/dags/.
`quality_gate` is a ShortCircuitOperator: when too many rows are rejected the downstream tasks are skipped.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from airflow import DAG                                            # noqa: E402
from airflow.operators.python import PythonOperator, ShortCircuitOperator   # noqa: E402

from src.config import load_config                                 # noqa: E402
from src.pipeline.orchestrator import QualityGateFailed            # noqa: E402
from src.pipeline.tasks import TASK_SPECS                          # noqa: E402

CFG = load_config(PROJECT)
DEFAULT_ARGS = {"owner": "data-team", "retries": 1, "retry_delay": timedelta(minutes=2)}


def _call(fn):
    def runner(**_ctx):
        fn(PROJECT, CFG)                  # results are written to disk / MLflow; nothing is pushed to XCom
    return runner


def _gate(fn):
    def runner(**_ctx):
        try:
            fn(PROJECT, CFG)
            return True
        except QualityGateFailed:
            return False                  # short-circuit: downstream tasks are skipped
    return runner


with DAG("recsys_data_pipeline", default_args=DEFAULT_ARGS, start_date=datetime(2026, 1, 1), schedule="@daily",
         catchup=False, tags=["recsys", "dm4ml"], doc_md=__doc__) as dag:
    ops = {}
    for name, fn, deps, desc in TASK_SPECS:
        if name == "quality_gate":
            ops[name] = ShortCircuitOperator(task_id=name, python_callable=_gate(fn), doc_md=desc)
        else:
            ops[name] = PythonOperator(task_id=name, python_callable=_call(fn), doc_md=desc)
    for name, _fn, deps, _desc in TASK_SPECS:
        for d in deps:
            ops[d] >> ops[name]
