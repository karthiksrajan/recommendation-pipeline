"""Stage 2 - Automated data validation (Pandera schemas + custom rules).

Rule severity
* reject  -> row cannot be trusted / joined: moved to the error dataset (data/errors/)
* warning -> row is usable but a field is poor: logged and repaired in the preparation stage
Tables are validated in dependency order: users -> items -> interactions (referential integrity).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa      # pandera >= 0.24
except ImportError:                  # pragma: no cover
    import pandera as pa

from src.common.genres import GenreParser
from src.common.log import get_logger
from src.config import Paths

log = get_logger("validation")

MIN_DATE = pd.Timestamp("2000-01-01")
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
LOG_COLS = ["table", "row_index", "column", "rule", "value", "severity"]
HIST_COLS = [f"rated_{k}" for k in range(1, 6)]


# ----------------------------------------------------------------------------- helpers
def run_schema(df, schema, table, severity):
    """Lazy Pandera validation -> failures in a tidy log format."""
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=LOG_COLS)
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
    schema_level = fc[fc["index"].isna()]
    if len(schema_level):
        log.info(f"[{table}] schema-level findings (not row-specific): {schema_level[['column', 'check']].drop_duplicates().values.tolist()}")
    fc = fc[fc["index"].notna()]
    return pd.DataFrame({"table": table, "row_index": fc["index"].astype(int).to_numpy(), "column": fc["column"].to_numpy(),
                         "rule": fc["check"].astype(str).to_numpy(), "value": fc["failure_case"].astype(str).to_numpy(), "severity": severity})


def flag(mask, table, rule, column, values, severity="reject"):
    """Boolean mask -> log rows."""
    idx = np.flatnonzero(np.asarray(mask))
    return pd.DataFrame({"table": table, "row_index": idx, "column": column, "rule": rule,
                         "value": np.asarray(values, dtype=object)[idx].astype(str), "severity": severity})


def typed_check(raw, parsed, table, column):
    """Raw value present but could not be parsed into the expected dtype => 'dtype_mismatch'."""
    return flag(raw.notna().to_numpy() & parsed.isna().to_numpy(), table, "dtype_mismatch", column, raw.to_numpy())


def drop_null_dupes(lg):
    """A failed parse is reported once as dtype_mismatch (not additionally as 'null')."""
    dt = lg.loc[lg["rule"] == "dtype_mismatch", ["table", "row_index", "column"]].assign(_d=1)
    m = lg.merge(dt, on=["table", "row_index", "column"], how="left")
    return lg[~((m["rule"] == "not_nullable").to_numpy() & m["_d"].notna().to_numpy())].reset_index(drop=True)


# ----------------------------------------------------------------------------- per-table validators
def validate_users(users_raw, end_date):
    ut = users_raw.copy()
    ut["signup_date"] = pd.to_datetime(users_raw["signup_date"], errors="coerce", format="%Y-%m-%d")
    hard = pa.DataFrameSchema({
        "user_id": pa.Column(str, [pa.Check(lambda s: s.str.match(r"^U\d{5}$"), name="user_id_format")], nullable=False),
        "signup_date": pa.Column(nullable=False, checks=[pa.Check(lambda s: s <= end_date, name="signup_date<=end_date"),
                                                          pa.Check(lambda s: s >= MIN_DATE, name="signup_date>=2000")]),
    }, strict=False)
    soft = pa.DataFrameSchema({
        "country": pa.Column(str, nullable=False),
        "age_group": pa.Column(str, [pa.Check(lambda s: s.isin(AGE_GROUPS), name="age_group_valid")], nullable=False),
    }, strict=False)
    lg = pd.concat([
        run_schema(ut, hard, "users", "reject"),
        typed_check(users_raw["signup_date"], ut["signup_date"], "users", "signup_date"),
        flag(users_raw["user_id"].duplicated(keep="first"), "users", "duplicate_user_id", "user_id", users_raw["user_id"]),
        run_schema(ut, soft, "users", "warning"),
    ], ignore_index=True)
    return drop_null_dupes(lg), ut


def validate_items(items_raw, parser: GenreParser):
    hard = pa.DataFrameSchema({
        "bookId": pa.Column(int, nullable=False),
        "title": pa.Column(str, nullable=False),
        "num_ratings": pa.Column(float, [pa.Check(lambda s: s >= 0, name="num_ratings>=0")], nullable=False),
        "avg_rating": pa.Column(float, [pa.Check(lambda s: s.between(0, 5), name="avg_rating_in_[0,5]")], nullable=False),
    }, strict=False)
    soft = pa.DataFrameSchema({
        "num_pages": pa.Column(float, [pa.Check(lambda s: s.between(1, 10000), name="num_pages_in_[1,10000]")], nullable=True),
    }, strict=False)
    hist_sum = items_raw[HIST_COLS].sum(axis=1, min_count=5)
    gen_empty = items_raw["genres"].map(parser.parse).map(len) == 0
    return pd.concat([
        run_schema(items_raw, hard, "items", "reject"),
        flag(items_raw["bookId"].duplicated(keep="first"), "items", "duplicate_bookId", "bookId", items_raw["bookId"]),
        run_schema(items_raw, soft, "items", "warning"),
        flag(items_raw["num_pages"].isna(), "items", "missing_num_pages", "num_pages", items_raw["num_pages"], "warning"),
        flag(gen_empty, "items", "no_usable_genres", "genres", items_raw["genres"].fillna("<null>").str[:30], "warning"),
        flag(items_raw["language"].isna(), "items", "missing_language", "language", items_raw["language"], "warning"),
        flag(items_raw["rated_1"].isna(), "items", "missing_star_histogram", "rated_1..5", items_raw["rated_1"], "warning"),
        flag(hist_sum.notna() & ((hist_sum - items_raw["num_ratings"]).abs() > 0.02 * items_raw["num_ratings"].clip(lower=1) + 5),
             "items", "histogram_sum_mismatch", "rated_1..5", hist_sum, "warning"),
        flag(items_raw["first_publish_date"].isna() & items_raw["publish_date"].isna(), "items", "missing_publish_date", "publish_date", items_raw["publish_date"], "warning"),
    ], ignore_index=True)


def parse_interactions(inter_raw):
    xt = inter_raw.copy()
    xt["rating"] = pd.to_numeric(inter_raw["rating"], errors="coerce")
    xt["item_id"] = pd.to_numeric(inter_raw["item_id"], errors="coerce")
    xt["timestamp"] = pd.to_datetime(inter_raw["timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    return xt


def validate_interactions(inter_raw, valid_user_ids, valid_item_ids, signup_map, end_date):
    xt = parse_interactions(inter_raw)
    schema = pa.DataFrameSchema({
        "interaction_id": pa.Column(str, nullable=False),
        "user_id": pa.Column(str, [pa.Check(lambda s: s.str.match(r"^U\d{5}$"), name="user_id_format")], nullable=False),
        "item_id": pa.Column(nullable=False, checks=[pa.Check(lambda s: (s > 0) & (s % 1 == 0), name="item_id_positive_integer")]),
        "rating": pa.Column(float, [pa.Check(lambda s: s.isin([1, 2, 3, 4, 5]), name="rating_in_{1..5}")], nullable=False),
        "timestamp": pa.Column(nullable=False, checks=[pa.Check(lambda s: s <= end_date, name="timestamp<=end_date"),
                                                        pa.Check(lambda s: s >= MIN_DATE, name="timestamp>=2000")]),
    }, strict=False)
    uid_ok = inter_raw["user_id"].notna()
    signup_for_row = inter_raw["user_id"].map(signup_map)
    lg = pd.concat([
        run_schema(xt, schema, "interactions", "reject"),
        typed_check(inter_raw["rating"], xt["rating"], "interactions", "rating"),
        typed_check(inter_raw["item_id"], xt["item_id"], "interactions", "item_id"),
        typed_check(inter_raw["timestamp"], xt["timestamp"], "interactions", "timestamp"),
        flag(inter_raw.duplicated(keep="first"), "interactions", "duplicate_row", "interaction_id", inter_raw["interaction_id"]),
        # referential integrity against the *validated* parent tables
        flag(uid_ok & ~inter_raw["user_id"].isin(valid_user_ids), "interactions", "ref_integrity_user", "user_id", inter_raw["user_id"]),
        flag(xt["item_id"].notna() & ~xt["item_id"].isin(valid_item_ids), "interactions", "ref_integrity_item", "item_id", inter_raw["item_id"]),
        # temporal consistency: cannot interact before signing up
        flag(xt["timestamp"].notna() & signup_for_row.notna() & (xt["timestamp"] < signup_for_row), "interactions", "timestamp>=signup_date", "timestamp", inter_raw["timestamp"]),
    ], ignore_index=True)
    return drop_null_dupes(lg)


# ----------------------------------------------------------------------------- detection check + HTML report
def detection_check(error_log, ledger):
    """Compare what the simulator injected with what the validators found."""
    def n(table, rule=None, column=None):
        m = (error_log.table == table)
        if rule: m &= error_log.rule == rule
        if column: m &= error_log.column == column
        return int(error_log[m]["row_index"].nunique())
    detected = {
        "interactions: rating out of range": n("interactions", "rating_in_{1..5}"),
        "interactions: rating wrong dtype": n("interactions", "dtype_mismatch", "rating"),
        "interactions: rating missing": n("interactions", "not_nullable", "rating"),
        "interactions: user_id missing": n("interactions", "not_nullable", "user_id"),
        "interactions: timestamp invalid": n("interactions", "dtype_mismatch", "timestamp"),
        "interactions: timestamp in future": n("interactions", "timestamp<=end_date"),
        "interactions: orphan user_id": n("interactions", "ref_integrity_user"),
        "interactions: orphan item_id": n("interactions", "ref_integrity_item"),
        "interactions: timestamp before signup": n("interactions", "timestamp>=signup_date"),
        "interactions: exact duplicate rows": n("interactions", "duplicate_row"),
        "users: country missing": n("users", "not_nullable", "country"),
        "users: invalid age_group": n("users", "age_group_valid"),
        "users: invalid/future signup_date": n("users", column="signup_date"),
        "users: duplicate user_id rows": n("users", "duplicate_user_id"),
    }
    cmp = ledger.assign(detected=pd.Series(detected))
    cmp["match"] = np.where(cmp["detected"] >= cmp["injected"] - 1, "OK", "CHECK")
    return cmp


def write_html_report(P, summary, error_log, cmp, generated_at):
    css = ("body{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#222}h1{color:#1f3b73}h2{border-bottom:2px solid #1f3b73;padding-bottom:4px;margin-top:28px}"
           "table{border-collapse:collapse;margin:8px 0;font-size:13px}th,td{border:1px solid #ccc;padding:4px 10px;text-align:left}th{background:#eef2fa}"
           ".OK{color:#1a7f37;font-weight:bold}.CHECK{color:#b42318;font-weight:bold}")
    rows = "".join(f"<tr><td>{t}</td><td>{v['rows']:,}</td><td>{v['rejected_rows']:,}</td><td>{v['rejected_rows']/v['rows']:.2%}</td><td>{v['warning_rows']:,}</td></tr>" for t, v in summary.items())
    rules = (error_log.groupby(["table", "rule", "column", "severity"]).size().rename("rows").reset_index().sort_values(["table", "severity", "rows"], ascending=[True, True, False]))
    sample = (error_log.groupby(["table", "rule"]).head(2)[["table", "rule", "column", "row_index", "value", "severity"]].sort_values(["table", "rule"]))
    det = ""
    if cmp is not None:
        det = ("<h2>3. Detection check - injected vs detected faults</h2><table><tr><th>Fault family</th><th>Injected</th><th>Detected</th><th>Status</th></tr>"
               + "".join(f"<tr><td>{i}</td><td>{r.injected}</td><td>{r.detected}</td><td class='{r.match}'>{r.match}</td></tr>" for i, r in cmp.iterrows()) + "</table>"
               "<p><i>'orphan user_id' can exceed the injected count because interactions of users rejected for an invalid signup_date also become orphans (cascade).</i></p>")
    html = (f"<html><head><meta charset='utf-8'><title>Data validation report</title><style>{css}</style></head><body>"
            f"<h1>Data validation report</h1><p>Generated {generated_at} - Pandera schemas + custom rules (duplicates, referential integrity, temporal consistency).</p>"
            f"<h2>1. Summary per table</h2><table><tr><th>Table</th><th>Rows</th><th>Rejected</th><th>Reject %</th><th>Rows with warnings</th></tr>{rows}</table>"
            f"<h2>2. Findings per rule</h2>{rules.to_html(index=False)}{det}"
            f"<h2>4. Example offending values (2 per rule)</h2>{sample.to_html(index=False)}</body></html>")
    (P.reports / "data_validation_report.html").write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------------- stage entry point
def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    end_date = pd.Timestamp(cfg["end_date"])
    items_raw = pd.read_csv(P.raw / "items_raw.csv")
    users_raw = pd.read_csv(P.raw / "users_raw.csv", dtype=str)          # read as text: nothing is silently coerced
    inter_raw = pd.read_csv(P.raw / "interactions_raw.csv", dtype=str)
    parser = GenreParser.fit(items_raw["genres"])

    # users -> items -> interactions (dependency order)
    log_users, ut = validate_users(users_raw, end_date)
    good_users = users_raw.index.difference(log_users.loc[log_users.severity == "reject", "row_index"])
    valid_user_ids = set(users_raw.loc[good_users, "user_id"])
    signup_map = ut.loc[good_users].set_index("user_id")["signup_date"]
    log_items = validate_items(items_raw, parser)
    good_items = items_raw.index.difference(log_items.loc[log_items.severity == "reject", "row_index"])
    valid_item_ids = set(items_raw.loc[good_items, "bookId"].astype(int))
    log_inter = validate_interactions(inter_raw, valid_user_ids, valid_item_ids, signup_map, end_date)
    rej_inter = log_inter.loc[log_inter.severity == "reject", "row_index"].unique()

    # error dataset + machine-readable log
    error_log = pd.concat([log_users, log_items, log_inter], ignore_index=True)
    error_log.to_csv(P.errors / "validation_error_log.csv", index=False)
    for name, raw, lg in [("users", users_raw, log_users), ("items", items_raw, log_items), ("interactions", inter_raw, log_inter)]:
        rej = lg[lg.severity == "reject"].groupby("row_index")["rule"].agg(lambda s: "; ".join(sorted(set(s))))
        bad = raw.loc[rej.index].copy(); bad["error_reasons"] = rej
        bad = bad.astype({c: "string" for c in bad.columns if bad[c].dtype == object})
        bad.to_parquet(P.errors / f"{name}_rejected.parquet", index=False)
        bad.to_csv(P.errors / f"{name}_rejected.csv", index=False)
        log.info(f"{name:13s} rows={len(raw):>7,} | rejected={len(bad):>6,} | warning rows={lg.query('severity==\"warning\"')['row_index'].nunique():>5,}")

    summary = {t: {"rows": int(len(r)), "rejected_rows": int(l.loc[l.severity == "reject", "row_index"].nunique()),
                   "warning_rows": int(l.loc[l.severity == "warning", "row_index"].nunique()),
                   "rules": {k: int(v) for k, v in l.groupby("rule").size().to_dict().items()}}
               for t, r, l in [("users", users_raw, log_users), ("items", items_raw, log_items), ("interactions", inter_raw, log_inter)]}
    (P.reports / "validation_report.json").write_text(json.dumps(summary, indent=2))

    ledger_path = P.raw / "injected_fault_ledger.csv"
    cmp = detection_check(error_log, pd.read_csv(ledger_path, index_col=0)) if ledger_path.exists() else None
    if cmp is not None:
        cmp.to_csv(P.reports / "injection_detection_check.csv")
    write_html_report(P, summary, error_log, cmp, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    ratio = len(rej_inter) / len(inter_raw)
    log.info(f"interactions rejected: {len(rej_inter):,} ({ratio:.2%}) | report -> reports/data_validation_report.html")
    return {"interaction_reject_ratio": ratio, "all_injected_faults_detected": bool((cmp["match"] == "OK").all()) if cmp is not None else None}


def rejected_ratio(base) -> float:
    """Used by the orchestrator's quality gate."""
    rep = json.loads((Paths(base).reports / "validation_report.json").read_text())["interactions"]
    return rep["rejected_rows"] / rep["rows"]
