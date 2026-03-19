from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    infer_answer_file_prefix,
    mode,
    offline_answer_path,
    user_data_dir,
    data_dir
)

ANSWER_COLS = ["user_id", "item_id", "time"]
CLICK_COLS = ["user_id", "item_id", "time"]
DEFAULT_TOPKS = (5, 10, 20, 50)


def _read_answer_df(phase: int) -> pd.DataFrame:
    answer_file = Path(offline_answer_path) / f"{infer_answer_file_prefix}-{phase}.csv"
    if not answer_file.exists():
        raise FileNotFoundError(f"answer file not found: {answer_file}")

    answer_df = pd.read_csv(
        answer_file,
        header=None,
        names=ANSWER_COLS,
        dtype={"user_id": "int32", "item_id": "int32"},
    )
    answer_df = answer_df.drop_duplicates(subset=["user_id"], keep="first")
    return answer_df[["user_id", "item_id"]]


def _pick_score_col(recall_df: pd.DataFrame) -> str:
    for col in ("sim", "score"):
        if col in recall_df.columns:
            return col
    raise ValueError("recall dataframe must include one score column: `sim` or `score`")


def _to_rank_df(phase_recall_df: pd.DataFrame, score_col: str, max_k: int) -> pd.DataFrame:
    rank_df = (
        phase_recall_df.groupby(["user_id", "item_id"], as_index=False)[score_col]
        .max()
        .sort_values(["user_id", score_col, "item_id"], ascending=[True, False, True], kind="mergesort")
    )
    rank_df["rank"] = rank_df.groupby("user_id").cumcount() + 1
    return rank_df[rank_df["rank"] <= max_k][["user_id", "item_id", "rank"]]


def _metric_from_rank(eval_df: pd.DataFrame, k: int) -> dict:
    """
    eval_df 必须至少包含:
    - user_id
    - item_id
    - rank (命中时为排名，未命中时为 NaN)
    """
    rank = eval_df["rank"].to_numpy(dtype=float)
    hit_mask = np.isfinite(rank) & (rank <= k)

    hit_count = int(hit_mask.sum())
    user_count = int(len(eval_df))
    denom = max(user_count, 1)

    ndcg_vals = np.zeros(user_count, dtype=np.float64)
    mrr_vals = np.zeros(user_count, dtype=np.float64)

    if hit_count > 0:
        ndcg_vals[hit_mask] = 1.0 / np.log2(rank[hit_mask] + 1.0)
        mrr_vals[hit_mask] = 1.0 / rank[hit_mask]

    return {
        "k": int(k),
        "hit_count": hit_count,
        "user_count": user_count,
        "hit_rate": float(hit_count / denom),
        "ndcg": float(ndcg_vals.sum() / denom),
        "mrr": float(mrr_vals.sum() / denom),
    }


def _find_latest_total_recall_pkl(recall_dir: Path) -> Path:
    candidates = sorted(recall_dir.glob("*_total_recall_df.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    fallback_root = Path(user_data_dir) / "recall"
    fallback_candidates = sorted(
        fallback_root.glob("*/*_total_recall_df.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if fallback_candidates:
        return fallback_candidates[0]

    raise FileNotFoundError(f"no total recall pkl found under: {fallback_root}")


def _resolve_train_click_file(phase: int) -> Path:
    """
    尽量兼容常见目录结构。
    优先找:
    1) {user_data_dir}/underexpose_train/underexpose_train_click-{phase}.csv
    2) {user_data_dir}/underexpose_train/underexpose_train_click-{phase}.zip
    3) {user_data_dir}/underexpose_train_click-{phase}.csv
    4) {user_data_dir}/underexpose_train_click-{phase}.zip
    5) 递归搜索
    """
    base = Path(data_dir)
    filename_csv = f"underexpose_train_click-{phase}.csv"
    filename_zip = f"underexpose_train_click-{phase}.zip"

    candidates = [
        base / "offline_underexpose_train" / filename_csv,
        base / "offline_underexpose_train" / filename_zip,
        base / filename_csv,
        base / filename_zip,
    ]

    for p in candidates:
        if p.exists():
            return p

    recursive = list(base.rglob(filename_csv)) + list(base.rglob(filename_zip))
    if recursive:
        return sorted(recursive)[0]

    raise FileNotFoundError(f"cannot find train click file for phase={phase} under {base}")


def _read_train_click_df(phase: int) -> pd.DataFrame:
    file_path = _resolve_train_click_file(phase)
    click_df = pd.read_csv(
        file_path,
        header=None,
        names=CLICK_COLS,
        dtype={"user_id": "int32", "item_id": "int32"},
        compression="infer",
    )
    return click_df[["item_id"]]


def _build_cumulative_item_count_map(max_phase: int) -> dict[int, pd.Series]:
    """
    返回:
    phase -> cumulative item exposure count Series(index=item_id, value=count)
    统计 underexpose_train_click-0 ... underexpose_train_click-phase
    """
    cumulative_map: dict[int, pd.Series] = {}
    cumulative_count = pd.Series(dtype="int64")

    for phase in range(max_phase + 1):
        click_df = _read_train_click_df(phase)
        one_phase_count = click_df["item_id"].value_counts().astype("int64")

        if cumulative_count.empty:
            cumulative_count = one_phase_count
        else:
            cumulative_count = cumulative_count.add(one_phase_count, fill_value=0).astype("int64")

        cumulative_map[phase] = cumulative_count.copy()

    return cumulative_map


def _build_rare_subset(answer_df: pd.DataFrame, item_count_series: pd.Series) -> pd.DataFrame:
    """
    rare 定义:
    对当前 phase 的 answer 样本，根据真实 item 在历史训练中的曝光次数升序排序，
    取前一半作为 rare 子集。
    若出现并列，按 user_id / item_id 稳定打破。
    """
    rare_df = answer_df.copy()
    rare_df["item_freq"] = rare_df["item_id"].map(item_count_series).fillna(0).astype("int64")

    rare_df = rare_df.sort_values(
        ["item_freq", "user_id", "item_id"],
        ascending=[True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    rare_n = len(rare_df) // 2
    return rare_df.iloc[:rare_n][["user_id", "item_id"]].copy()


def evaluate_recall(
    phase: int,
    phase_recall_df: pd.DataFrame,
    item_count_series: pd.Series,
    topks: Iterable[int] = DEFAULT_TOPKS,
) -> dict:
    topks = tuple(sorted(set(int(k) for k in topks)))
    max_k = topks[-1]

    score_col = _pick_score_col(phase_recall_df)
    answer_df = _read_answer_df(phase)
    rank_df = _to_rank_df(phase_recall_df, score_col=score_col, max_k=max_k)

    # full
    eval_full_df = answer_df.merge(rank_df, on=["user_id", "item_id"], how="left")

    # rare
    rare_answer_df = _build_rare_subset(answer_df, item_count_series=item_count_series)
    eval_rare_df = rare_answer_df.merge(rank_df, on=["user_id", "item_id"], how="left")

    # diagnostics
    user_recall_count = (
        phase_recall_df.groupby("user_id")["item_id"]
        .nunique()
        .rename("candidate_cnt")
        .reindex(answer_df["user_id"])
        .fillna(0)
    )
    user_recall_coverage = float((user_recall_count > 0).mean())

    full_metrics = [_metric_from_rank(eval_full_df, k) for k in topks]
    rare_metrics = [_metric_from_rank(eval_rare_df, k) for k in topks]

    phase_summary = {
        "phase": int(phase),

        "answer_user_cnt": int(answer_df["user_id"].nunique()),
        "rare_user_cnt": int(rare_answer_df["user_id"].nunique()),
        "recall_user_cnt": int(phase_recall_df["user_id"].nunique()),

        "candidate_avg": float(user_recall_count.mean()),
        "candidate_median": float(user_recall_count.median()),
        "user_recall_coverage": user_recall_coverage,

        "hit_rate_full_at_max_k": float(full_metrics[-1]["hit_rate"]),
        "ndcg_full_at_max_k": float(full_metrics[-1]["ndcg"]),
        "mrr_full_at_max_k": float(full_metrics[-1]["mrr"]),

        "hit_rate_rare_at_max_k": float(rare_metrics[-1]["hit_rate"]),
        "ndcg_rare_at_max_k": float(rare_metrics[-1]["ndcg"]),
        "mrr_rare_at_max_k": float(rare_metrics[-1]["mrr"]),
    }

    return {
        "phase_summary": phase_summary,
        "topk_metrics_full": full_metrics,
        "topk_metrics_rare": rare_metrics,
        "eval_full_df": eval_full_df[["user_id", "item_id", "rank"]].copy(),
        "eval_rare_df": eval_rare_df[["user_id", "item_id", "rank"]].copy(),
    }


def evaluate_by_phase(
    recall_pkl_path: str | None = None,
    phases: Iterable[int] | None = None,
    topks: Iterable[int] = DEFAULT_TOPKS,
    save: bool = True,
) -> dict:
    recall_dir = Path(user_data_dir) / "recall" / mode
    recall_path = Path(recall_pkl_path) if recall_pkl_path else _find_latest_total_recall_pkl(recall_dir)

    recall_df = pickle.load(open(recall_path, "rb"))
    if not isinstance(recall_df, pd.DataFrame):
        raise TypeError(f"recall pkl must be pandas DataFrame, got: {type(recall_df)}")
    if "phase" not in recall_df.columns:
        raise ValueError("recall dataframe must include `phase` column")

    valid_phases = sorted(recall_df["phase"].dropna().astype("int32").unique().tolist())
    if phases is None:
        target_phases = valid_phases
    else:
        target_phases = sorted(set(int(p) for p in phases))

    if not target_phases:
        raise ValueError("no target phases")

    max_phase = max(target_phases)
    cumulative_item_count_map = _build_cumulative_item_count_map(max_phase=max_phase)

    phase_rows = []
    topk_rows_full = []
    topk_rows_rare = []
    all_eval_full_parts = []
    all_eval_rare_parts = []

    for phase in target_phases:
        phase_recall_df = recall_df[recall_df["phase"] == phase]
        if phase_recall_df.empty:
            phase_rows.append(
                {
                    "phase": int(phase),
                    "answer_user_cnt": 0,
                    "rare_user_cnt": 0,
                    "recall_user_cnt": 0,
                    "candidate_avg": 0.0,
                    "candidate_median": 0.0,
                    "user_recall_coverage": 0.0,
                    "hit_rate_full_at_max_k": 0.0,
                    "ndcg_full_at_max_k": 0.0,
                    "mrr_full_at_max_k": 0.0,
                    "hit_rate_rare_at_max_k": 0.0,
                    "ndcg_rare_at_max_k": 0.0,
                    "mrr_rare_at_max_k": 0.0,
                    "status": "empty_recall",
                }
            )
            continue

        one_phase_result = evaluate_recall(
            phase=phase,
            phase_recall_df=phase_recall_df,
            item_count_series=cumulative_item_count_map[phase],
            topks=topks,
        )

        phase_rows.append({**one_phase_result["phase_summary"], "status": "ok"})

        for row in one_phase_result["topk_metrics_full"]:
            topk_rows_full.append({"phase": int(phase), "subset": "full", **row})

        for row in one_phase_result["topk_metrics_rare"]:
            topk_rows_rare.append({"phase": int(phase), "subset": "rare", **row})

        eval_full_part = one_phase_result["eval_full_df"].copy()
        eval_full_part["phase"] = int(phase)
        all_eval_full_parts.append(eval_full_part)

        eval_rare_part = one_phase_result["eval_rare_df"].copy()
        eval_rare_part["phase"] = int(phase)
        all_eval_rare_parts.append(eval_rare_part)

    phase_metrics_df = pd.DataFrame(phase_rows).sort_values("phase").reset_index(drop=True)

    phase_topk_metrics_df = pd.concat(
        [
            pd.DataFrame(topk_rows_full),
            pd.DataFrame(topk_rows_rare),
        ],
        ignore_index=True
    ).sort_values(["subset", "phase", "k"]).reset_index(drop=True)

    global_full_metrics = []
    global_rare_metrics = []

    if all_eval_full_parts:
        all_eval_full_df = pd.concat(all_eval_full_parts, ignore_index=True)
        for k in sorted(set(int(k) for k in topks)):
            global_full_metrics.append({"subset": "full", **_metric_from_rank(all_eval_full_df, k)})

    if all_eval_rare_parts:
        all_eval_rare_df = pd.concat(all_eval_rare_parts, ignore_index=True)
        for k in sorted(set(int(k) for k in topks)):
            global_rare_metrics.append({"subset": "rare", **_metric_from_rank(all_eval_rare_df, k)})

    global_topk_metrics_df = pd.concat(
        [pd.DataFrame(global_full_metrics), pd.DataFrame(global_rare_metrics)],
        ignore_index=True
    ).sort_values(["subset", "k"]).reset_index(drop=True)

    result = {
        "recall_pkl": str(recall_path),
        "mode": mode,
        "phases": target_phases,
        "topks": list(sorted(set(int(k) for k in topks))),
        "phase_metrics": phase_metrics_df,
        "phase_topk_metrics": phase_topk_metrics_df,
        "global_topk_metrics": global_topk_metrics_df,
    }

    if save:
        save_dir = Path(user_data_dir) / "evaluation" / mode
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        recall_prefix = recall_path.stem.replace("_total_recall_df", "")

        phase_file = save_dir / f"{recall_prefix}_phase_metrics_{ts}.csv"
        phase_topk_file = save_dir / f"{recall_prefix}_phase_topk_metrics_{ts}.csv"
        global_file = save_dir / f"{recall_prefix}_global_topk_metrics_{ts}.csv"
        summary_file = save_dir / f"{recall_prefix}_summary_{ts}.json"

        phase_metrics_df.to_csv(phase_file, index=False)
        phase_topk_metrics_df.to_csv(phase_topk_file, index=False)
        global_topk_metrics_df.to_csv(global_file, index=False)

        summary_obj = {
            "generated_at": ts,
            "recall_pkl": str(recall_path),
            "mode": mode,
            "phases": target_phases,
            "topks": result["topks"],
            "phase_metrics_file": str(phase_file),
            "phase_topk_metrics_file": str(phase_topk_file),
            "global_topk_metrics_file": str(global_file),
            "note": "rare subset is built by selecting the half of answer cases whose ground-truth items have the lowest historical exposure in train_click-0..phase",
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, ensure_ascii=False, indent=2)

        result["summary_file"] = str(summary_file)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recall result by phase with full/rare official-style metrics.")
    parser.add_argument("--recall-pkl", type=str, default=None, help="path to *_total_recall_df.pkl")
    parser.add_argument("--phases", type=str, default=None, help="comma separated phase list, e.g. 0,1,2")
    parser.add_argument("--topks", type=str, default="5,10,20,50", help="comma separated K list")
    parser.add_argument("--no-save", action="store_true", help="do not save csv/json files")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    selected_phases = None if args.phases is None else [int(x.strip()) for x in args.phases.split(",") if x.strip()]
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    result = evaluate_by_phase(
        recall_pkl_path=args.recall_pkl,
        phases=selected_phases,
        topks=topks,
        save=not args.no_save,
    )

    print("recall_pkl:", result["recall_pkl"])
    print("mode:", result["mode"])
    print("phases:", result["phases"])

    if not result["phase_metrics"].empty:
        print("phase metrics:")
        print(result["phase_metrics"].to_string(index=False))

    if not result["global_topk_metrics"].empty:
        print("global metrics:")
        print(result["global_topk_metrics"].to_string(index=False))

    if "summary_file" in result:
        print("saved summary:", result["summary_file"])