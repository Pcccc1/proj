from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import mode, now_phase, start_phase, user_data_dir
from src.data.load_data import get_phase_click, get_whole_phase_click, obtain_topk_click
from src.data.save_data import save_recall_df_as_user_tuples_dict
from src.recall.TwoTower.trainer import YouTubeDNNConfig, run_phase_youtube_dnn, save_artifact
from src.recall.recall import get_predict


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run YouTubeDNN two-tower recall by phase.")
    parser.add_argument("--start-phase", type=int, default=start_phase)
    parser.add_argument("--end-phase", type=int, default=now_phase)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--last-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda/cuda:0/cpu")
    parser.add_argument("--save-model", action="store_true", help="save per-phase model checkpoints")
    parser.add_argument("--prefix", type=str, default=None, help="file prefix for recall outputs")
    parser.add_argument("--submit-path", type=str, default="submit.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = YouTubeDNNConfig(
        max_seq_len=args.max_seq_len,
        last_k=args.last_k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        topk=args.topk,
        seed=args.seed,
        device=args.device if args.device else YouTubeDNNConfig().device,
    )

    print(f"[YouTubeDNN] mode={mode} phase_range=[{args.start_phase}, {args.end_phase}] device={cfg.device}")

    total_recall_df = pd.DataFrame()
    phase_full_sim_dict: dict[int, dict] = {}

    model_dir = Path(user_data_dir) / "recall" / mode / "youtube_dnn_models"
    if args.save_model:
        model_dir.mkdir(parents=True, exist_ok=True)

    for phase in range(args.start_phase, args.end_phase + 1):
        print(f"[YouTubeDNN] begin phase={phase}")
        all_click, click_q_time = get_phase_click(phase=phase)
        phase_whole_click = get_whole_phase_click(all_click=all_click, click_q_time=click_q_time)
        target_user_ids = click_q_time["user_id"].unique()

        phase_recall_df, artifact, info = run_phase_youtube_dnn(
            phase_whole_click=phase_whole_click,
            target_user_ids=target_user_ids,
            config=cfg,
        )
        phase_recall_df["phase"] = phase
        total_recall_df = pd.concat([total_recall_df, phase_recall_df], axis=0, ignore_index=True)
        phase_full_sim_dict[phase] = {"youtube_dnn_train_info": info}

        if args.save_model:
            model_path = model_dir / f"phase_{phase}.pt"
            save_artifact(artifact, model_path)
            print(f"[YouTubeDNN] phase={phase} model saved -> {model_path}")

        print(
            f"[YouTubeDNN] phase={phase} done, users={len(target_user_ids)}, "
            f"rows={len(phase_recall_df)}, items={info.get('num_items', 0)}, "
            f"samples={info.get('num_train_samples', 0)}"
        )

    today = time.strftime("%Y%m%d")
    prefix = args.prefix if args.prefix else f"youtube_dnn-{today}"
    save_recall_df_as_user_tuples_dict(total_recall_df, phase_full_sim_dict, prefix=prefix)
    print(f"[YouTubeDNN] recall outputs saved with prefix={prefix}")

    _, top50_click = obtain_topk_click()
    submit_df = get_predict(total_recall_df, "sim", top50_click)
    submit_path = Path(args.submit_path)
    submit_df.to_csv(submit_path, index=False, header=None)
    print(f"[YouTubeDNN] submit file saved -> {submit_path.resolve()}")


if __name__ == "__main__":
    main()
