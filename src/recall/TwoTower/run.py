import yaml
import pandas as pd
from pathlib import Path
from src.data.load_data import get_phase_click, get_whole_phase_click, obtain_topk_click
from src.recall.TwoTower.trainer import run_phase_youtube_dnn, YoutubeDNNConfig, save_artifact
import time
from src.data.save_data import save_recall_df_as_user_tuples_dict
from src.recall.recall import get_predict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path: str = 'src/recall/TwoTower/config.yml') -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    confg = load_config()
    print(f"Loaded config: {confg}")

    default_cfg = YoutubeDNNConfig()
    cfg = YoutubeDNNConfig(
        max_seq_len=confg.get('max_seq_len', default_cfg.max_seq_len),
        last_k=confg.get('last_k', default_cfg.last_k),
        min_seq_len=confg.get('min_seq_len', default_cfg.min_seq_len),
        batch_size=confg.get('batch_size', default_cfg.batch_size),
        epochs=confg.get('epochs', default_cfg.epochs),
        lr=confg.get('lr', default_cfg.lr),
        weight_decay=confg.get('weight_decay', default_cfg.weight_decay),
        num_workers=confg.get('num_workers', default_cfg.num_workers),
        embedding_dim=confg.get('embedding_dim', default_cfg.embedding_dim),
        user_hidden_dims=tuple(confg.get('user_hidden_dims', default_cfg.user_hidden_dims)),
        item_hidden_dims=tuple(confg.get('item_hidden_dims', default_cfg.item_hidden_dims)),
        output_dim=confg.get('output_dim', default_cfg.output_dim),
        dropout=confg.get('dropout', default_cfg.dropout),
        temperature=confg.get('temperature', default_cfg.temperature),
        user_batch_size=confg.get('user_batch_size', default_cfg.user_batch_size),
        item_batch_size=confg.get('item_batch_size', default_cfg.item_batch_size),
        topk=confg.get('topk', default_cfg.topk),
        exclude_history=confg.get('exclude_history', default_cfg.exclude_history),
        seed=confg.get('seed', default_cfg.seed),
        use_amp=confg.get('use_amp', default_cfg.use_amp),
        device=confg.get('device', default_cfg.device) or default_cfg.device,
        log_every=confg.get('log_every', default_cfg.log_every),
    )

    total_racall_df = pd.DataFrame()
    phase_full_sim_dict = {}
    model_dir = Path(confg['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    for phase in range(confg['start_phase'], confg['now_phase'] + 1):
        print(f"begin phase {phase} TwoTower recall...")
        all_click, click_q_time = get_phase_click(phase=phase)
        phase_whole_click = get_whole_phase_click(all_click=all_click, click_q_time=click_q_time)
        if confg['mode'] == 'offline':
            phase_click = all_click
        else:
            phase_click = phase_whole_click
        target_user_ids = click_q_time['user_id'].unique()

        phase_recall_df, artifact, info = run_phase_youtube_dnn(
            phase_click=phase_click,
            target_user_ids=target_user_ids,
            config=cfg
        )
        phase_recall_df['phase'] = phase
        total_racall_df = pd.concat([total_racall_df, phase_recall_df], axis=0, ignore_index=True)
        phase_full_sim_dict[phase] = {'youtube_dnn_train_info': info}

        model_path = model_dir / f'phase_{phase}.pt'
        save_artifact(artifact, model_path)
        print(f"phase {phase} model saved -> {model_path}")
        
        print(
            f'TwoTower phase={phase} done, users={len(target_user_ids)}, '
            f'rows={len(phase_recall_df)}, items={info.get("num_items", 0)}, '
            f'samples={info.get("num_train_samples", 0)}'
        )
        
    today = time.strftime("%Y%m%d")
    prefix = f'two_tower-{today}'
    save_recall_df_as_user_tuples_dict(total_racall_df, phase_full_sim_dict, prefix=prefix)

    _, top50_click = obtain_topk_click()
    submit_df = get_predict(total_racall_df, 'sim', top50_click)
    submit_path = Path('submit_two_tower.csv')
    submit_df.to_csv(submit_path, index=False, header=None)
    print(f"submit saved -> {submit_path}")

if __name__ == '__main__':
    main()
