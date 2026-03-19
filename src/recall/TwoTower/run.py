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

    cfg = YoutubeDNNConfig(
        max_seq_len=confg['max_seq_len'],
        last_k=confg['last_k'],
        batch_size=confg['batch_size'],
        epochs=confg['epochs'],
        lr=confg['lr'],
        weight_decay=confg['weight_decay'],
        embedding_dim=confg['embedding_dim'],
        output_dim=confg['output_dim'],
        dropout=confg['dropout'],
        temperature=confg['temperature'],
        topk=confg['topk'],
        seed=confg['seed'],
        device=confg['device'] if confg['device'] else YoutubeDNNConfig().device,
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
            f'rows={len(phase_recall_df)}, items={info.get("num_items", 0)}'
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