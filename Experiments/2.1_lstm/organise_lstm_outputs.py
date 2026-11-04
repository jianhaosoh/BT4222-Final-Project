"""One-time organisation of existing LSTM artifacts; preview by default.

Run where the shared Drive folder is mounted (for example, Google Colab):

    python organise_lstm_outputs.py "/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder"

Add --apply after reviewing the preview. No training, model loading, file-content
rewriting or copying is performed. Notebook files and their saved outputs are
untouched. Existing JSON/log fields retain their historical recorded paths.
"""
import argparse
from pathlib import Path
import re


def plan_moves(data_root):
    root = Path(data_root) / 'Experiments' / '2.1_lstm'
    if not root.is_dir():
        raise FileNotFoundError(f'LSTM output folder not found: {root}')
    moves = []
    for source in sorted(root.iterdir()):
        if not source.is_dir():
            continue
        if re.fullmatch(r'\d{8}_\d{6}_seed\d+', source.name):
            destination = root / '2.1.1_initial_run' / source.name
        elif re.fullmatch(r'\d{8}_\d{6}_improved_seed\d+', source.name):
            destination = root / '2.1.3_improved_run' / source.name
        elif source.name == 'ablation':
            destination = root / '2.1.2_ablation_study'
        else:
            continue
        if destination.exists():
            raise FileExistsError(f'Will not overwrite an existing destination: {destination}')
        moves.append((source, destination))
    return moves


def organise(data_root, apply=False):
    moves = plan_moves(data_root)
    for source, destination in moves:
        print(f'{"MOVE" if apply else "PREVIEW"}: {source}\n      -> {destination}')
    if apply:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Both locations are inside the same Drive mount. Do not fall back
            # to copying large model files if a directory rename is unsupported.
            source.rename(destination)
    if not moves:
        print('No legacy LSTM folders need moving.')
    elif not apply:
        print('\nPreview only. Add --apply to perform these moves.')
    return moves


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--apply', action='store_true', help='Move the previewed folders.')
    args = parser.parse_args()
    organise(args.data_root, apply=args.apply)
