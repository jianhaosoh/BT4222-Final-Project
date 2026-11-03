"""Export feature coefficients from a trusted, saved baseline model.

Run from the repository root:
    python "Baseline Testing/export_coefficients.py"
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd


def export_coefficients(model_path, output_path):
    """Save the processed feature names and their learned coefficients."""
    # Load only trusted model files: joblib can execute code when loading.
    saved = joblib.load(model_path)
    pipeline = saved['pipeline']
    coefficients = pd.DataFrame({
        'feature': pipeline['features'].get_feature_names_out(),
        'coefficient': pipeline['model'].coef_[0],
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_path, index=False)
    return coefficients


def main():
    results = Path(__file__).resolve().parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=results / 'model.joblib')
    parser.add_argument('--output', type=Path,
                        default=results / 'logistic_regression_coefficients.csv')
    args = parser.parse_args()
    coefficients = export_coefficients(args.model, args.output)
    print(f'Saved {len(coefficients)} coefficients to {args.output.resolve()}')


if __name__ == '__main__':
    main()
