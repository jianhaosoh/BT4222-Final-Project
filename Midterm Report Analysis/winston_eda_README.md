# Winston's cold-start EDA

- `winston_ebnerd_eda.ipynb`: Executed cold-start article EDA, including data-quality checks, cold-start windows, feature distributions, transformation comparisons, multicollinearity, and model diagnostics.
- `winston_eda_utils.py`: Supporting analysis and plotting functions.

## Running the cold-start EDA

Keep `winston_eda_utils.py` beside the notebook in `Midterm Report Analysis`.

Install the analysis dependencies in your notebook's Python environment:

```sh
pip install duckdb matplotlib numpy pandas pyarrow scikit-learn seaborn jupyter
```

Place the required dataset files under the repository root:

```text
data/
  ebnerd_large/
    articles.parquet
    train/behaviors.parquet
    validation/behaviors.parquet
  ebnerd_demo/
    document_vector.parquet
```

Start Jupyter from the repository root and open `Midterm Report Analysis/winston_ebnerd_eda.ipynb`. The notebook searches its working directory and parents for `data/ebnerd_large` and locates the helper in `Midterm Report Analysis`. The source datasets are not included in this repository; keep them local.

Saved charts and tables can be viewed without rerunning the analysis. A full run scans the LARGE dataset, can take several minutes, and configures DuckDB with a 10 GB memory limit and a temporary spill directory.
