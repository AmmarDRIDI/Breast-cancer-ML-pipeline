

## Project structure

- `src/train.py`: main ML pipeline (split, preprocessing, train, eval, logging)
- `src/serve.py`: FastAPI inference route (`POST /predict`)
- `tests/test_data_split.py`: sanity test for data split and feature consistency
- `logs/final_val.log`: final validation line (created after training)
- `artifacts/error_analysis.json`: confusion matrix + misclassified examples
- `notebooks/error_analysis.ipynb`: notebook summary for application evidence

## Setup

### pip

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate upsaclay-m1-ml
```

## Train

```bash
python src/train.py --seed 42 --max-iter 300
```

## Test

```bash
pytest -q
```

## Run API (optional)

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

## Reproducibility notes

Seeds are fixed for Python, NumPy and split random state (`seed=42`). Remaining nondeterminism can still arise from BLAS threading / floating-point ordering on different CPUs.
