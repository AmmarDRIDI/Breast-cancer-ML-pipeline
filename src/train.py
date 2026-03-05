import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(seed: int):
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test, dataset.feature_names.tolist()


def build_pipeline(max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=max_iter,
                    C=1.0,
                ),
            ),
        ]
    )


def save_error_analysis(x_val: pd.DataFrame, y_val, y_pred, output_dir: str) -> None:
    mis_idx = np.where(y_pred != y_val)[0]
    samples = []
    for idx in mis_idx[:5]:
        row = x_val.iloc[idx].to_dict()
        compact_row = {k: float(v) for k, v in list(row.items())[:6]}
        samples.append(
            {
                "index": int(idx),
                "true": int(y_val.iloc[idx]),
                "pred": int(y_pred[idx]),
                "feature_head": compact_row,
            }
        )

    cm = confusion_matrix(y_val, y_pred).tolist()
    payload = {
        "confusion_matrix": cm,
        "misclassified_examples": samples,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args):
    set_seed(args.seed)
    x_train, x_val, x_test, y_train, y_val, y_test, feature_names = load_data(
        args.seed)

    model = build_pipeline(args.max_iter)
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    val_proba = model.predict_proba(x_val)[:, 1]
    test_pred = model.predict(x_test)
    test_proba = model.predict_proba(x_test)[:, 1]

    val_f1 = f1_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_proba)
    test_f1 = f1_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"artifacts/model_{run_id}.joblib"
    latest_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)
    joblib.dump(model, latest_path)

    save_error_analysis(x_val.reset_index(drop=True),
                        y_val.reset_index(drop=True), val_pred, "artifacts")

    report = classification_report(
        y_test, test_pred, target_names=["malignant", "benign"])
    with open("logs/test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    final_log_line = (
        f"run_id={run_id} seed={args.seed} "
        f"val_f1={val_f1:.4f} val_auc={val_auc:.4f} "
        f"test_f1={test_f1:.4f} test_auc={test_auc:.4f} "
        f"checkpoint={model_path}"
    )

    with open("logs/final_val.log", "w", encoding="utf-8") as f:
        f.write(final_log_line + "\n")

    metadata = {
        "run_id": run_id,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "features": feature_names,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "checkpoint": model_path,
    }

    with open("artifacts/run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(final_log_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train breast cancer classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    main(parser.parse_args())
