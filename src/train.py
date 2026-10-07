import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu cua bo du lieu Adult va do lech toi da cho phep
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: canh bao lech lac du lieu neu ty le lop duong lech qua 5 diem phan tram
    positive_rate = float(y_train.mean())
    if abs(positive_rate - REFERENCE_POSITIVE_RATE) > DRIFT_TOLERANCE:
        print(
            f"CANH BAO DATA DRIFT: ty le lop duong {positive_rate:.1%} lech qua "
            f"{DRIFT_TOLERANCE:.0%} so voi tham chieu {REFERENCE_POSITIVE_RATE:.1%}"
        )

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so
        # f1_score tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_rate", positive_rate)

        # Bonus 2: quet nguong quyet dinh tu 0.1 den 0.9 (buoc 0.05) tren xac suat lop duong
        probs = model.predict_proba(X_eval)[:, 1]
        thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)
        f1_by_threshold = [f1_score(y_eval, (probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_by_threshold))
        best_threshold = float(thresholds[best_idx])
        best_threshold_f1 = float(f1_by_threshold[best_idx])
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | Positive rate: {positive_rate:.4f}")
        print(f"Nguong tot nhat: {best_threshold:.2f} -> F1 {best_threshold_f1:.4f} (nguong 0.5: F1 {f1:.4f})")

        # Luu metrics ra file outputs/report.json (doc boi GitHub Actions o Buoc 2)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "positive_rate": positive_rate,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_threshold_f1,
                },
                f,
            )

        # Luu mo hinh ra file models/model.joblib (upload len cloud storage o Buoc 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
