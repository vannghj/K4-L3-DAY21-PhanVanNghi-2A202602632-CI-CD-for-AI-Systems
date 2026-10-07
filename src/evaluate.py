import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


def evaluate(
    model_path: str = "models/model.joblib",
    eval_path: str = "data/holdout.csv",
    out_path: str = "outputs/detail.txt",
) -> str:
    """
    Bonus 3: tao bao cao chi tiet cho mo hinh da huan luyen tren tap holdout.

    Ghi confusion matrix (dang van ban) va precision / recall cua tung lop
    ra out_path, dong thoi tra ve noi dung bao cao.
    """
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    preds = model.predict(X_eval)

    tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()

    lines = [
        f"Bao cao chi tiet tren {len(y_eval)} mau holdout (nguong 0.5)",
        "",
        "Confusion matrix (hang = thuc te, cot = du doan):",
        f"{'':>16}{'pred=0':>10}{'pred=1':>10}",
        f"{'thuc te=0':>16}{tn:>10}{fp:>10}",
        f"{'thuc te=1':>16}{fn:>10}{tp:>10}",
        "",
        f"{'Lop':<22}{'precision':>10}{'recall':>10}",
    ]
    for label, name in LABELS.items():
        p = precision_score(y_eval, preds, pos_label=label, zero_division=0)
        r = recall_score(y_eval, preds, pos_label=label, zero_division=0)
        lines.append(f"{f'{label} ({name})':<22}{p:>10.4f}{r:>10.4f}")
    lines += [
        "",
        f"Bo sot nguoi thu nhap cao (FN): {fn}",
        f"Gan nham nguoi thu nhap thap la cao (FP): {fp}",
    ]
    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    print(evaluate(*sys.argv[1:]))
