"""加载三个已训练模型，用 matplotlib 复现论文式六联图（特征重要性 + ROC/parity）。

不绘制混淆矩阵。特征重要性取自模型训练结果（按原始输入特征聚合后归一化）。
PyCharm 直接运行；无窗口：python generate_model_performance_figure.py --no-show
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from generate_dhoip_abc import MODEL_FILES, atomic_bytes


BAR_COLOR = "#A6616D"
ROC_COLOR = "#C0392B"
SCATTER_COLOR = "#E67E22"
DIAG_COLOR = "#2E86C1"


FEATURE_LATEX = {
    "r_A": r"$IR_{A}$",
    "r_B": r"$IR_{B}$",
    "r_B′": r"$IR_{B^{\prime}}$",
    "r_X1": r"$IR_{X1}$",
    "χ_B": r"$\chi_{B}$",
    "χ_B′": r"$\chi_{B^{\prime}}$",
    "χ_X1": r"$\chi_{X1}$",
    "r_A + r_B": r"$IR_{A+B}$",
    "r_A − r_B": r"$IR_{A-B}$",
    "r_A / r_B": r"$IR_{A/B}$",
    "r_A + r_B′": r"$IR_{A+B^{\prime}}$",
    "r_A − r_B′": r"$IR_{A-B^{\prime}}$",
    "r_A / r_B′": r"$IR_{A/B^{\prime}}$",
    "r_A + r_X1": r"$IR_{A+X1}$",
    "r_A − r_X1": r"$IR_{A-X1}$",
    "r_A / r_X1": r"$IR_{A/X1}$",
    "r_B + r_B′": r"$IR_{B+B^{\prime}}$",
    "r_B − r_B′": r"$IR_{B-B^{\prime}}$",
    "r_B / r_B′": r"$IR_{B/B^{\prime}}$",
    "r_B + r_X1": r"$IR_{B+X1}$",
    "r_B − r_X1": r"$IR_{B-X1}$",
    "r_B / r_X1": r"$IR_{B/X1}$",
    "r_B′ + r_X1": r"$IR_{B^{\prime}+X1}$",
    "r_B′ − r_X1": r"$IR_{B^{\prime}-X1}$",
    "r_B′ / r_X1": r"$IR_{B^{\prime}/X1}$",
    "χ_B + χ_B′": r"$\chi_{B+B^{\prime}}$",
    "χ_B − χ_B′": r"$\chi_{B-B^{\prime}}$",
    "χ_B / χ_B′": r"$\chi_{B/B^{\prime}}$",
    "χ_B + χ_X1": r"$\chi_{B+X1}$",
    "χ_B − χ_X1": r"$\chi_{B-X1}$",
    "χ_B / χ_X1": r"$\chi_{B/X1}$",
    "χ_B′ + χ_X1": r"$\chi_{B^{\prime}+X1}$",
    "χ_B′ − χ_X1": r"$\chi_{B^{\prime}-X1}$",
    "χ_B′ / χ_X1": r"$\chi_{B^{\prime}/X1}$",
    "B_site": r"$B$",
    "Bprime_site": r"$B^{\prime}$",
    "A_site": r"$A$",
    "X1_site": r"$X1$",
}


def latex_label(name: str) -> str:
    return FEATURE_LATEX.get(name, name.replace("_", r"\_"))


def load_models_and_data(base: Path):
    paths = [base / "models" / name for name in MODEL_FILES]
    paths += [base / "data" / name for name in ["training_data.csv", "splits.csv"]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少文件：\n" + "\n".join(missing))
    models = [joblib.load(p) for p in paths[:3]]
    known = pd.read_csv(paths[3], encoding="utf-8-sig", float_precision="round_trip")
    splits = pd.read_csv(paths[4], encoding="utf-8-sig", float_precision="round_trip")
    split_cols = [
        "System", "partition",
        "task_01_structure", "task_02_bandgap_classification", "task_03_bandgap_regression",
    ]
    frame = known.merge(splits[split_cols], on="System", how="left", validate="one_to_one")
    if frame["partition"].isna().any():
        raise ValueError("训练数据无法匹配 train/test 划分。")
    return models, frame


def original_feature_importance(pipeline) -> pd.Series:
    """将编码后的 feature_importances_ 聚合回 Pipeline 原始输入列，再转为相对百分比。"""
    preprocessor = pipeline.named_steps["preprocessor"]
    tree_model = pipeline.named_steps["model"]
    importances = np.asarray(tree_model.feature_importances_, dtype=float)
    grouped = {}
    covered = []
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        span = preprocessor.output_indices_[name]
        if name == "numeric":
            if span.stop - span.start != len(columns):
                raise ValueError("数值列与编码列数量不一致。")
            for offset, column in enumerate(columns):
                index = span.start + offset
                grouped[column] = float(importances[index])
                covered.append(index)
        elif name == "categorical":
            index = span.start
            categories_list = transformer.categories_
            drop_idx = getattr(transformer, "drop_idx_", None)
            for column, categories in zip(columns, categories_list):
                n_cats = len(categories)
                if drop_idx is not None:
                    # drop='first' 等会少一列
                    kept = n_cats - 1
                else:
                    kept = n_cats
                stop = index + kept
                grouped[column] = float(importances[index:stop].sum())
                covered.extend(range(index, stop))
                index = stop
            if index != span.stop:
                raise ValueError("类别特征映射未覆盖全部编码列。")
        else:
            raise ValueError("未处理的预处理步骤：" + name)
    if sorted(covered) != list(range(len(importances))):
        raise ValueError("特征重要性编码列覆盖错误。")
    series = pd.Series(grouped, dtype=float)
    total = float(series.sum())
    if total <= 0:
        raise ValueError("特征重要性全为零。")
    return (100.0 * series / total).sort_values(ascending=False)


def positive_proba(pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.classes_)
    index = int(np.flatnonzero(classes == 1)[0])
    return pipeline.predict_proba(frame[list(pipeline.feature_names_in_)])[:, index]


def evaluate_models(models, frame: pd.DataFrame) -> dict:
    structure, gap_class, gap_reg = models
    is_test = frame["partition"] == "test"

    # (a)(d) 可形成性分类：task_01 的 test
    struct_test = frame.loc[is_test & frame["task_01_structure"]].copy()
    y_struct = struct_test["perovskite"].astype(int).to_numpy()
    p_struct = positive_proba(structure, struct_test)
    fpr_s, tpr_s, _ = roc_curve(y_struct, p_struct)
    auc_s = float(roc_auc_score(y_struct, p_struct))

    # (b)(e) 带隙分类：task_02 的 test，标签 LowGap_PBE > 0.2
    gap_test = frame.loc[is_test & frame["task_02_bandgap_classification"]].copy()
    y_gap = (gap_test["LowGap_PBE"] > 0.2).astype(int).to_numpy()
    p_gap = positive_proba(gap_class, gap_test)
    fpr_g, tpr_g, _ = roc_curve(y_gap, p_gap)
    auc_g = float(roc_auc_score(y_gap, p_gap))

    # (c)(f) 带隙回归：task_03 的 test
    reg_test = frame.loc[is_test & frame["task_03_bandgap_regression"]].copy()
    y_true = reg_test["LowGap_PBE"].astype(float).to_numpy()
    y_pred = gap_reg.predict(reg_test[list(gap_reg.feature_names_in_)])
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))

    return {
        "structure_importance": original_feature_importance(structure),
        "gap_class_importance": original_feature_importance(gap_class),
        "gap_reg_importance": original_feature_importance(gap_reg),
        "structure_roc": (fpr_s, tpr_s, auc_s),
        "gap_class_roc": (fpr_g, tpr_g, auc_g),
        "regression": (y_true, y_pred, r2, mse),
        "counts": {
            "structure_test": int(len(struct_test)),
            "gap_class_test": int(len(gap_test)),
            "gap_reg_test": int(len(reg_test)),
        },
    }


def style_axes(ax, label: str, title: str):
    ax.set_title(title, loc="left", pad=10, fontsize=12, fontweight="bold")
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.tick_params(direction="in", length=4, width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")


def plot_importance(ax, series: pd.Series, label: str, title: str):
    style_axes(ax, label, title)
    names = [latex_label(name) for name in series.index]
    values = series.to_numpy(dtype=float)
    x = np.arange(len(values))
    ax.bar(x, values, color=BAR_COLOR, width=0.72, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=9)
    ax.set_ylabel("Feature Relative Importance (%)")
    ymax = max(25.0, float(np.ceil(values.max() / 5.0) * 5.0 + 1.0))
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.grid(False)


def plot_roc(ax, fpr, tpr, auc: float, label: str, title: str):
    style_axes(ax, label, title)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, zorder=1)
    ax.plot(fpr, tpr, color=ROC_COLOR, linewidth=2.2, label=f"AUC = {auc:.2f}", zorder=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black", fontsize=10)
    ax.grid(False)


def plot_parity(ax, y_true, y_pred, r2: float, mse: float, label: str, title: str):
    style_axes(ax, label, title)
    lo = 0.0
    hi = max(5.0, float(np.max(np.r_[y_true, y_pred])) + 0.2)
    ax.scatter(y_true, y_pred, s=28, color=SCATTER_COLOR, alpha=0.75, edgecolors="none", zorder=2)
    ax.plot([lo, hi], [lo, hi], color=DIAG_COLOR, linewidth=1.8, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Test bandgap $E_g^{PBE}$ (eV)")
    ax.set_ylabel(r"Prediction bandgap $E_g^{ML}$ (eV)")
    text = f"$R^2 = {r2:.2f}$\n$MSE = {mse:.2f}$ eV"
    ax.text(
        0.97, 0.05, text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                               edgecolor="black", linewidth=0.9),
    )
    ax.grid(False)


def draw_figure(results: dict, out: Path, plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.1,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), layout="constrained")

    plot_importance(axes[0, 0], results["structure_importance"], "(a)", "Formability classification")
    plot_importance(axes[0, 1], results["gap_class_importance"], "(b)", "Bandgap classification")
    plot_importance(axes[0, 2], results["gap_reg_importance"], "(c)", "Bandgap regression")

    fpr_s, tpr_s, auc_s = results["structure_roc"]
    fpr_g, tpr_g, auc_g = results["gap_class_roc"]
    y_true, y_pred, r2, mse = results["regression"]
    plot_roc(axes[1, 0], fpr_s, tpr_s, auc_s, "(d)", "Formability classification")
    plot_roc(axes[1, 1], fpr_g, tpr_g, auc_g, "(e)", "Bandgap classification")
    plot_parity(axes[1, 2], y_true, y_pred, r2, mse, "(f)", "Bandgap regression")

    for extension in ("png", "pdf"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
        atomic_bytes(out / f"model_performance_abcdef.{extension}", buffer.getvalue())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = args.base_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else base / "results"
    out.mkdir(parents=True, exist_ok=True)

    models, frame = load_models_and_data(base)
    results = evaluate_models(models, frame)
    fig = draw_figure(results, out, plt)

    summary = {
        "structure_auc": results["structure_roc"][2],
        "gap_class_auc": results["gap_class_roc"][2],
        "regression_r2": results["regression"][2],
        "regression_mse_eV": results["regression"][3],
        "counts": results["counts"],
        "structure_top_features": results["structure_importance"].round(3).to_dict(),
        "gap_class_top_features": results["gap_class_importance"].round(3).to_dict(),
        "gap_reg_top_features": results["gap_reg_importance"].round(3).to_dict(),
        "no_confusion_matrix": True,
        "importance_source": "pipeline model.feature_importances_ aggregated to original inputs",
    }
    atomic_bytes(
        out / "model_performance_metadata.json",
        json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps({k: summary[k] for k in [
        "structure_auc", "gap_class_auc", "regression_r2", "regression_mse_eV", "counts"
    ]}, ensure_ascii=False, indent=2))
    print("Saved:", out / "model_performance_abcdef.png")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
