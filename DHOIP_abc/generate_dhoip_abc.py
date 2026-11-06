"""加载已保存模型，预测混卤候选材料并生成论文式 a、b、c 散点图。

PyCharm：解压后直接运行本文件。默认弹出合并图，同时保存图片和 UTF-8 BOM CSV。
命令行无窗口：python generate_dhoip_abc.py --no-show
本脚本只加载已有模型，不重新训练。模型文件应来自可信来源。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_FILES = ["01_structure_GBC.joblib", "02_bandgap_GBC.joblib", "03_bandgap_GBR.joblib"]
R_BPRIME = "r_B′"
CHI_BPRIME = "χ_B′"


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def frame_payload(frame: pd.DataFrame) -> dict:
    # pandas JSON 将缺失值转为 null，确保 JSON 有效；数值直接取原数组，保留双精度。
    clean = frame.astype(object).where(pd.notna(frame), None)
    return {"columns": list(frame.columns), "rows": clean.values.tolist()}


def load_inputs(base: Path) -> tuple:
    paths = [base / "models" / name for name in MODEL_FILES]
    paths += [base / "data" / name for name in ["candidates.csv", "training_data.csv", "splits.csv"]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少文件，请先完整解压压缩包：\n" + "\n".join(missing))
    models = [joblib.load(p) for p in paths[:3]]
    candidates = pd.read_csv(paths[3], encoding="utf-8-sig", float_precision="round_trip")
    known = pd.read_csv(paths[4], encoding="utf-8-sig", float_precision="round_trip")
    splits = pd.read_csv(paths[5], encoding="utf-8-sig", float_precision="round_trip")
    for frame in [candidates, known, splits]:
        if frame["System"].isna().any() or frame["System"].duplicated().any():
            raise ValueError("System 编号缺失或重复，无法可靠匹配材料。")
    for model in models:
        features = list(model.feature_names_in_)
        absent = sorted(set(features) - set(candidates.columns))
        if absent:
            raise ValueError("候选数据缺少模型输入列：" + str(absent))
        for column in features:
            if candidates[column].isna().any():
                raise ValueError("模型输入有缺失值：" + column)
    return models, candidates, known, splits, paths


def positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    index = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(index) != 1:
        raise ValueError("模型正类必须为 1，请核对模型类别定义。")
    return model.predict_proba(frame[list(model.feature_names_in_)])[:, index[0]]


def predict_tables(models, candidates, known, splits):
    structure, gap_class, gap_reg = models
    n = len(candidates)
    p_structure = positive_probability(structure, candidates)
    perovskite = structure.predict(candidates[list(structure.feature_names_in_)]).astype(int)
    structure_mask = perovskite == 1
    p_gap = np.full(n, np.nan)
    gap_class_label = np.full(n, np.nan)
    if structure_mask.any():
        frame = candidates.loc[structure_mask]
        p_gap[structure_mask] = positive_probability(gap_class, frame)
        gap_class_label[structure_mask] = gap_class.predict(frame[list(gap_class.feature_names_in_)])
    selected_mask = structure_mask & (gap_class_label == 1)
    estimated_gap = np.full(n, np.nan)
    if selected_mask.any():
        estimated_gap[selected_mask] = gap_reg.predict(
            candidates.loc[selected_mask, list(gap_reg.feature_names_in_)]
        )
    metadata = ["System", "Formula", "Family_ID", "A_site", "B_site", "Bprime_site",
                "X1_site", "X2_site", "X3_site", "Cl_count", "Br_count", "I_count"]
    result = candidates[metadata].copy()
    result["r_B_plus_r_Bprime_A"] = candidates["r_B"] + candidates[R_BPRIME]
    result["r_A_over_r_B"] = candidates["r_A"] / candidates["r_B"]
    result["chi_B_plus_chi_Bprime_Pauling"] = candidates["χ_B"] + candidates[CHI_BPRIME]
    result["mean_X_radius_A"] = candidates[["r_X1", "r_X2", "r_X3"]].mean(axis=1)
    result["probability_perovskite"] = p_structure
    result["predicted_perovskite"] = perovskite
    result["probability_Eg_gt_0p2_given_structure_pass"] = p_gap
    result["predicted_Eg_gt_0p2_class"] = gap_class_label
    result["selected_for_bandgap_regression"] = selected_mask.astype(int)
    result["predicted_PBE_bandgap_eV"] = estimated_gap
    result["screening_stage"] = np.where(
        ~structure_mask, "predicted_non_perovskite",
        np.where(selected_mask, "bandgap_regression_completed", "predicted_Eg_le_0p2")
    )
    selected = result.loc[selected_mask].copy()
    known_good = known.loc[(known["perovskite"] == 1) & (known["LowGap_PBE"] > 0.2)].copy()
    known_plot = known_good[["System", "A_site", "B_site", "Bprime_site", "X1_site"]].copy()
    known_plot["chi_B_plus_chi_Bprime_Pauling"] = known_good["χ_B"] + known_good[CHI_BPRIME]
    known_plot["DFT_PBE_bandgap_eV"] = known_good["LowGap_PBE"]
    known_plot = known_plot.merge(splits[["System", "partition"]], on="System", how="left", validate="one_to_one")
    if known_plot["partition"].isna().any():
        raise ValueError("已知材料无法匹配训练/测试划分。")
    low_gap_count = int((selected["predicted_PBE_bandgap_eV"] <= 0.2).sum())
    summary_items = [
        ("all_candidate_configurations", n),
        ("predicted_perovskite", int(structure_mask.sum())),
        ("predicted_non_perovskite", int((~structure_mask).sum())),
        ("gap_classifier_pass_and_regression_completed", int(selected_mask.sum())),
        ("gap_classifier_reject_after_structure_pass", int((structure_mask & ~selected_mask).sum())),
        ("regression_Eg_le_0p2_after_classifier_pass", low_gap_count),
        ("selected_unique_compositions", int(selected["Formula"].nunique())),
        ("known_DFT_comparison_rows", len(known_plot)),
    ]
    summary = pd.DataFrame(summary_items, columns=["item", "count"])
    tables = {
        "candidate_predictions.csv": result,
        "screened_bandgap_predictions.csv": selected,
        "figure_b_known_DFT.csv": known_plot,
        "screening_summary.csv": summary,
    }
    return tables


def draw_figures(tables, out: Path, plt):
    all_rows = tables["candidate_predictions.csv"]
    selected = tables["screened_bandgap_predictions.csv"]
    known = tables["figure_b_known_DFT.csv"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10, "axes.linewidth": 1.1, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    def style(ax, label, title):
        ax.set_title(title, loc="left", pad=13)
        ax.text(-0.15, 1.045, label, transform=ax.transAxes, fontsize=19, fontweight="bold")
        ax.tick_params(direction="out", length=4, width=1)
        ax.grid(color="#E4E7EA", linewidth=0.65, alpha=0.7)
        ax.set_axisbelow(True)

    def panel_a(ax):
        style(ax, "(a)", "Predicted structure class")
        for label, color, marker, text in [
            (0, "#4C89A4", "x", "Non-perovskite"),
            (1, "#D65B3F", "o", "Perovskite"),
        ]:
            d = all_rows.loc[all_rows["predicted_perovskite"] == label]
            ax.scatter(d["r_B_plus_r_Bprime_A"], d["r_A_over_r_B"], s=28,
                       c=color, marker=marker, alpha=0.40 if label == 0 else 0.45,
                       linewidths=0.8 if label == 0 else 0,
                       label=f"{text} (n={len(d):,})", rasterized=True)
        ax.set_xlabel(r"$r_B+r_{B^{\prime}}$ (Å)")
        ax.set_ylabel(r"$r_A/r_B$")
        ax.legend(loc="upper right", frameon=True, framealpha=0.94)
        ax.margins(0.06)

    x_col = "chi_B_plus_chi_Bprime_Pauling"
    y_col = "predicted_PBE_bandgap_eV"
    y_values = np.r_[known["DFT_PBE_bandgap_eV"].values, selected[y_col].values]
    ymin = min(0.0, float(np.min(y_values)) - 0.15)
    ymax = float(np.max(y_values)) + 0.65
    xmin = min(float(known[x_col].min()), float(selected[x_col].min())) - 0.17
    xmax = max(float(known[x_col].max()), float(selected[x_col].max())) + 0.17

    def gap_axes(ax):
        ax.set_xlabel(r"$\chi_B+\chi_{B^{\prime}}$ (Pauling)")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

    def panel_b(ax):
        style(ax, "(b)", "Known DFT and candidate predictions")
        ax.scatter(selected[x_col], selected[y_col], s=21, color="#006D54",
                   alpha=0.34, linewidths=0, rasterized=True,
                   label=f"Candidate predictions (n={len(selected):,})")
        ax.scatter(known[x_col], known["DFT_PBE_bandgap_eV"], s=30,
                   facecolors="none", edgecolors="#69B879", alpha=0.90,
                   linewidths=0.9, rasterized=True,
                   label=f"Known DFT: train + test (n={len(known):,})")
        gap_axes(ax)
        ax.set_ylabel(r"PBE band gap $E_g$ (eV)")
        ax.legend(loc="upper right", frameon=True, framealpha=0.94)

    def panel_c(ax):
        style(ax, "(c)", "Candidate predictions by X-site radius")
        d = selected.sort_values("mean_X_radius_A", kind="stable")
        scatter = ax.scatter(d[x_col], d[y_col], c=d["mean_X_radius_A"],
                             cmap="YlOrRd", s=25, alpha=0.7, linewidths=0,
                             rasterized=True)
        gap_axes(ax)
        ax.set_ylabel(r"Predicted PBE band gap $E_g$ (eV)")
        cb = ax.figure.colorbar(scatter, ax=ax, fraction=0.048, pad=0.035)
        cb.set_label("Mean X-site radius (Å)", fontsize=11, labelpad=9)
        cb.ax.tick_params(labelsize=10)

    def save(fig, filename):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        atomic_bytes(out / filename, buffer.getvalue())

    # 先输出三张单图，便于分别放入 PPT。
    for letter, drawer in [("a", panel_a), ("b", panel_b), ("c", panel_c)]:
        fig, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
        drawer(ax)
        save(fig, f"figure_{letter}.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8), layout="constrained")
    for ax, drawer in zip(axes, [panel_a, panel_b, panel_c]):
        drawer(ax)
    save(fig, "figure_abc.png")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    atomic_bytes(out / "figure_abc.pdf", buffer.getvalue())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true", help="只保存文件，不弹出绘图窗口")
    parser.add_argument("--defer-csv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = args.base_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else base / "results"
    out.mkdir(parents=True, exist_ok=True)
    models, candidates, known, splits, paths = load_inputs(base)
    tables = predict_tables(models, candidates, known, splits)
    if args.defer_csv:
        payload = {name: frame_payload(df) for name, df in tables.items()}
        atomic_bytes(out / "tables_intermediate.json", json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    else:
        for name, frame in tables.items():
            text = frame.to_csv(index=False, lineterminator="\r\n")
            atomic_bytes(out / name, text.encode("utf-8-sig"))
    fig = draw_figures(tables, out, plt)
    selected = tables["screened_bandgap_predictions.csv"]
    provenance = {
        "input_sha256": {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "screening_counts": dict(tables["screening_summary.csv"].itertuples(index=False, name=None)),
        "candidate_predictions_are_DFT": False,
        "candidate_unit": "paired-halide-site configuration; formulas can repeat",
        "classification_rule": "saved Pipeline.predict; positive label 1; default probability threshold 0.5",
        "regression_scope": "only candidates passing both classifiers; other predictions remain blank",
        "gap_min_eV": float(selected["predicted_PBE_bandgap_eV"].min()),
        "gap_max_eV": float(selected["predicted_PBE_bandgap_eV"].max()),
        "known_DFT_scope": "true perovskite=1 and true PBE gap >0.2 eV; original train and test partitions",
        "X_radius_mean": "(r_X1+r_X2+r_X3)/3; each slot represents 2 halide atoms",
        "electronegativity_scale": "positive Pauling values from current dataset; no sign reversal to mimic paper",
        "limitations": [
            "All three saved models omit X2/X3-dependent selected features; mixed-halide discrimination is limited.",
            "Same selected model features can produce repeated values; no jitter or synthetic noise is added.",
            "Known DFT vs candidate distribution is not an accuracy evaluation for unlabeled candidates.",
            "Structure classification does not establish thermodynamic, dynamical or thermal stability."
        ],
    }
    atomic_bytes(out / "run_metadata.json", json.dumps(provenance, ensure_ascii=False, indent=2).encode("utf-8"))
    for key, value in provenance["screening_counts"].items():
        print(f"{key}: {value}")
    print(f"Predicted gap range: {provenance['gap_min_eV']:.6f} to {provenance['gap_max_eV']:.6f} eV")
    print(f"Saved to: {out}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
