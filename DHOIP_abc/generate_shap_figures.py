"""计算已保存模型的真实 Tree SHAP，并生成依赖散点图。

直接在 PyCharm 运行；无窗口执行：python generate_shap_figures.py --no-show
生成两版：论文原特征（未使用特征明确标为0）和当前模型实际选用特征。
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from generate_dhoip_abc import atomic_bytes, frame_payload, load_inputs, predict_tables


def grouped_tree_shap(pipeline, frame, task):
    preprocessor = pipeline.named_steps["preprocessor"]
    tree_model = pipeline.named_steps["model"]
    X = frame[list(pipeline.feature_names_in_)]
    matrix = preprocessor.transform(X)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    # 相同编码输入只计算一次，再按原顺序还原。无抽样、无抖动、无模拟数据。
    unique, inverse = np.unique(matrix, axis=0, return_inverse=True)
    explainer = shap.TreeExplainer(tree_model, model_output="raw", feature_perturbation="tree_path_dependent")
    unique_values = np.asarray(explainer.shap_values(unique, check_additivity=True))
    if unique_values.ndim != 2:
        raise ValueError(f"预期单输出SHAP矩阵，实际形状为{unique_values.shape}")
    values = unique_values[inverse]
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    grouped = {}
    covered = []
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        span = preprocessor.output_indices_[name]
        if name == "numeric":
            if span.stop - span.start != len(columns):
                raise ValueError("数字预处理列数变化，需重新核对原始特征映射。")
            for offset, column in enumerate(columns):
                index = span.start + offset
                grouped[column] = values[:, index]
                covered.append(index)
        elif name == "categorical":
            if transformer.drop_idx_ is not None:
                raise ValueError("本脚本要求当前保存的无drop独热编码器。")
            index = span.start
            for column, categories in zip(columns, transformer.categories_):
                stop = index + len(categories)
                grouped[column] = values[:, index:stop].sum(axis=1)
                covered.extend(range(index, stop))
                index = stop
            if index != span.stop:
                raise ValueError("独热编码映射未覆盖全部编码列。")
        else:
            raise ValueError("未处理的预处理步骤：" + name)
    if sorted(covered) != list(range(matrix.shape[1])):
        raise ValueError("SHAP编码列覆盖错误。")
    prediction = pipeline.decision_function(X) if task == "structure" else pipeline.predict(X)
    reconstruction = baseline + np.sum(np.column_stack(list(grouped.values())), axis=1)
    error = float(np.max(np.abs(reconstruction - prediction)))
    if error > 1e-8:
        raise AssertionError(f"SHAP加和检查失败：最大误差{error}")
    return grouped, baseline, np.asarray(prediction), error, len(unique)


def build_shap_tables(models, candidates, known, splits, prediction_tables):
    partition = splits.set_index("System")["partition"]
    selected_ids = prediction_tables["screened_bandgap_predictions.csv"]["System"]
    known_reg = known.loc[(known["perovskite"] == 1) & (known["LowGap_PBE"] > 0.2)].copy()
    candidate_reg = candidates.loc[candidates["System"].isin(selected_ids)].copy()
    structure_frame = pd.concat([known, candidates], ignore_index=True, sort=False)
    reg_frame = pd.concat([known_reg, candidate_reg], ignore_index=True, sort=False)
    frames = [("structure", models[0], structure_frame, len(known)),
              ("bandgap", models[2], reg_frame, len(known_reg))]
    tables, metadata = {}, {}
    for task, model, frame, known_count in frames:
        grouped, base, output, error, unique_count = grouped_tree_shap(model, frame, task)
        result = frame[["System", "A_site", "B_site", "Bprime_site", "X1_site"]].copy()
        result["dataset"] = np.where(np.arange(len(frame)) < known_count, "known_train_test", "candidate_prediction")
        result["partition"] = result["System"].map(partition).fillna("candidate")
        result["base_value"] = base
        result["explained_output"] = output
        for feature, shap_values in grouped.items():
            result["value: " + feature] = frame[feature].values
            result["SHAP: " + feature] = shap_values
        if task == "structure":
            feature = "r_A − r_B"
            result["value: " + feature] = (frame["r_A"] - frame["r_B"]).values
            # 该输入不进入预处理器，模型直接依赖为零。不是从拟合线估算的SHAP。
            result["SHAP: " + feature] = 0.0
            units = "log-odds for perovskite class 1"
            filename = "structure_SHAP.csv"
        else:
            feature = "χ_B′ − χ_X2"
            x2 = frame["χ_X2"].copy()
            # 已知训练材料为纯卤素，X2=X1；候选数据保留真实混卤X2。
            is_known = np.arange(len(frame)) < known_count
            x2.loc[is_known] = frame.loc[is_known, "χ_X1"]
            if x2.isna().any():
                raise ValueError("X2电负性缺失。")
            result["value: " + feature] = (frame["χ_B′"] - x2).values
            result["SHAP: " + feature] = 0.0
            units = "eV"
            filename = "bandgap_regression_SHAP.csv"
        tables[filename] = result
        metadata[task] = {
            "known_rows": known_count, "candidate_rows": len(frame) - known_count,
            "unique_encoded_inputs": unique_count, "base_value": base,
            "output_units": units, "max_additivity_error": error,
            "selected_features": list(grouped),
            "categorical_attribution": "sum of Tree SHAP values for one-hot columns of each original category",
        }
    definitions = [
        ["reference", "a", "structure", "r_B + r_B′", True, ""],
        ["reference", "b", "structure", "r_A − r_B", False, "not selected by saved model; SHAP=0"],
        ["reference", "c", "bandgap", "χ_B + χ_B′", True, "positive Pauling electronegativity values"],
        ["reference", "d", "bandgap", "χ_B′ − χ_X2", False, "not selected by saved model; SHAP=0"],
        ["used_features", "a", "structure", "r_B + r_B′", True, "same as reference"],
        ["used_features", "b", "structure", "r_A / r_B", True, "replaces the unused radius difference"],
        ["used_features", "c", "bandgap", "χ_B + χ_B′", True, "same as reference"],
        ["used_features", "d", "bandgap", "χ_B′ − χ_X1", True, "replaces the unused X2 feature with the selected X1 feature"],
    ]
    tables["SHAP_feature_status.csv"] = pd.DataFrame(definitions, columns=["figure", "panel", "model", "feature", "selected_by_model", "note"])
    return tables, metadata


def plot_shap(tables, out, plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 13, "axes.titlesize": 14,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 10, "axes.linewidth": 1.1,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    source = {"structure": tables["structure_SHAP.csv"], "bandgap": tables["bandgap_regression_SHAP.csv"]}
    definitions = {
        "reference": [
            ("structure", "r_B + r_B′", r"$r_B+r_{B^{\prime}}$ (Å)", True),
            ("structure", "r_A − r_B", r"$r_A-r_B$ (Å)", False),
            ("bandgap", "χ_B + χ_B′", r"$\chi_B+\chi_{B^{\prime}}$ (Pauling)", True),
            ("bandgap", "χ_B′ − χ_X2", r"$\chi_{B^{\prime}}-\chi_{X_2}$ (Pauling)", False),
        ],
        "used_features": [
            ("structure", "r_B + r_B′", r"$r_B+r_{B^{\prime}}$ (Å)", True),
            ("structure", "r_A / r_B", r"$r_A/r_B$", True),
            ("bandgap", "χ_B + χ_B′", r"$\chi_B+\chi_{B^{\prime}}$ (Pauling)", True),
            ("bandgap", "χ_B′ − χ_X1", r"$\chi_{B^{\prime}}-\chi_{X_1}$ (Pauling)", True),
        ],
    }
    trends = {}

    def panel(ax, index, definition, variant):
        task, feature, xlabel, used = definition
        frame = source[task]
        xcol, ycol = "value: " + feature, "SHAP: " + feature
        predicted = frame.loc[frame["dataset"] == "candidate_prediction"]
        known = frame.loc[frame["dataset"] == "known_train_test"]
        ax.scatter(predicted[xcol], predicted[ycol], s=19, color="#92969B", alpha=0.38,
                   linewidths=0, rasterized=True, label=f"Candidates (n={len(predicted):,})")
        ax.scatter(known[xcol], known[ycol], s=25, facecolors="none", edgecolors="#E66B55",
                   alpha=0.73, linewidths=0.8, rasterized=True,
                   label=f"Known train + test (n={len(known):,})")
        ax.axhline(0, color="#69727A", linewidth=0.8, linestyle="--", zorder=0)
        if used:
            x, y = known[xcol].to_numpy(dtype=float), known[ycol].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            fit_x = np.array([float(frame[xcol].min()), float(frame[xcol].max())])
            ax.plot(fit_x, slope * fit_x + intercept, color="#242424", linewidth=1.5,
                    label="Linear fit: known data")
            trends[f"{variant}_{index}"] = {"feature": feature, "slope": float(slope), "intercept": float(intercept), "fit_population": "known train + test only"}
        else:
            ax.set_ylim(-0.5, 0.5)
            ax.text(0.5, 0.30, "Not used by saved model\nSHAP contribution = 0", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12, color="#343A40")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHAP value (log-odds)" if task == "structure" else "SHAP value (eV)")
        ax.set_title("Structure formability" if task == "structure" else "PBE band gap regression", loc="left", pad=13)
        ax.text(-0.15, 1.045, f"({chr(97 + index)})", transform=ax.transAxes, fontsize=20, fontweight="bold")
        ax.grid(color="#E5E8EB", linewidth=0.65, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=4)
        ax.legend(loc="upper left" if index == 0 or not used else "upper right", frameon=True, framealpha=0.94)
        ax.margins(x=0.06, y=0.17)

    def save(fig, name):
        for extension in ["png", "pdf"]:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
            atomic_bytes(out / f"{name}.{extension}", buffer.getvalue())

    figures = []
    for variant, definitions_list in definitions.items():
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6), layout="constrained")
        for index, (ax, definition) in enumerate(zip(axes.flat, definitions_list)):
            panel(ax, index, definition, variant)
        save(fig, f"SHAP_{variant}")
        figures.append(fig)
        if variant == "used_features":
            for index, definition in enumerate(definitions_list):
                single, ax = plt.subplots(figsize=(7.2, 5.5), layout="constrained")
                panel(ax, index, definition, variant)
                buffer = io.BytesIO()
                single.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                atomic_bytes(out / f"SHAP_{chr(97 + index)}_used_feature.png", buffer.getvalue())
                plt.close(single)
    return figures, trends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
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
    predictions = predict_tables(models, candidates, known, splits)
    tables, metadata = build_shap_tables(models, candidates, known, splits, predictions)
    if args.defer_csv:
        payload = {name: frame_payload(df) for name, df in tables.items()}
        atomic_bytes(out / "shap_tables_intermediate.json", json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    else:
        for name, frame in tables.items():
            atomic_bytes(out / name, frame.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig"))
    figures, trends = plot_shap(tables, out, plt)
    metadata.update({"shap_version": shap.__version__, "feature_perturbation": "tree_path_dependent",
                     "model_output": "raw", "trend_lines": trends,
                     "method_source": "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html",
                     "unused_feature_policy": "0 direct contribution for features dropped from the saved pipeline",
                     "no_model_retraining": True,
                     "interpretation": "model attributions, not causal effects; correlated engineered inputs can share credit"})
    atomic_bytes(out / "SHAP_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps({key: value for key, value in metadata.items() if key in ["structure", "bandgap", "shap_version"]}, ensure_ascii=False, indent=2))
    print("Saved SHAP figures and data to:", out)
    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
