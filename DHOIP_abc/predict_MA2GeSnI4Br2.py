"""预测 MA2GeSnI4Br2 的可形成性与 PBE 带隙。

直接加载三个已保存模型，不重新训练。
Sn 不在原训练/候选元素表中：半径与电负性按 Shannon(Sn2+, CN=6)=1.18 Å、
Pauling χ(Sn)=1.96 构造；类别特征 OneHot 对未见 Sn 使用 handle_unknown='ignore'。
混卤构型按数据集惯例生成（每个 X 槽位代表 2 个卤素原子）。

运行：python predict_MA2GeSnI4Br2.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from generate_dhoip_abc import MODEL_FILES, atomic_bytes, positive_probability

# 与 candidates.csv / training_data.csv 中已有元素取值保持一致
SITE_PROPS = {
    "MA": {"r": 2.17, "chi": None, "ox": None},
    "Ge": {"r": 0.73, "chi": 2.01, "ox": 2},
    # Sn 不在原库：采用 Shannon Sn2+ (VI) 与 Pauling 电负性
    "Sn": {"r": 1.18, "chi": 1.96, "ox": 2},
    "Br": {"r": 1.96, "chi": 2.96, "ox": -1},
    "I": {"r": 2.20, "chi": 2.66, "ox": -1},
}

# I4Br2：三个 X 槽位各 2 个卤素，共 6 个卤素原子
HALIDE_CONFIGS_I4BR2 = [
    ("Br", "I", "I"),
    ("I", "Br", "I"),
    ("I", "I", "Br"),
]


def _pair_features(prefix_a: str, a: float, prefix_b: str, b: float) -> dict:
    return {
        f"{prefix_a} + {prefix_b}": a + b,
        f"{prefix_a} − {prefix_b}": a - b,
        f"{prefix_a} / {prefix_b}": a / b,
    }


def build_row(a_site: str, b_site: str, bp_site: str, x1: str, x2: str, x3: str) -> dict:
    r_a = SITE_PROPS[a_site]["r"]
    r_b = SITE_PROPS[b_site]["r"]
    r_bp = SITE_PROPS[bp_site]["r"]
    r_x1 = SITE_PROPS[x1]["r"]
    r_x2 = SITE_PROPS[x2]["r"]
    r_x3 = SITE_PROPS[x3]["r"]
    chi_b = SITE_PROPS[b_site]["chi"]
    chi_bp = SITE_PROPS[bp_site]["chi"]
    chi_x1 = SITE_PROPS[x1]["chi"]
    chi_x2 = SITE_PROPS[x2]["chi"]
    chi_x3 = SITE_PROPS[x3]["chi"]

    counts = {"Cl": 0, "Br": 0, "I": 0}
    for x in (x1, x2, x3):
        counts[x] += 2

    formula = f"{a_site}2{b_site}{bp_site}I{counts['I']}Br{counts['Br']}"
    row = {
        "System": f"{a_site}-{b_site}-{bp_site}-{x1}{x2}{x3}",
        "Formula": formula,
        "A_site": a_site,
        "B_site": b_site,
        "Bprime_site": bp_site,
        "X1_site": x1,
        "X2_site": x2,
        "X3_site": x3,
        "Cl_count": counts["Cl"],
        "Br_count": counts["Br"],
        "I_count": counts["I"],
        "Oxidation_B": SITE_PROPS[b_site]["ox"],
        "Oxidation_Bprime": SITE_PROPS[bp_site]["ox"],
        "r_A": r_a,
        "r_B": r_b,
        "r_B′": r_bp,
        "r_X1": r_x1,
        "r_X2": r_x2,
        "r_X3": r_x3,
        "χ_B": chi_b,
        "χ_B′": chi_bp,
        "χ_X1": chi_x1,
        "χ_X2": chi_x2,
        "χ_X3": chi_x3,
    }
    row.update(_pair_features("r_A", r_a, "r_B", r_b))
    row.update(_pair_features("r_A", r_a, "r_B′", r_bp))
    row.update(_pair_features("r_A", r_a, "r_X1", r_x1))
    row.update(_pair_features("r_A", r_a, "r_X2", r_x2))
    row.update(_pair_features("r_A", r_a, "r_X3", r_x3))
    row.update(_pair_features("r_B", r_b, "r_B′", r_bp))
    row.update(_pair_features("r_B", r_b, "r_X1", r_x1))
    row.update(_pair_features("r_B", r_b, "r_X2", r_x2))
    row.update(_pair_features("r_B", r_b, "r_X3", r_x3))
    row.update(_pair_features("r_B′", r_bp, "r_X1", r_x1))
    row.update(_pair_features("r_B′", r_bp, "r_X2", r_x2))
    row.update(_pair_features("r_B′", r_bp, "r_X3", r_x3))
    row.update(_pair_features("r_X1", r_x1, "r_X2", r_x2))
    row.update(_pair_features("r_X1", r_x1, "r_X3", r_x3))
    row.update(_pair_features("r_X2", r_x2, "r_X3", r_x3))
    row.update(_pair_features("χ_B", chi_b, "χ_B′", chi_bp))
    row.update(_pair_features("χ_B", chi_b, "χ_X1", chi_x1))
    row.update(_pair_features("χ_B", chi_b, "χ_X2", chi_x2))
    row.update(_pair_features("χ_B", chi_b, "χ_X3", chi_x3))
    row.update(_pair_features("χ_B′", chi_bp, "χ_X1", chi_x1))
    row.update(_pair_features("χ_B′", chi_bp, "χ_X2", chi_x2))
    row.update(_pair_features("χ_B′", chi_bp, "χ_X3", chi_x3))
    row.update(_pair_features("χ_X1", chi_x1, "χ_X2", chi_x2))
    row.update(_pair_features("χ_X1", chi_x1, "χ_X3", chi_x3))
    row.update(_pair_features("χ_X2", chi_x2, "χ_X3", chi_x3))
    return row


def build_ma2gesni4br2_frame() -> pd.DataFrame:
    rows = []
    for b, bp in (("Ge", "Sn"), ("Sn", "Ge")):
        for x1, x2, x3 in HALIDE_CONFIGS_I4BR2:
            rows.append(build_row("MA", b, bp, x1, x2, x3))
    return pd.DataFrame(rows)


def predict_composition(models, frame: pd.DataFrame) -> pd.DataFrame:
    structure, gap_class, gap_reg = models
    n = len(frame)
    p_structure = positive_probability(structure, frame)
    perovskite = structure.predict(frame[list(structure.feature_names_in_)]).astype(int)
    structure_mask = perovskite == 1

    p_gap = np.full(n, np.nan)
    gap_label = np.full(n, np.nan)
    if structure_mask.any():
        sub = frame.loc[structure_mask]
        p_gap[structure_mask] = positive_probability(gap_class, sub)
        gap_label[structure_mask] = gap_class.predict(sub[list(gap_class.feature_names_in_)])

    selected = structure_mask & (gap_label == 1)
    gap_pred = np.full(n, np.nan)
    if selected.any():
        gap_pred[selected] = gap_reg.predict(frame.loc[selected, list(gap_reg.feature_names_in_)])

    # 因模型未使用 X2/X3，额外给出仅依赖 X1 的“有效输入”分组均值
    result = frame[[
        "System", "Formula", "A_site", "B_site", "Bprime_site",
        "X1_site", "X2_site", "X3_site", "Br_count", "I_count",
        "r_B", "r_B′", "χ_B", "χ_B′", "r_X1", "χ_X1",
    ]].copy()
    result["probability_perovskite"] = p_structure
    result["predicted_perovskite"] = perovskite
    result["probability_Eg_gt_0p2"] = p_gap
    result["predicted_Eg_gt_0p2_class"] = gap_label
    result["predicted_PBE_bandgap_eV"] = gap_pred
    result["screening_stage"] = np.where(
        ~structure_mask, "predicted_non_perovskite",
        np.where(selected, "bandgap_regression_completed", "predicted_Eg_le_0p2"),
    )
    result["note_Sn_OOD"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base = args.base_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else base / "results"
    out.mkdir(parents=True, exist_ok=True)

    models = [joblib.load(base / "models" / name) for name in MODEL_FILES]
    frame = build_ma2gesni4br2_frame()
    for model in models:
        missing = sorted(set(model.feature_names_in_) - set(frame.columns))
        if missing:
            raise ValueError("构造特征缺少列：" + str(missing))

    result = predict_composition(models, frame)
    csv_path = out / "MA2GeSnI4Br2_predictions.csv"
    atomic_bytes(csv_path, result.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig"))

    # 模型实际只用 X1，对 B/B′ 与 X1 汇总
    summary = (
        result.groupby(["B_site", "Bprime_site", "X1_site"], sort=False)
        .agg(
            n_configs=("System", "count"),
            probability_perovskite=("probability_perovskite", "mean"),
            predicted_perovskite=("predicted_perovskite", "max"),
            probability_Eg_gt_0p2=("probability_Eg_gt_0p2", "mean"),
            predicted_PBE_bandgap_eV=("predicted_PBE_bandgap_eV", "mean"),
            screening_stage=("screening_stage", "first"),
        )
        .reset_index()
    )
    atomic_bytes(
        out / "MA2GeSnI4Br2_summary.csv",
        summary.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig"),
    )

    finished = result.loc[result["screening_stage"] == "bandgap_regression_completed"]
    meta = {
        "target": "MA2GeSnI4Br2",
        "sn_not_in_training_set": True,
        "sn_radius_A": SITE_PROPS["Sn"]["r"],
        "sn_pauling_chi": SITE_PROPS["Sn"]["chi"],
        "ge_radius_A": SITE_PROPS["Ge"]["r"],
        "ge_pauling_chi": SITE_PROPS["Ge"]["chi"],
        "n_configurations": int(len(result)),
        "n_bandgap_predictions": int(len(finished)),
        "predicted_PBE_bandgap_eV_min": None if finished.empty else float(finished["predicted_PBE_bandgap_eV"].min()),
        "predicted_PBE_bandgap_eV_max": None if finished.empty else float(finished["predicted_PBE_bandgap_eV"].max()),
        "predicted_PBE_bandgap_eV_mean": None if finished.empty else float(finished["predicted_PBE_bandgap_eV"].mean()),
        "limitation": (
            "Sn is absent from the saved models' categorical vocabulary; "
            "numeric Sn descriptors are used, categorical Sn one-hots are ignored. "
            "Saved models omit X2/X3-selected features, so halide permutations with the same X1 give identical scores."
        ),
    }
    atomic_bytes(out / "MA2GeSnI4Br2_metadata.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

    print("MA2GeSnI4Br2 predictions (grouped by B/B'/X1):")
    print(summary.to_string(index=False))
    if finished.empty:
        print("No configuration passed both classifiers; bandgap regression was not run.")
    else:
        print(
            f"\nPredicted PBE bandgap range: "
            f"{meta['predicted_PBE_bandgap_eV_min']:.4f} – "
            f"{meta['predicted_PBE_bandgap_eV_max']:.4f} eV "
            f"(mean {meta['predicted_PBE_bandgap_eV_mean']:.4f} eV)"
        )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
