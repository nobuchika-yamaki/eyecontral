#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chiasm_geometric_routing_improved.py

Improved analysis for a minimal geometric routing model of partial optic chiasm
decussation.

This script supersedes the previous weighted-objective-only analysis. It separates:

1. Exact analytical constraint analysis
   - Full enumeration of all 16 deterministic retinal-to-hemisphere architectures.
   - Direct test of binocular pairing and hemifield-separation constraints.
   - Z2 quotient analysis under global hemisphere relabeling.

2. Pareto analysis
   - Cost-vector comparison before applying arbitrary weights.

3. Weighted objective verification
   - Uses the previous theta/weight-selection logic only as a robustness check.
   - J_topo is removed from the primary objective because it is structurally non-informative
     in the binary hemifield model.

4. Boundary-noise robustness
   - Uses continuous component values rather than only exact-zero criteria.

5. Probabilistic routing analysis
   - Optimizes p_N and p_T over the continuous square [0,1]^2 using an exact candidate
     set for the piecewise-bilinear objective, rather than a coarse probability grid.

6. Null models
   - Eye-only, pair-only, segment-only, pair+segment objectives.
   - Retinal-geometry null enumeration.

All outputs are written to:
~/Desktop/chiasm_geometric_routing_improved_results
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEFT_EYE = -1
RIGHT_EYE = +1
TEMPORAL = -1
NASAL = +1
LEFT_HEMI = -1
RIGHT_HEMI = +1

EYES = [LEFT_EYE, RIGHT_EYE]
RETINAL_SIDES = [TEMPORAL, NASAL]
HEMISPHERES = [LEFT_HEMI, RIGHT_HEMI]
HEMIFIELDS = [-1, +1]

INPUT_ORDER = [
    (LEFT_EYE, TEMPORAL),
    (LEFT_EYE, NASAL),
    (RIGHT_EYE, TEMPORAL),
    (RIGHT_EYE, NASAL),
]

INPUT_LABEL = {
    (LEFT_EYE, TEMPORAL): "L_T",
    (LEFT_EYE, NASAL): "L_N",
    (RIGHT_EYE, TEMPORAL): "R_T",
    (RIGHT_EYE, NASAL): "R_N",
}

HEMI_LABEL = {LEFT_HEMI: "L", RIGHT_HEMI: "R"}

THETA_VALUES = np.round(np.arange(0.00, 1.0001, 0.01), 2)
LAMBDA_PAIR = 1.0
LAMBDA_SEG_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0]
LAMBDA_CROSS_VALUES = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
LAMBDA_SYM_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

BOUNDARY_NOISE_VALUES = [0.0, 0.01, 0.025, 0.05, 0.10]
N_X_PER_SIDE = 400
X_MIN_ABS = 0.005

EPS_TIE = 1.0e-12
B_BOOT = 2000
RNG_SEED = 20260428

OUTDIR = Path.home() / "Desktop" / "chiasm_geometric_routing_improved_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def sign_nonzero(value: float, fallback: int | None = None) -> int:
    if value > 0:
        return +1
    if value < 0:
        return -1
    if fallback is not None:
        return int(fallback)
    raise ValueError("sign_nonzero received zero without fallback")


def biological_retinal_map(e: int, h: int) -> int:
    return int(e * h)


def make_x_grid() -> np.ndarray:
    x_left = np.linspace(-1.0, -X_MIN_ABS, N_X_PER_SIDE)
    x_right = np.linspace(X_MIN_ABS, 1.0, N_X_PER_SIDE)
    return np.concatenate([x_left, x_right])


def h_from_x(x: float) -> int:
    return sign_nonzero(float(x))


def r_from_x_with_noise(e: int, x: float, zeta: float) -> int:
    h = h_from_x(x)
    fallback = biological_retinal_map(e, h)
    return sign_nonzero(e * x + zeta, fallback=fallback)


def arch_code(arch: Dict[Tuple[int, int], int]) -> str:
    return "".join("R" if arch[key] == RIGHT_HEMI else "L" for key in INPUT_ORDER)


def flip_hemispheres(arch: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    return {k: -v for k, v in arch.items()}


def quotient_code(arch: Dict[Tuple[int, int], int]) -> str:
    c1 = arch_code(arch)
    c2 = arch_code(flip_hemispheres(arch))
    return min(c1, c2)


def crossing_indicator(arch: Dict[Tuple[int, int], int], e: int, r: int) -> int:
    return int(arch[(e, r)] == -e)


def is_lr_symmetric(arch: Dict[Tuple[int, int], int]) -> bool:
    return all(arch[(-e, r)] == -arch[(e, r)] for e in EYES for r in RETINAL_SIDES)


def canonical_q_values(arch: Dict[Tuple[int, int], int]) -> Tuple[float, float]:
    q = {}
    for r in RETINAL_SIDES:
        c_left = crossing_indicator(arch, LEFT_EYE, r)
        c_right = crossing_indicator(arch, RIGHT_EYE, r)
        q[r] = float(c_left) if c_left == c_right else float("nan")
    return q[TEMPORAL], q[NASAL]


def canonical_label(arch: Dict[Tuple[int, int], int]) -> str:
    q_t, q_n = canonical_q_values(arch)
    if math.isnan(q_t) or math.isnan(q_n):
        return "noncanonical_or_lr_asymmetric"
    q_t_i, q_n_i = int(q_t), int(q_n)
    if q_n_i == 0 and q_t_i == 0:
        return "all_ipsilateral"
    if q_n_i == 1 and q_t_i == 1:
        return "all_contralateral"
    if q_n_i == 1 and q_t_i == 0:
        return "nasal_crossing_temporal_ipsilateral"
    if q_n_i == 0 and q_t_i == 1:
        return "nasal_ipsilateral_temporal_crossing"
    return "unclassified"


def is_partial_symmetric(arch: Dict[Tuple[int, int], int]) -> bool:
    if not is_lr_symmetric(arch):
        return False
    q_t, q_n = canonical_q_values(arch)
    if math.isnan(q_t) or math.isnan(q_n):
        return False
    return int(q_t) != int(q_n)


def build_architectures() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arch_id, outputs in enumerate(itertools.product(HEMISPHERES, repeat=4)):
        arch = {key: int(out) for key, out in zip(INPUT_ORDER, outputs)}
        q_t, q_n = canonical_q_values(arch)
        rows.append(
            {
                "arch_id": arch_id,
                "arch_code": arch_code(arch),
                "quotient_code": quotient_code(arch),
                "A_L_T": arch[(LEFT_EYE, TEMPORAL)],
                "A_L_N": arch[(LEFT_EYE, NASAL)],
                "A_R_T": arch[(RIGHT_EYE, TEMPORAL)],
                "A_R_N": arch[(RIGHT_EYE, NASAL)],
                "A_L_T_label": HEMI_LABEL[arch[(LEFT_EYE, TEMPORAL)]],
                "A_L_N_label": HEMI_LABEL[arch[(LEFT_EYE, NASAL)]],
                "A_R_T_label": HEMI_LABEL[arch[(RIGHT_EYE, TEMPORAL)]],
                "A_R_N_label": HEMI_LABEL[arch[(RIGHT_EYE, NASAL)]],
                "q_temporal": q_t,
                "q_nasal": q_n,
                "left_right_symmetric": is_lr_symmetric(arch),
                "partial_symmetric": is_partial_symmetric(arch),
                "canonical_label": canonical_label(arch),
                "arch": arch,
            }
        )
    return rows


def exact_components_for_geometry(
    arch: Dict[Tuple[int, int], int],
    retinal_map: Dict[Tuple[int, int], int] | None = None,
) -> Dict[str, float | bool]:
    if retinal_map is None:
        retinal_map = {(e, h): biological_retinal_map(e, h) for e in EYES for h in HEMIFIELDS}

    j_eye = float(np.mean([int(arch[(e, TEMPORAL)] != arch[(e, NASAL)]) for e in EYES]))

    pair_terms = []
    clusters = {}
    for h in HEMIFIELDS:
        s_left = arch[(LEFT_EYE, retinal_map[(LEFT_EYE, h)])]
        s_right = arch[(RIGHT_EYE, retinal_map[(RIGHT_EYE, h)])]
        pair_terms.append(int(s_left != s_right))
        clusters[h] = (s_left, s_right)
    j_pair = float(np.mean(pair_terms))

    d = {}
    for h in HEMIFIELDS:
        counts = {LEFT_HEMI: 0, RIGHT_HEMI: 0}
        for e in EYES:
            s = arch[(e, retinal_map[(e, h)])]
            counts[s] += 1
        d[h] = {s: counts[s] / 2.0 for s in HEMISPHERES}
    j_seg = float(sum(min(d[-1][s], d[+1][s]) for s in HEMISPHERES))

    j_cross = float(np.mean([crossing_indicator(arch, e, r) for e in EYES for r in RETINAL_SIDES]))
    j_sym = float(np.mean([int(arch[(-e, r)] != -arch[(e, r)]) for e in EYES for r in RETINAL_SIDES]))

    left_field_pair = clusters[-1][0] == clusters[-1][1]
    right_field_pair = clusters[+1][0] == clusters[+1][1]
    paired = bool(left_field_pair and right_field_pair)
    separated = bool(paired and clusters[-1][0] != clusters[+1][0])

    return {
        "J_eye": j_eye,
        "J_pair": j_pair,
        "J_seg": j_seg,
        "J_cross": j_cross,
        "J_sym": j_sym,
        "pairing_constraint": paired,
        "hemifield_separation_constraint": separated,
        "hemifield_consistent": bool(j_pair <= EPS_TIE and j_seg <= EPS_TIE),
        "left_field_cluster_hemi": clusters[-1][0] if paired else np.nan,
        "right_field_cluster_hemi": clusters[+1][0] if paired else np.nan,
    }


def build_component_table(architectures: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for row in architectures:
        arch = row["arch"]
        assert isinstance(arch, dict)
        comp = exact_components_for_geometry(arch)
        out = {k: v for k, v in row.items() if k != "arch"}
        out.update(comp)
        out["analytic_partial_solution"] = bool(comp["hemifield_consistent"] and row["partial_symmetric"])
        rows.append(out)
    return pd.DataFrame(rows)


def pareto_frontier(df: pd.DataFrame, cost_cols: Sequence[str]) -> pd.DataFrame:
    costs = df[list(cost_cols)].to_numpy(dtype=float)
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    dominance_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            j_dominates_i = np.all(costs[j] <= costs[i] + EPS_TIE) and np.any(costs[j] < costs[i] - EPS_TIE)
            if j_dominates_i:
                dominated[i] = True
                dominance_count[i] += 1
                dominated_by[i].append(int(df.iloc[j]["arch_id"]))

    out = df.copy()
    out["pareto_nondominated"] = ~dominated
    out["n_architectures_dominating_this"] = dominance_count
    out["dominated_by_arch_ids"] = [";".join(map(str, x)) for x in dominated_by]
    return out


def weighted_cost(row: pd.Series, theta: float, lambda_seg: float, lambda_cross: float, lambda_sym: float) -> float:
    return float(
        (1.0 - theta) * row["J_eye"]
        + theta * (LAMBDA_PAIR * row["J_pair"] + lambda_seg * row["J_seg"])
        + lambda_cross * row["J_cross"]
        + lambda_sym * row["J_sym"]
    )


def weight_grid() -> List[Tuple[float, float, float]]:
    return list(itertools.product(LAMBDA_SEG_VALUES, LAMBDA_CROSS_VALUES, LAMBDA_SYM_VALUES))


def evaluate_weighted_grid(component_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    arch_rows = list(component_df.to_dict("records"))
    weights = weight_grid()

    for theta in THETA_VALUES:
        for lambda_seg, lambda_cross, lambda_sym in weights:
            costs = np.array([
                (1.0 - theta) * r["J_eye"]
                + theta * (LAMBDA_PAIR * r["J_pair"] + lambda_seg * r["J_seg"])
                + lambda_cross * r["J_cross"]
                + lambda_sym * r["J_sym"]
                for r in arch_rows
            ], dtype=float)
            min_cost = float(np.min(costs))
            tie_idx = np.where(costs <= min_cost + EPS_TIE)[0]
            tied = [arch_rows[i] for i in tie_idx]
            quotient_classes = sorted(set(str(t["quotient_code"]) for t in tied))
            partial_qclasses = sorted(set(str(t["quotient_code"]) for t in tied if bool(t["partial_symmetric"])))
            hemi_qclasses = sorted(set(str(t["quotient_code"]) for t in tied if bool(t["hemifield_consistent"])))
            unique = len(tied) == 1

            rows.append({
                "theta": theta,
                "lambda_pair": LAMBDA_PAIR,
                "lambda_seg": lambda_seg,
                "lambda_cross": lambda_cross,
                "lambda_sym": lambda_sym,
                "min_J_total": min_cost,
                "n_tied_architectures": len(tied),
                "n_tied_quotient_classes": len(quotient_classes),
                "tied_arch_ids": ";".join(str(t["arch_id"]) for t in tied),
                "tied_arch_codes": ";".join(str(t["arch_code"]) for t in tied),
                "tied_quotient_codes": ";".join(quotient_classes),
                "tied_canonical_labels": ";".join(str(t["canonical_label"]) for t in tied),
                "partial_symmetric_in_tie": any(bool(t["partial_symmetric"]) for t in tied),
                "unique_partial_winner": bool(unique and bool(tied[0]["partial_symmetric"])),
                "hemifield_consistent_in_tie": any(bool(t["hemifield_consistent"]) for t in tied),
                "unique_hemifield_consistent_winner": bool(unique and bool(tied[0]["hemifield_consistent"])),
                "partial_quotient_in_tie": len(partial_qclasses) > 0,
                "unique_partial_quotient_winner": bool(len(quotient_classes) == 1 and len(partial_qclasses) == 1),
                "hemifield_quotient_in_tie": len(hemi_qclasses) > 0,
                "unique_hemifield_quotient_winner": bool(len(quotient_classes) == 1 and len(hemi_qclasses) == 1),
                "unique_winner_arch_id": tied[0]["arch_id"] if unique else "",
                "unique_winner_label": tied[0]["canonical_label"] if unique else "",
            })
    return pd.DataFrame(rows)


def selection_frequency_by_theta(best_df: pd.DataFrame) -> pd.DataFrame:
    return best_df.groupby("theta", as_index=False).agg(
        n_parameter_cells=("theta", "size"),
        partial_symmetric_in_tie_freq=("partial_symmetric_in_tie", "mean"),
        unique_partial_winner_freq=("unique_partial_winner", "mean"),
        partial_quotient_in_tie_freq=("partial_quotient_in_tie", "mean"),
        unique_partial_quotient_winner_freq=("unique_partial_quotient_winner", "mean"),
        hemifield_consistent_in_tie_freq=("hemifield_consistent_in_tie", "mean"),
        unique_hemifield_consistent_winner_freq=("unique_hemifield_consistent_winner", "mean"),
        hemifield_quotient_in_tie_freq=("hemifield_quotient_in_tie", "mean"),
        unique_hemifield_quotient_winner_freq=("unique_hemifield_quotient_winner", "mean"),
        mean_n_tied_architectures=("n_tied_architectures", "mean"),
        mean_n_tied_quotient_classes=("n_tied_quotient_classes", "mean"),
        median_min_J_total=("min_J_total", "median"),
    )


def estimate_theta_c(theta_values: Sequence[float], freq_values: Sequence[float], target: float = 0.5) -> float:
    theta = np.asarray(theta_values, dtype=float)
    freq = np.asarray(freq_values, dtype=float)
    exact = np.where(np.isclose(freq, target, atol=1e-15))[0]
    if len(exact) > 0:
        return float(theta[exact[0]])
    for i in range(len(theta) - 1):
        f0 = freq[i]
        f1 = freq[i + 1]
        if (f0 - target) * (f1 - target) < 0:
            return float(theta[i] + (target - f0) / (f1 - f0) * (theta[i + 1] - theta[i]))
    return float("nan")


def transition_summary(freq_df: pd.DataFrame, analysis_label: str) -> pd.DataFrame:
    outcomes = [c for c in freq_df.columns if c.endswith("_freq") and c != "n_parameter_cells"]
    rows = []
    for out in outcomes:
        rows.append({
            "analysis": analysis_label,
            "outcome": out,
            "theta_c_linear_interpolation": estimate_theta_c(freq_df["theta"].values, freq_df[out].values),
            "frequency_at_theta_0": float(freq_df.loc[freq_df["theta"] == 0.0, out].iloc[0]),
            "frequency_at_theta_1": float(freq_df.loc[freq_df["theta"] == 1.0, out].iloc[0]),
            "min_frequency": float(freq_df[out].min()),
            "max_frequency": float(freq_df[out].max()),
        })
    return pd.DataFrame(rows)


def weight_sensitivity_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in best_df.groupby(["lambda_seg", "lambda_cross", "lambda_sym"]):
        lambda_seg, lambda_cross, lambda_sym = keys
        freq = selection_frequency_by_theta(sub)
        rows.append({
            "lambda_seg": lambda_seg,
            "lambda_cross": lambda_cross,
            "lambda_sym": lambda_sym,
            "theta_c_unique_partial_quotient": estimate_theta_c(freq["theta"], freq["unique_partial_quotient_winner_freq"]),
            "theta_c_partial_in_tie": estimate_theta_c(freq["theta"], freq["partial_symmetric_in_tie_freq"]),
            "high_theta_unique_partial_quotient_freq": float(sub[sub["theta"] >= 0.75]["unique_partial_quotient_winner"].mean()),
            "high_theta_partial_in_tie_freq": float(sub[sub["theta"] >= 0.75]["partial_symmetric_in_tie"].mean()),
            "low_theta_unique_partial_quotient_freq": float(sub[sub["theta"] <= 0.25]["unique_partial_quotient_winner"].mean()),
            "mean_tied_architectures": float(sub["n_tied_architectures"].mean()),
            "mean_tied_quotient_classes": float(sub["n_tied_quotient_classes"].mean()),
        })
    return pd.DataFrame(rows)


def geometry_dependent_components_with_noise(arch: Dict[Tuple[int, int], int], b: float) -> Dict[str, float | bool]:
    x_values = make_x_grid()
    zetas = (0.0,) if b == 0.0 else (-float(b), 0.0, float(b))

    pair_vals = []
    seg_vals = []
    for zeta in zetas:
        pair_mismatch = []
        distributions = {
            -1: {LEFT_HEMI: 0, RIGHT_HEMI: 0, "n": 0},
            +1: {LEFT_HEMI: 0, RIGHT_HEMI: 0, "n": 0},
        }
        for x in x_values:
            h = h_from_x(float(x))
            r_left = r_from_x_with_noise(LEFT_EYE, float(x), zeta)
            r_right = r_from_x_with_noise(RIGHT_EYE, float(x), zeta)
            s_left = arch[(LEFT_EYE, r_left)]
            s_right = arch[(RIGHT_EYE, r_right)]
            pair_mismatch.append(int(s_left != s_right))
            for e, r in [(LEFT_EYE, r_left), (RIGHT_EYE, r_right)]:
                s = arch[(e, r)]
                distributions[h][s] += 1
                distributions[h]["n"] += 1
        d_left = {s: distributions[-1][s] / distributions[-1]["n"] for s in HEMISPHERES}
        d_right = {s: distributions[+1][s] / distributions[+1]["n"] for s in HEMISPHERES}
        pair_vals.append(float(np.mean(pair_mismatch)))
        seg_vals.append(float(sum(min(d_left[s], d_right[s]) for s in HEMISPHERES)))

    intr = exact_components_for_geometry(arch)
    return {
        "J_eye": float(intr["J_eye"]),
        "J_pair": float(np.mean(pair_vals)),
        "J_seg": float(np.mean(seg_vals)),
        "J_cross": float(intr["J_cross"]),
        "J_sym": float(intr["J_sym"]),
        "hemifield_consistent": bool(np.mean(pair_vals) <= EPS_TIE and np.mean(seg_vals) <= EPS_TIE),
    }


def component_table_with_boundary_noise(architectures: List[Dict[str, object]], b: float) -> pd.DataFrame:
    rows = []
    for row in architectures:
        arch = row["arch"]
        assert isinstance(arch, dict)
        comp = geometry_dependent_components_with_noise(arch, b)
        out = {k: v for k, v in row.items() if k != "arch"}
        out.update(comp)
        rows.append(out)
    return pd.DataFrame(rows)


def boundary_noise_summary(architectures: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for b in BOUNDARY_NOISE_VALUES:
        print(f"[boundary noise] b={b}")
        df_b = component_table_with_boundary_noise(architectures, b)
        best_b = evaluate_weighted_grid(df_b)
        freq_b = selection_frequency_by_theta(best_b)
        for _, r in freq_b.iterrows():
            row = r.to_dict()
            row["boundary_noise_b"] = b
            rows.append(row)

        # Continuous component separation: best partial versus best non-partial on J_pair + J_seg.
        df_b["J_pair_plus_seg"] = df_b["J_pair"] + df_b["J_seg"]
        part = df_b[df_b["partial_symmetric"]]
        nonpart = df_b[~df_b["partial_symmetric"]]
        rows.append({
            "boundary_noise_b": b,
            "theta": np.nan,
            "n_parameter_cells": np.nan,
            "partial_symmetric_in_tie_freq": np.nan,
            "unique_partial_winner_freq": np.nan,
            "partial_quotient_in_tie_freq": np.nan,
            "unique_partial_quotient_winner_freq": np.nan,
            "hemifield_consistent_in_tie_freq": np.nan,
            "unique_hemifield_consistent_winner_freq": np.nan,
            "hemifield_quotient_in_tie_freq": np.nan,
            "unique_hemifield_quotient_winner_freq": np.nan,
            "mean_n_tied_architectures": np.nan,
            "mean_n_tied_quotient_classes": np.nan,
            "median_min_J_total": np.nan,
            "diagnostic_min_partial_J_pair_plus_seg": float(part["J_pair_plus_seg"].min()),
            "diagnostic_min_nonpartial_J_pair_plus_seg": float(nonpart["J_pair_plus_seg"].min()),
            "diagnostic_partial_advantage": float(nonpart["J_pair_plus_seg"].min() - part["J_pair_plus_seg"].min()),
        })
    return pd.DataFrame(rows)


def bootstrap_from_best_df(best_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    cell_cols = ["lambda_seg", "lambda_cross", "lambda_sym"]
    # Matrix: theta x parameter cell.
    pivot = best_df.pivot_table(index="theta", columns=cell_cols, values=outcome, aggfunc="first")
    matrix = pivot.to_numpy(dtype=int)
    n_cells = matrix.shape[1]

    patterns, counts = np.unique(matrix.T, axis=0, return_counts=True)
    probs = counts / counts.sum()

    theta_cs = []
    f0s = []
    f1s = []
    for _ in range(B_BOOT):
        sampled_counts = rng.multinomial(n_cells, probs)
        freq = sampled_counts @ patterns / n_cells
        theta_cs.append(estimate_theta_c(pivot.index.values, freq))
        f0s.append(float(freq[0]))
        f1s.append(float(freq[-1]))

    arr = np.array(theta_cs, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        med = lo = hi = float("nan")
        valid_fraction = 0.0
    else:
        med = float(np.median(valid))
        lo = float(np.quantile(valid, 0.025))
        hi = float(np.quantile(valid, 0.975))
        valid_fraction = float(len(valid) / len(arr))

    return pd.DataFrame([{
        "analysis": "weighted_parameter_cell_bootstrap",
        "outcome": outcome,
        "B": B_BOOT,
        "n_cells": n_cells,
        "n_unique_selection_patterns": int(len(patterns)),
        "theta_c_median": med,
        "theta_c_95ci_low": lo,
        "theta_c_95ci_high": hi,
        "theta_c_valid_fraction": valid_fraction,
        "frequency_at_theta_0_median": float(np.median(f0s)),
        "frequency_at_theta_1_median": float(np.median(f1s)),
    }])


def run_bootstrap(best_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "partial_symmetric_in_tie",
        "unique_partial_winner",
        "partial_quotient_in_tie",
        "unique_partial_quotient_winner",
        "hemifield_consistent_in_tie",
        "unique_hemifield_consistent_winner",
        "hemifield_quotient_in_tie",
        "unique_hemifield_quotient_winner",
    ]
    return pd.concat([bootstrap_from_best_df(best_df, o) for o in outcomes], ignore_index=True)


def prob_components(p_n: float, p_t: float) -> Dict[str, float]:
    # Exact expected costs for the probabilistic symmetric routing model.
    j_eye = p_n + p_t - 2.0 * p_n * p_t
    j_pair = 1.0 - p_n - p_t + 2.0 * p_n * p_t
    j_seg = 1.0 - abs(p_n - p_t)
    j_cross = 0.5 * (p_n + p_t)
    return {
        "J_eye": float(j_eye),
        "J_pair": float(j_pair),
        "J_seg": float(j_seg),
        "J_cross": float(j_cross),
        "J_sym": 0.0,
    }


def prob_objective(p_n: float, p_t: float, theta: float, lambda_seg: float, lambda_cross: float) -> float:
    c = prob_components(p_n, p_t)
    return float(
        (1.0 - theta) * c["J_eye"]
        + theta * (LAMBDA_PAIR * c["J_pair"] + lambda_seg * c["J_seg"])
        + lambda_cross * c["J_cross"]
    )


def candidate_prob_points(theta: float, lambda_seg: float, lambda_cross: float) -> List[Tuple[float, float, str]]:
    pts: List[Tuple[float, float, str]] = []
    for u, v in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        pts.append((float(u), float(v), "corner"))

    # Diagonal p_N = p_T = t. Objective on diagonal is quadratic.
    # J_eye = 2t - 2t^2, J_pair = 1 - 2t + 2t^2, J_seg = 1, J_cross = t.
    a2 = 2.0 * ((theta) - (1.0 - theta))  # coefficient on t^2 = 4theta - 2
    a1 = 2.0 * (1.0 - theta) - 2.0 * theta + lambda_cross
    if abs(a2) > 1e-15:
        t_star = -a1 / (2.0 * a2)
        if 0.0 <= t_star <= 1.0:
            pts.append((float(t_star), float(t_star), "diagonal_stationary"))
    # Always include midpoint on the kink line.
    pts.append((0.5, 0.5, "diagonal_midpoint"))

    # Interior stationary points in the two piecewise regions.
    A = 1.0 - 2.0 * theta + 0.5 * lambda_cross
    B = 4.0 * theta - 2.0
    L = lambda_seg
    if abs(B) > 1e-15:
        # Region u >= v, |u-v| = u-v.
        v = (theta * L - A) / B
        u = (-theta * L - A) / B
        if 0.0 <= v <= u <= 1.0:
            pts.append((float(u), float(v), "interior_u_ge_v"))
        # Region v >= u, |u-v| = v-u.
        u = (theta * L - A) / B
        v = (-theta * L - A) / B
        if 0.0 <= u <= v <= 1.0:
            pts.append((float(u), float(v), "interior_v_ge_u"))

    # Deduplicate.
    seen = set()
    clean = []
    for u, v, label in pts:
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        key = (round(u, 15), round(v, 15), label)
        if key not in seen:
            seen.add(key)
            clean.append((u, v, label))
    return clean


def optimize_probabilistic_routing() -> pd.DataFrame:
    rows = []
    for theta in THETA_VALUES:
        for lambda_seg in LAMBDA_SEG_VALUES:
            for lambda_cross in LAMBDA_CROSS_VALUES:
                pts = candidate_prob_points(float(theta), float(lambda_seg), float(lambda_cross))
                evals = []
                for p_n, p_t, source in pts:
                    j = prob_objective(p_n, p_t, float(theta), float(lambda_seg), float(lambda_cross))
                    evals.append((j, p_n, p_t, source))
                min_j = min(x[0] for x in evals)
                tied = [x for x in evals if x[0] <= min_j + EPS_TIE]
                for j, p_n, p_t, source in tied:
                    c = prob_components(p_n, p_t)
                    rows.append({
                        "theta": theta,
                        "lambda_seg": lambda_seg,
                        "lambda_cross": lambda_cross,
                        "J_total": j,
                        "p_nasal_crossing": p_n,
                        "p_temporal_crossing": p_t,
                        "candidate_source": source,
                        "n_tied_continuous_minima_in_candidate_set": len(tied),
                        "partial_probability_contrast_abs": abs(p_n - p_t),
                        "partial_extreme_solution": bool(abs(abs(p_n - p_t) - 1.0) <= 1e-10),
                        **c,
                    })
    return pd.DataFrame(rows)


def null_objective_winners(component_df: pd.DataFrame) -> pd.DataFrame:
    objectives = {
        "eye_only": ["J_eye"],
        "pair_only": ["J_pair"],
        "segment_only": ["J_seg"],
        "pair_plus_segment": ["J_pair", "J_seg"],
        "pair_plus_segment_plus_symmetry": ["J_pair", "J_seg", "J_sym"],
        "pair_plus_segment_plus_cross": ["J_pair", "J_seg", "J_cross"],
    }
    rows = []
    for name, cols in objectives.items():
        vals = component_df[cols].sum(axis=1)
        min_val = float(vals.min())
        tied = component_df[vals <= min_val + EPS_TIE]
        rows.append({
            "null_or_reduced_objective": name,
            "cost_columns": "+".join(cols),
            "min_value": min_val,
            "n_tied_architectures": int(len(tied)),
            "tied_arch_ids": ";".join(map(str, tied["arch_id"].tolist())),
            "tied_canonical_labels": ";".join(tied["canonical_label"].astype(str).tolist()),
            "partial_symmetric_in_tie": bool(tied["partial_symmetric"].any()),
            "hemifield_consistent_in_tie": bool(tied["hemifield_consistent"].any()),
            "n_tied_quotient_classes": int(tied["quotient_code"].nunique()),
            "unique_partial_quotient": bool(tied["quotient_code"].nunique() == 1 and tied["partial_symmetric"].any()),
        })
    return pd.DataFrame(rows)


def retinal_geometry_null_enumeration(architectures: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    geom_keys = [(LEFT_EYE, -1), (LEFT_EYE, +1), (RIGHT_EYE, -1), (RIGHT_EYE, +1)]
    bio_map = {(e, h): biological_retinal_map(e, h) for e in EYES for h in HEMIFIELDS}
    for geom_id, outputs in enumerate(itertools.product(RETINAL_SIDES, repeat=4)):
        rmap = {key: int(r) for key, r in zip(geom_keys, outputs)}
        is_biological = all(rmap[k] == bio_map[k] for k in geom_keys)
        rows_arch = []
        for row in architectures:
            arch = row["arch"]
            assert isinstance(arch, dict)
            comp = exact_components_for_geometry(arch, retinal_map=rmap)
            out = {k: v for k, v in row.items() if k != "arch"}
            out.update(comp)
            rows_arch.append(out)
        df = pd.DataFrame(rows_arch)
        vals = df["J_pair"] + df["J_seg"]
        min_val = float(vals.min())
        tied = df[vals <= min_val + EPS_TIE]
        rows.append({
            "geometry_id": geom_id,
            "r_L_leftfield": rmap[(LEFT_EYE, -1)],
            "r_L_rightfield": rmap[(LEFT_EYE, +1)],
            "r_R_leftfield": rmap[(RIGHT_EYE, -1)],
            "r_R_rightfield": rmap[(RIGHT_EYE, +1)],
            "is_biological_geometry": bool(is_biological),
            "min_J_pair_plus_seg": min_val,
            "n_tied_architectures": int(len(tied)),
            "n_tied_quotient_classes": int(tied["quotient_code"].nunique()),
            "partial_symmetric_in_tie": bool(tied["partial_symmetric"].any()),
            "hemifield_consistent_in_tie": bool(tied["hemifield_consistent"].any()),
            "unique_partial_quotient": bool(tied["quotient_code"].nunique() == 1 and tied["partial_symmetric"].any()),
            "tied_canonical_labels": ";".join(tied["canonical_label"].astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def orientation_secondary(architectures: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for row in architectures:
        arch = row["arch"]
        assert isinstance(arch, dict)
        mismatches = []
        for h in HEMIFIELDS:
            target = -h
            for e in EYES:
                r = biological_retinal_map(e, h)
                mismatches.append(int(arch[(e, r)] != target))
        out = {k: v for k, v in row.items() if k != "arch"}
        out["J_orient_secondary"] = float(np.mean(mismatches))
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["J_orient_secondary", "arch_id"]).reset_index(drop=True)


def write_readme() -> None:
    text = f"""Improved optic-chiasm geometric routing analysis

Output directory:
{OUTDIR}

Key files to inspect first:
1. improved_analytic_constraint_results.csv
   Direct constraint result. This is the primary non-weighted analysis.

2. improved_z2_quotient_classes.csv
   Shows mirror-related equivalence classes under global hemisphere relabeling.

3. improved_pareto_frontier.csv
   Cost-vector Pareto analysis before weighted objectives.

4. improved_weighted_transition_summary_with_bootstrap.csv
   Weighted-objective verification and bootstrap transition summaries.

5. improved_boundary_noise_summary.csv
   Boundary-noise robustness using continuous costs and selection outcomes.

6. improved_probabilistic_continuous_optima.csv
   Exact candidate-set continuous optimization over p_N and p_T.

7. improved_null_objective_winners.csv and improved_retinal_geometry_nulls.csv
   Null and reduced-objective checks.

8. improved_orientation_secondary_summary.csv
   Secondary biological orientation analysis. This is not part of the primary objective.
"""
    (OUTDIR / "README_improved_outputs.txt").write_text(text, encoding="utf-8")


def save_plots(component_df: pd.DataFrame, freq_df: pd.DataFrame, prob_df: pd.DataFrame, null_geom_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(component_df["J_pair"], component_df["J_seg"])
    for _, r in component_df.iterrows():
        plt.text(r["J_pair"] + 0.01, r["J_seg"] + 0.01, str(int(r["arch_id"])), fontsize=8)
    plt.xlabel("J_pair")
    plt.ylabel("J_seg")
    plt.title("Exact architecture costs before weighting")
    plt.tight_layout()
    plt.savefig(OUTDIR / "improved_fig1_exact_architecture_costs.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(freq_df["theta"], freq_df["partial_symmetric_in_tie_freq"], label="Partial in tie")
    plt.plot(freq_df["theta"], freq_df["unique_partial_quotient_winner_freq"], label="Unique partial quotient")
    plt.plot(freq_df["theta"], freq_df["hemifield_consistent_in_tie_freq"], label="Hemifield-consistent in tie")
    plt.xlabel("theta")
    plt.ylabel("Selection frequency")
    plt.ylim(-0.02, 1.02)
    plt.title("Weighted-objective verification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "improved_fig2_weighted_selection_transition.png", dpi=300)
    plt.close()

    # Probabilistic optima at baseline lambda_seg=1, lambda_cross=0.1.
    sub = prob_df[(prob_df["lambda_seg"] == 1.0) & (prob_df["lambda_cross"] == 0.1)].copy()
    # One representative optimum per theta.
    sub = sub.sort_values(["theta", "J_total", "p_nasal_crossing", "p_temporal_crossing"]).groupby("theta", as_index=False).first()
    plt.figure(figsize=(8, 5))
    plt.plot(sub["theta"], sub["p_nasal_crossing"], label="p_N")
    plt.plot(sub["theta"], sub["p_temporal_crossing"], label="p_T")
    plt.xlabel("theta")
    plt.ylabel("Optimal crossing probability")
    plt.ylim(-0.02, 1.02)
    plt.title("Continuous probabilistic-routing optima")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "improved_fig3_probabilistic_optima.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    colors = [1 if x else 0 for x in null_geom_df["is_biological_geometry"]]
    plt.scatter(null_geom_df["geometry_id"], null_geom_df["min_J_pair_plus_seg"], c=colors)
    plt.xlabel("Retinal-geometry null ID")
    plt.ylabel("Minimum J_pair + J_seg")
    plt.title("Retinal-geometry null enumeration")
    plt.tight_layout()
    plt.savefig(OUTDIR / "improved_fig4_retinal_geometry_nulls.png", dpi=300)
    plt.close()


def main() -> None:
    print(f"Output directory: {OUTDIR}")
    print("[1/11] Building architectures")
    architectures = build_architectures()
    arch_defs = pd.DataFrame([{k: v for k, v in a.items() if k != "arch"} for a in architectures])
    arch_defs.to_csv(OUTDIR / "improved_architecture_definitions.csv", index=False)

    print("[2/11] Exact analytical constraint analysis")
    component_df = build_component_table(architectures)
    component_df.to_csv(OUTDIR / "improved_analytic_constraint_results.csv", index=False)

    quotient_df = component_df.groupby("quotient_code", as_index=False).agg(
        arch_ids=("arch_id", lambda x: ";".join(map(str, x))),
        arch_codes=("arch_code", lambda x: ";".join(map(str, x))),
        canonical_labels=("canonical_label", lambda x: ";".join(map(str, x))),
        any_partial_symmetric=("partial_symmetric", "any"),
        any_hemifield_consistent=("hemifield_consistent", "any"),
        min_J_pair=("J_pair", "min"),
        min_J_seg=("J_seg", "min"),
        min_J_eye=("J_eye", "min"),
        min_J_cross=("J_cross", "min"),
        min_J_sym=("J_sym", "min"),
    )
    quotient_df.to_csv(OUTDIR / "improved_z2_quotient_classes.csv", index=False)

    exact_summary = pd.DataFrame([
        {
            "n_architectures": len(component_df),
            "n_hemifield_consistent_architectures": int(component_df["hemifield_consistent"].sum()),
            "hemifield_consistent_arch_ids": ";".join(map(str, component_df.loc[component_df["hemifield_consistent"], "arch_id"])),
            "hemifield_consistent_labels": ";".join(component_df.loc[component_df["hemifield_consistent"], "canonical_label"].astype(str)),
            "n_hemifield_consistent_quotient_classes": int(component_df.loc[component_df["hemifield_consistent"], "quotient_code"].nunique()),
            "n_partial_symmetric_architectures": int(component_df["partial_symmetric"].sum()),
            "n_partial_symmetric_quotient_classes": int(component_df.loc[component_df["partial_symmetric"], "quotient_code"].nunique()),
        }
    ])
    exact_summary.to_csv(OUTDIR / "improved_exact_constraint_summary.csv", index=False)

    print("[3/11] Pareto analysis")
    pareto_df = pareto_frontier(component_df, ["J_eye", "J_pair", "J_seg", "J_cross", "J_sym"])
    pareto_df.to_csv(OUTDIR / "improved_pareto_frontier.csv", index=False)

    print("[4/11] Weighted objective verification")
    best_df = evaluate_weighted_grid(component_df)
    best_df.to_csv(OUTDIR / "improved_weighted_best_by_condition.csv", index=False)
    freq_df = selection_frequency_by_theta(best_df)
    freq_df.to_csv(OUTDIR / "improved_weighted_selection_frequency_by_theta.csv", index=False)
    trans_df = transition_summary(freq_df, "weighted_objective_verification")
    trans_df.to_csv(OUTDIR / "improved_weighted_transition_summary.csv", index=False)
    weight_df = weight_sensitivity_summary(best_df)
    weight_df.to_csv(OUTDIR / "improved_weight_sensitivity_summary.csv", index=False)

    print("[5/11] Bootstrap over weighted parameter cells")
    boot_df = run_bootstrap(best_df)
    boot_df.to_csv(OUTDIR / "improved_weighted_bootstrap_transition_summary.csv", index=False)
    pd.concat([trans_df, boot_df], ignore_index=True, sort=False).to_csv(
        OUTDIR / "improved_weighted_transition_summary_with_bootstrap.csv", index=False
    )

    print("[6/11] Boundary-noise robustness")
    bnoise_df = boundary_noise_summary(architectures)
    bnoise_df.to_csv(OUTDIR / "improved_boundary_noise_summary.csv", index=False)

    print("[7/11] Continuous probabilistic-routing optimization")
    prob_df = optimize_probabilistic_routing()
    prob_df.to_csv(OUTDIR / "improved_probabilistic_continuous_optima.csv", index=False)

    print("[8/11] Null and reduced-objective analyses")
    null_df = null_objective_winners(component_df)
    null_df.to_csv(OUTDIR / "improved_null_objective_winners.csv", index=False)
    null_geom_df = retinal_geometry_null_enumeration(architectures)
    null_geom_df.to_csv(OUTDIR / "improved_retinal_geometry_nulls.csv", index=False)

    print("[9/11] Secondary orientation-resolved analysis")
    orient_df = orientation_secondary(architectures)
    orient_df.to_csv(OUTDIR / "improved_orientation_secondary_summary.csv", index=False)

    print("[10/11] Saving plots")
    save_plots(component_df, freq_df, prob_df, null_geom_df)

    print("[11/11] Writing README")
    write_readme()

    print("Done.")
    print(f"All outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
