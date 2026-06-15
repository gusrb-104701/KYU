"""
Octiv torque CSV -> 15T-baseline PCA/MSPC -> 2x2 master diagnostic dashboard (Plotly).
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

METADATA_ROWS = 7
LAST_N_ROWS = 20
N_FEATURES = 192
N_COMPONENTS = 3
NORMAL_CONDITION = "15 Nm"
FAULT_CONDITION = "1 Nm"
UCL_SIGMA = 3

TORQUE_GROUPS: dict[str, list[str]] = {
    "1 Nm": ["1torque_6.csv", "1torque_7.csv", "1torque_8.csv"],
    "5 Nm": ["5torque_7.csv", "5torque_8.csv", "5torque_9.csv"],
    "10 Nm": ["10torque_9.csv", "10torque_10.csv", "10torque_11.csv"],
    "15 Nm": ["15torque_9.csv", "15torque_10.csv", "15torque_11.csv"],
}

TORQUE_ORDER = ["15 Nm", "10 Nm", "5 Nm", "1 Nm"]
FAULT_CONDITIONS = ["1 Nm", "5 Nm", "10 Nm"]
# Contribution panels left→right: mild fault → severe (matches T²/Q control chart)
FAULT_PANEL_TORQUES = [10, 5, 1]
DASHBOARD_COLS = 3
OUTPUT_HTML = "dashboard.html"
SCALER_PATH = "baseline_scaler.pkl"

TORQUE_COLORS = {
    "1 Nm": "red",
    "5 Nm": "#2ca02c",
    "10 Nm": "#ff7f0e",
    "15 Nm": "blue",
}
MARKER_SIZE = 1.8
TOP_CONTRIB_VARS = 15
TOP_PC1_VARS = 15
TOP_CONDITION_CONTRIB = 10

# Global sample_order indices to exclude (e.g. 112 = 10 Nm outlier)
OUTLIER_SAMPLE_ORDERS: list[int] = []


def _is_data_row(first_cell: object) -> bool:
    try:
        float(str(first_cell).strip())
        return True
    except (TypeError, ValueError):
        return False


def load_last_stabilization_rows(csv_path: Path) -> pd.DataFrame:
    """Skip 7 metadata lines, then keep the last 20 *pure numeric* rows only."""
    raw = pd.read_csv(
        csv_path,
        skiprows=METADATA_ROWS,
        header=0,
        low_memory=False,
        encoding="latin-1",
    )
    raw.columns = [str(c).strip() for c in raw.columns]
    time_col = raw.columns[0]

    # Some exports append non-numeric "header" or text rows at the end.
    # Keep only rows where *all* columns are numeric after coercion.
    time_numeric = pd.to_numeric(raw[time_col], errors="coerce")
    feat_numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    mask = time_numeric.notna() & feat_numeric.notna().all(axis=1)

    numeric = pd.concat([time_numeric, feat_numeric], axis=1).loc[mask].copy()
    numeric.columns = [time_col] + list(raw.columns[1:])

    if len(numeric) < LAST_N_ROWS:
        raise ValueError(
            f"{csv_path.name}: only {len(numeric)} pure numeric rows found "
            f"(need at least {LAST_N_ROWS})"
        )

    # Take the last N numeric rows (end of file in numeric-only sense)
    return numeric.tail(LAST_N_ROWS).reset_index(drop=True)


def select_valid_feature_columns(columns: pd.Index) -> list[str]:
    """Octiv export: 192 RF variables (F1/F2, excluding freq/pulse & derived power metrics)."""
    global_drop = {
        "Frequency 1 MHz",
        "Frequency 2 MHz",
        "Pulse Period us",
        "Pulse On Time us",
        "Pulse Frequency Hz",
        "Pulse Duty Cycle %",
    }
    fundamental_drop = (
        "Forward Power Watts",
        "Reflected Power Watts",
        "Energy Joules",
        "Standing Wave Ratio",
        "Harmonic Phase Degrees",
    )
    selected: list[str] = []
    for col in columns:
        name = str(col).strip()
        if name in global_drop:
            continue
        is_harmonic = any(tag in col for tag in (" H2", " H3", " H4", " H5", " H6"))
        if not is_harmonic and any(token in col for token in fundamental_drop):
            continue
        selected.append(col)
    return selected


def average_condition_samples(
    base_dir: Path,
    filenames: list[str],
    torque_label: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Average the last-20-row segments across all 3 replicate files for a condition."""
    segments: list[pd.DataFrame] = []
    for filename in filenames:
        seg = load_last_stabilization_rows(base_dir / filename).copy()
        segments.append(seg)

    # Align columns and average numeric features
    numeric_frames = [s[feature_names].apply(pd.to_numeric, errors="coerce") for s in segments]
    mean_features = pd.concat(numeric_frames).groupby(level=0).mean()
    # Each replicate contributes 20 rows; average per-row position across 3 files
    stacked = np.stack([f.to_numpy() for f in numeric_frames], axis=0)  # (3, 20, F)
    avg_array = stacked.mean(axis=0)  # (20, F)

    avg_df = pd.DataFrame(avg_array, columns=feature_names)
    avg_df["condition"] = torque_label
    avg_df["source_file"] = f"avg({','.join(filenames)})"
    avg_df["row_in_file"] = np.arange(1, len(avg_df) + 1)

    # Recover time column from first segment (approximate)
    time_col = segments[0].columns[0]
    time_vals = segments[0][time_col].apply(pd.to_numeric, errors="coerce").to_numpy()
    avg_df.insert(0, time_col, time_vals[:len(avg_df)])
    avg_df["hover_label"] = avg_df.apply(
        lambda r: (
            f"{torque_label} | 3-file avg | "
            f"row {int(r['row_in_file'])} | time {float(r[time_col]):.3f} s"
        ),
        axis=1,
    )
    return avg_df


def load_all_samples(base_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build dataset: per-condition average of 3 files x 20 rows -> 1 averaged point per row position."""
    feature_names: list[str] | None = None

    for torque_label in TORQUE_ORDER:
        filenames = TORQUE_GROUPS[torque_label]
        for filename in filenames:
            path = base_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing CSV: {path}")

            cols = select_valid_feature_columns(
                load_last_stabilization_rows(path).columns[1:]
            )
            if feature_names is None:
                feature_names = cols
            elif cols != feature_names:
                raise ValueError(f"Feature mismatch in {filename}")

    assert feature_names is not None
    condition_frames: list[pd.DataFrame] = []
    for torque_label in TORQUE_ORDER:
        filenames = TORQUE_GROUPS[torque_label]
        block = average_condition_samples(base_dir, filenames, torque_label, feature_names)
        condition_frames.append(block)

    combined = pd.concat(condition_frames, ignore_index=True)
    combined["torque_labels"] = (
        combined["condition"].str.replace(" Nm", "", regex=False).astype(int)
    )
    combined["sample_order"] = np.arange(len(combined))
    combined = drop_outlier_samples(combined)
    combined["sample_order"] = np.arange(len(combined))
    return combined, feature_names


def drop_outlier_samples(combined: pd.DataFrame) -> pd.DataFrame:
    """Remove rows flagged as outliers by global sample_order index."""
    if not OUTLIER_SAMPLE_ORDERS:
        return combined

    mask = combined["sample_order"].isin(OUTLIER_SAMPLE_ORDERS)
    if not mask.any():
        return combined

    removed = combined.loc[mask]
    for _, row in removed.iterrows():
        print(
            "Removed outlier: "
            f"sample_order={int(row['sample_order'])} | "
            f"{row['condition']} | {row['source_file']} | "
            f"row_in_file={int(row['row_in_file'])}"
        )

    return combined.loc[~mask].reset_index(drop=True)


def _normal_sample_mask(combined: pd.DataFrame) -> np.ndarray:
    return (combined["condition"] == NORMAL_CONDITION).to_numpy()


def fit_baseline_scaler(
    X: pd.DataFrame,
    combined: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray]:
    """
    Z-score scaler: fit ONLY on 15 Nm raw samples (fit_transform on baseline).
    Returns the fitted scaler and scaled baseline matrix for PCA fitting.
    """
    normal_mask = _normal_sample_mask(combined)
    X_normal = X.loc[normal_mask].to_numpy()

    if len(X_normal) < N_COMPONENTS:
        raise ValueError(
            f"Need at least {N_COMPONENTS} normal samples; got {len(X_normal)}"
        )

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    return scaler, X_normal_scaled


def transform_all_samples(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply baseline Z-score scaler to full dataset (transform only, no refit)."""
    return scaler.transform(X.to_numpy())


def fit_baseline_pca(X_normal_scaled: np.ndarray) -> PCA:
    """PCA fit on Z-score scaled 15 Nm baseline samples only."""
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(X_normal_scaled)
    return pca


def save_baseline_scaler(scaler: StandardScaler, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(scaler, f)


def print_scaling_diagnostics(
    scaler: StandardScaler,
    X_scaled: np.ndarray,
    combined: pd.DataFrame,
) -> None:
    normal_mask = _normal_sample_mask(combined)
    X_normal_scaled = X_scaled[normal_mask]
    print(
        f"Z-score baseline ({NORMAL_CONDITION}, n={normal_mask.sum()}): "
        f"scaler fit on raw features only"
    )
    print(
        f"  Feature means (15 Nm scaled): min={X_normal_scaled.mean(axis=0).min():.3f}, "
        f"max={X_normal_scaled.mean(axis=0).max():.3f} "
        f"(expect ~0)"
    )
    print(
        f"  Feature stds  (15 Nm scaled): min={X_normal_scaled.std(axis=0).min():.3f}, "
        f"max={X_normal_scaled.std(axis=0).max():.3f} "
        f"(expect ~1)"
    )
    print(f"  Scaler saved: mean_/scale_ vectors length = {len(scaler.mean_)}")


def compute_mspc(
    X_scaled: np.ndarray,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Hotelling T² and Q (SPE) from PCA scores and reconstruction residuals."""
    scores = pca.transform(X_scaled)
    eigenvalues = pca.explained_variance_
    if np.any(eigenvalues <= 0):
        raise ValueError("PCA eigenvalues must be positive for T² calculation.")

    t2 = np.sum((scores**2) / eigenvalues, axis=1)

    x_hat = pca.inverse_transform(scores)
    residuals = X_scaled - x_hat
    q = np.sum(residuals**2, axis=1)

    return t2, q


def ucl_from_normal(values: np.ndarray, normal_mask: np.ndarray) -> float:
    ref = values[normal_mask]
    return float(ref.mean() + UCL_SIGMA * ref.std(ddof=1))


def _fault_pooled_mean(
    X_scaled: np.ndarray,
    combined: pd.DataFrame,
) -> np.ndarray:
    fault_mask = (combined["condition"] == FAULT_CONDITION).to_numpy()
    return X_scaled[fault_mask].mean(axis=0)


def _top_contribution_df(
    feature_names: list[str],
    contrib: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "variable": feature_names,
            "contribution": contrib,
            "abs_contribution": np.abs(contrib),
        }
    )
    return df.nlargest(TOP_CONTRIB_VARS, "abs_contribution").sort_values(
        "contribution", ascending=True
    )


def contribution_scores_q_1t(
    X_scaled: np.ndarray,
    combined: pd.DataFrame,
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """Q (SPE) variable contributions: squared reconstruction residuals."""
    x_fault_mean = _fault_pooled_mean(X_scaled, combined)

    scores = pca.transform(x_fault_mean.reshape(1, -1))
    x_hat = pca.inverse_transform(scores)[0]
    residual = x_fault_mean - x_hat
    return _top_contribution_df(feature_names, residual**2)


def pc1_loading_contributions(
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """PC1 loading contributions: absolute loadings of each variable on PC1."""
    loadings_pc1 = pca.components_[0]  # shape (n_features,)
    df = pd.DataFrame(
        {
            "variable": feature_names,
            "loading": loadings_pc1,
            "abs_loading": np.abs(loadings_pc1),
        }
    )
    return df.nlargest(TOP_PC1_VARS, "abs_loading").sort_values("loading", ascending=True)


def compute_t2_contribution_per_sample(scores: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Per-sample T² variable contribution (n_samples × n_features).
    c_ij = sum_k [ (t_ik * P[k,j]) / sqrt(lambda_k) ]^2
    """
    loadings = pca.components_
    sqrt_lam = np.sqrt(pca.explained_variance_)
    contrib = np.zeros((scores.shape[0], loadings.shape[1]), dtype=np.float64)

    for k, lam_sqrt in enumerate(sqrt_lam):
        term = (scores[:, k] / lam_sqrt)[:, np.newaxis] * loadings[k, :]
        contrib += term**2

    return contrib


def compute_q_contribution_per_sample(
    X_scaled_cond: np.ndarray,
    scores_cond: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """
    Per-sample Q (SPE) variable contribution (n_samples × n_features).
    e_ij^2 = (x_ij - x_hat_ij)^2
    """
    x_hat = pca.inverse_transform(scores_cond)
    residuals = X_scaled_cond - x_hat
    return residuals**2


def condition_mean_contributions(
    X_scaled: np.ndarray,
    scores: np.ndarray,
    combined: pd.DataFrame,
    pca: PCA,
    feature_names: list[str],
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Mean T² and Q variable contributions for one fault condition.
    Returns full contribution DataFrames plus mean T²/Q statistics for titles.
    """
    torque_val = int(condition.replace(" Nm", ""))
    mask = (combined["torque_labels"] == torque_val).to_numpy()
    if not mask.any():
        raise ValueError(f"No samples for torque_labels={torque_val} ({condition})")
    scores_c = scores[mask]
    X_c = X_scaled[mask]

    t2_contrib = compute_t2_contribution_per_sample(scores_c, pca).mean(axis=0)
    q_contrib = compute_q_contribution_per_sample(X_c, scores_c, pca).mean(axis=0)

    t2_df = pd.DataFrame({"variable": feature_names, "contribution": t2_contrib})
    q_df = pd.DataFrame({"variable": feature_names, "contribution": q_contrib})

    t2_vals, q_vals = compute_mspc(X_c, pca)
    t2_stat = float(t2_vals.mean())
    q_stat = float(q_vals.mean())

    return t2_df, q_df, t2_stat, q_stat


def top_contribution_bar_df(contrib_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-N contributors sorted descending (largest bar at top for horizontal bars)."""
    return (
        contrib_df.nlargest(top_n, "contribution")
        .sort_values("contribution", ascending=True)
        .reset_index(drop=True)
    )


def _condition_label(torque: int) -> str:
    """10 -> '10 Nm' condition key used in combined DataFrame."""
    return f"{torque} Nm"


def _torque_short_label(torque: int) -> str:
    """10 -> '10 torque' for subplot titles."""
    return f"{torque} torque"


def _grid_title_index(row: int, col: int, n_cols: int = DASHBOARD_COLS) -> int:
    """Map (row, col) in make_subplots grid to annotation index (0-based)."""
    return (row - 1) * n_cols + (col - 1)


def _set_subplot_title(
    fig: go.Figure,
    row: int,
    col: int,
    title: str,
    n_cols: int = DASHBOARD_COLS,
) -> None:
    idx = _grid_title_index(row, col, n_cols)
    if idx < len(fig.layout.annotations):
        fig.layout.annotations[idx].text = title


def validate_condition_grouping(
    combined: pd.DataFrame,
    t2_full: dict[str, pd.DataFrame],
    q_full: dict[str, pd.DataFrame],
    top_n: int = 5,
) -> None:
    """Print per-condition sample counts and top contributors to verify grouping."""
    print("\n=== Condition grouping validation (torque_labels) ===")
    for torque in FAULT_PANEL_TORQUES:
        condition = _condition_label(torque)
        mask = combined["torque_labels"] == torque
        labels = combined.loc[mask, "torque_labels"]
        print(
            f"\n{condition}: n_samples={int(mask.sum())}, "
            f"torque_labels={sorted(labels.unique().tolist())}"
        )

        t2_top = t2_full[condition].nlargest(top_n, "contribution")
        print(f"  T² top {top_n}:")
        for rank, row in enumerate(t2_top.itertuples(), start=1):
            print(f"    {rank}. {row.variable}  ({row.contribution:.4f})")

        q_top = q_full[condition].nlargest(top_n, "contribution")
        print(f"  Q top {top_n}:")
        for rank, row in enumerate(q_top.itertuples(), start=1):
            print(f"    {rank}. {row.variable}  ({row.contribution:.4f})")

    # Flag if contribution vectors are accidentally identical across conditions
    keys = FAULT_CONDITIONS
    t2_vectors = [t2_full[k].set_index("variable")["contribution"].values for k in keys]
    q_vectors = [q_full[k].set_index("variable")["contribution"].values for k in keys]
    if np.allclose(t2_vectors[0], t2_vectors[1]) and np.allclose(
        t2_vectors[1], t2_vectors[2]
    ):
        print(
            "\nWARNING: T² contribution vectors are identical across 1/5/10 Nm "
            "— check condition masking."
        )
    else:
        print(
            "\nGrouping OK: T² contribution vectors differ across conditions "
            f"(max pairwise diff: "
            f"{max(np.max(np.abs(t2_vectors[i]-t2_vectors[j])) for i in range(3) for j in range(i+1,3)):.4f})"
        )
    if np.allclose(q_vectors[0], q_vectors[1]) and np.allclose(
        q_vectors[1], q_vectors[2]
    ):
        print(
            "WARNING: Q contribution vectors are identical across 1/5/10 Nm "
            "— check condition masking."
        )
    else:
        print(
            "Grouping OK: Q contribution vectors differ across conditions "
            f"(max pairwise diff: "
            f"{max(np.max(np.abs(q_vectors[i]-q_vectors[j])) for i in range(3) for j in range(i+1,3)):.4f})"
        )


def find_common_fault_indicators(
    t2_dfs: dict[str, pd.DataFrame],
    q_dfs: dict[str, pd.DataFrame],
    top_n: int = TOP_CONDITION_CONTRIB,
) -> tuple[set[str], set[str]]:
    """Variables in top-N for T² (or Q) across all 1/5/10 Nm conditions."""
    t2_sets = [
        set(df.nlargest(top_n, "contribution")["variable"]) for df in t2_dfs.values()
    ]
    q_sets = [
        set(df.nlargest(top_n, "contribution")["variable"]) for df in q_dfs.values()
    ]
    t2_common = set.intersection(*t2_sets) if t2_sets else set()
    q_common = set.intersection(*q_sets) if q_sets else set()
    return t2_common, q_common


def print_common_fault_indicators(
    t2_common: set[str],
    q_common: set[str],
    t2_dfs: dict[str, pd.DataFrame],
    q_dfs: dict[str, pd.DataFrame],
) -> None:
    """Print ground-fault diagnostic candidates common to 1/5/10 Nm."""
    print(
        f"\nCommon fault indicators (top {TOP_CONDITION_CONTRIB} in ALL of 1/5/10 Nm):"
    )

    print(f"  T² common ({len(t2_common)}):")
    if t2_common:
        for var in sorted(t2_common):
            vals = ", ".join(
                f"{cond}: {t2_dfs[cond].set_index('variable').loc[var, 'contribution']:.4f}"
                for cond in FAULT_CONDITIONS
            )
            print(f"    - {var}  [{vals}]")
    else:
        print("    (none)")

    print(f"  Q common ({len(q_common)}):")
    if q_common:
        for var in sorted(q_common):
            vals = ", ".join(
                f"{cond}: {q_dfs[cond].set_index('variable').loc[var, 'contribution']:.4f}"
                for cond in FAULT_CONDITIONS
            )
            print(f"    - {var}  [{vals}]")
    else:
        print("    (none)")

    both = t2_common | q_common
    if both:
        summary = pd.DataFrame(
            {
                "variable": sorted(both),
                "in_T2_top10_all": [v in t2_common for v in sorted(both)],
                "in_Q_top10_all": [v in q_common for v in sorted(both)],
            }
        )
        print("\n  Combined candidate table:")
        print(summary.to_string(index=False))


def contribution_scores_t2_1t(
    X_scaled: np.ndarray,
    combined: pd.DataFrame,
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Hotelling T² variable contributions for 1T pooled mean vs 15T PCA model.
    c_i = x_i * sum_j (t_j / lambda_j) * p_ij
    """
    x_fault_mean = _fault_pooled_mean(X_scaled, combined)
    scores = pca.transform(x_fault_mean.reshape(1, -1))[0]
    eigenvalues = pca.explained_variance_
    loadings = pca.components_

    weighted_loadings = np.zeros(len(feature_names))
    for j, lam in enumerate(eigenvalues):
        weighted_loadings += (scores[j] / lam) * loadings[j, :]

    contrib = x_fault_mean * weighted_loadings
    return _top_contribution_df(feature_names, contrib)


def fit_linear_svm_for_pca(scores: np.ndarray, combined: pd.DataFrame) -> SVC:
    """
    Fit linear SVM in PC space:
    - Normal class 0: 15 Nm
    - Fault class 1: 10/5/1 Nm
    """
    X_svm = scores[:, :3]
    y_svm = (combined["condition"] != NORMAL_CONDITION).astype(int).to_numpy()
    svm = SVC(kernel="linear")
    svm.fit(X_svm, y_svm)
    return svm


def make_svm_boundary_surface(
    scores: np.ndarray,
    svm: SVC,
    grid_size: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build SVM decision plane z=f(x,y) on PC1-PC2 mesh:
    w0*x + w1*y + w2*z + b = 0 -> z = -(w0*x + w1*y + b)/w2
    """
    # Ensure numeric float arrays (avoid dtype issues -> NaN)
    pc1 = scores[:, 0].astype(np.float64, copy=False)
    pc2 = scores[:, 1].astype(np.float64, copy=False)
    pc3 = scores[:, 2].astype(np.float64, copy=False)

    w = np.asarray(svm.coef_[0], dtype=np.float64)
    b = float(np.asarray(svm.intercept_[0], dtype=np.float64))

    # Stabilize division when w[2] is extremely small (would create inf/NaN)
    w2 = float(w[2])
    eps = 1e-12
    if abs(w2) < eps:
        # Keep plane formula usable; sign preserves orientation for visualization.
        w2 = eps if w2 >= 0 else -eps

    x_pad = 0.1 * (pc1.max() - pc1.min() + 1e-9)
    y_pad = 0.1 * (pc2.max() - pc2.min() + 1e-9)
    z_pad = 0.1 * (pc3.max() - pc3.min() + 1e-9)

    x_vals = np.linspace(pc1.min() - x_pad, pc1.max() + x_pad, grid_size, dtype=np.float64)
    y_vals = np.linspace(pc2.min() - y_pad, pc2.max() + y_pad, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    # Linear SVM plane: w0*x + w1*y + w2*z + b = 0  ->  z = -(w0*x + w1*y + b)/w2
    grid_z = -((w[0] * grid_x) + (w[1] * grid_y) + b) / w2

    z_min = float(pc3.min() - z_pad)
    z_max = float(pc3.max() + z_pad)

    # Remove NaN/Inf and keep everything finite (Plotly writes NaN as null)
    grid_z = np.nan_to_num(
        grid_z,
        nan=(z_min + z_max) / 2.0,
        posinf=z_max,
        neginf=z_min,
        copy=False,
    )

    # Finally, clip to a reasonable z-range so the surface stays near the point cloud
    grid_z = np.clip(grid_z, z_min, z_max)

    return grid_x, grid_y, grid_z


def print_pc1_contributions(pc1_contrib_df: pd.DataFrame, evr_pc1: float) -> None:
    """Print ranked parameters that contribute most to PC1 (by absolute loading)."""
    ranked = pc1_contrib_df.sort_values("abs_loading", ascending=False)
    print(f"\nPC1 top contributors ({evr_pc1:.2f}% explained variance):")
    for rank, row in enumerate(ranked.itertuples(), start=1):
        sign = "+" if row.loading >= 0 else "-"
        print(
            f"  {rank:2d}. {row.variable}  "
            f"loading={sign}{abs(row.loading):.4f}"
        )


def build_dashboard(
    combined: pd.DataFrame,
    feature_names: list[str],
    scaler: StandardScaler,
    pca: PCA,
    X_scaled: np.ndarray,
) -> go.Figure:
    normal_mask = (combined["condition"] == NORMAL_CONDITION).to_numpy()
    t2, q = compute_mspc(X_scaled, pca)
    scores = pca.transform(X_scaled)
    evr = pca.explained_variance_ratio_ * 100
    svm = fit_linear_svm_for_pca(scores, combined)
    boundary_x, boundary_y, boundary_z = make_svm_boundary_surface(scores, svm)

    t2_ucl = ucl_from_normal(t2, normal_mask)
    q_ucl = ucl_from_normal(q, normal_mask)
    pc1_contrib_df = pc1_loading_contributions(pca, feature_names)

    plot_df = combined.copy()
    plot_df["PC1"] = scores[:, 0]
    plot_df["PC2"] = scores[:, 1]
    plot_df["PC3"] = scores[:, 2]
    plot_df["T2"] = t2
    plot_df["Q"] = q
    plot_df["time_s"] = plot_df.iloc[:, 0]

    # Condition-wise T² / Q contribution analysis (1, 5, 10 Nm fault conditions)
    t2_full: dict[str, pd.DataFrame] = {}
    q_full: dict[str, pd.DataFrame] = {}
    t2_top: dict[str, pd.DataFrame] = {}
    q_top: dict[str, pd.DataFrame] = {}
    cond_stats: dict[str, tuple[float, float]] = {}

    for torque in FAULT_PANEL_TORQUES:
        condition = _condition_label(torque)
        t2_df, q_df, t2_mean, q_mean = condition_mean_contributions(
            X_scaled, scores, combined, pca, feature_names, condition
        )
        t2_full[condition] = t2_df
        q_full[condition] = q_df
        t2_top[condition] = top_contribution_bar_df(t2_df, TOP_CONDITION_CONTRIB)
        q_top[condition] = top_contribution_bar_df(q_df, TOP_CONDITION_CONTRIB)
        cond_stats[condition] = (t2_mean, q_mean)

    validate_condition_grouping(combined, t2_full, q_full, top_n=5)

    t2_common, q_common = find_common_fault_indicators(t2_full, q_full)
    print_common_fault_indicators(t2_common, q_common, t2_full, q_full)

    svm_title = f"SVM: {NORMAL_CONDITION}=Normal, others=Fault"
    # Placeholder titles — exact text applied via _set_subplot_title after traces
    n_grid_cells = 5 * DASHBOARD_COLS
    fig = make_subplots(
        rows=5,
        cols=DASHBOARD_COLS,
        specs=[
            [{"type": "scene", "colspan": 3}, None, None],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[""] * n_grid_cells,
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
        row_heights=[0.28, 0.16, 0.12, 0.22, 0.22],
    )

    # [1,1] 3D PCA scatter + SVM decision boundary surface
    fig.add_trace(
        go.Surface(
            x=boundary_x,
            y=boundary_y,
            z=boundary_z,
            name="SVM Decision Boundary",
            legendgroup="svm",
            showlegend=True,
            showscale=False,
            opacity=0.35,
            colorscale=[[0.0, "rgba(220,220,220,1)"], [1.0, "rgba(220,220,220,1)"]],
            hovertemplate="SVM Decision Boundary<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for condition in TORQUE_ORDER:
        subset = plot_df[plot_df["condition"] == condition]
        fig.add_trace(
            go.Scatter3d(
                x=subset["PC1"],
                y=subset["PC2"],
                z=subset["PC3"],
                mode="markers",
                name=condition,
                legendgroup="pca",
                marker=dict(
                    size=MARKER_SIZE,
                    color=TORQUE_COLORS[condition],
                    opacity=0.9,
                ),
                text=subset["hover_label"],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    sample_x = plot_df["sample_order"]

    # [2,1] PC1 loading contribution bar chart (full width)
    colors_pc1 = [
        "#4ecdc4" if v >= 0 else "#ff6b6b" for v in pc1_contrib_df["loading"]
    ]
    fig.add_trace(
        go.Bar(
            x=pc1_contrib_df["loading"],
            y=pc1_contrib_df["variable"],
            orientation="h",
            name="PC1 loading",
            marker_color=colors_pc1,
            hovertemplate="%{y}<br>PC1 loading=%{x:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # [3,1] Hotelling T² control chart
    for condition in TORQUE_ORDER:
        subset = plot_df[plot_df["condition"] == condition]
        fig.add_trace(
            go.Scatter(
                x=subset["sample_order"],
                y=subset["T2"],
                mode="lines+markers",
                name=condition,
                legendgroup="t2",
                showlegend=False,
                line=dict(color=TORQUE_COLORS[condition], width=1.5),
                marker=dict(size=5, color=TORQUE_COLORS[condition]),
                hovertext=subset["hover_label"],
                hovertemplate="Sample %{x}<br>T²=%{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[sample_x.min(), sample_x.max()],
            y=[t2_ucl, t2_ucl],
            mode="lines",
            name=f"T² UCL ({NORMAL_CONDITION})",
            legendgroup="ucl",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate=f"T² UCL = {t2_ucl:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # [3,2] Q (SPE) control chart
    for condition in TORQUE_ORDER:
        subset = plot_df[plot_df["condition"] == condition]
        fig.add_trace(
            go.Scatter(
                x=subset["sample_order"],
                y=subset["Q"],
                mode="lines+markers",
                name=condition,
                legendgroup="q",
                showlegend=False,
                line=dict(color=TORQUE_COLORS[condition], width=1.5),
                marker=dict(size=5, color=TORQUE_COLORS[condition]),
                hovertext=subset["hover_label"],
                hovertemplate="Sample %{x}<br>Q=%{y:.4f}<extra></extra>",
            ),
            row=3,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[sample_x.min(), sample_x.max()],
            y=[q_ucl, q_ucl],
            mode="lines",
            name=f"Q UCL ({NORMAL_CONDITION})",
            legendgroup="ucl",
            showlegend=True,
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate=f"Q UCL = {q_ucl:.4f}<extra></extra>",
        ),
        row=3,
        col=2,
    )

    # [4,1-3] T² contribution: left→right = 10 → 5 → 1 torque (FAULT_PANEL_TORQUES)
    for col_idx, torque in enumerate(FAULT_PANEL_TORQUES, start=1):
        condition = _condition_label(torque)
        top_df = t2_top[condition]
        fig.add_trace(
            go.Bar(
                x=top_df["contribution"],
                y=top_df["variable"],
                orientation="h",
                name=f"{_torque_short_label(torque)} T² contrib",
                marker_color=TORQUE_COLORS[condition],
                hovertemplate="%{y}<br>T² contrib=%{x:.6f}<extra></extra>",
                showlegend=False,
            ),
            row=4,
            col=col_idx,
        )

    # [5,1-3] Q contribution: same panel order as T² row
    for col_idx, torque in enumerate(FAULT_PANEL_TORQUES, start=1):
        condition = _condition_label(torque)
        top_df = q_top[condition]
        fig.add_trace(
            go.Bar(
                x=top_df["contribution"],
                y=top_df["variable"],
                orientation="h",
                name=f"{_torque_short_label(torque)} Q contrib",
                marker_color=TORQUE_COLORS[condition],
                hovertemplate="%{y}<br>Q contrib=%{x:.6f}<extra></extra>",
                showlegend=False,
            ),
            row=5,
            col=col_idx,
        )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                "Diagnostic Pipeline: PCA → SVM → PC1 → MSPC (T²/Q) "
                "→ Condition Contributions"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=2000,
        width=1800,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title_text="Torque (Nm)",
        ),
        barmode="overlay",
    )

    fig.update_scenes(
        {
            "xaxis_title": f"PC1 ({evr[0]:.1f}%)",
            "yaxis_title": f"PC2 ({evr[1]:.1f}%)",
            "zaxis_title": f"PC3 ({evr[2]:.1f}%)",
            "bgcolor": "rgb(30,30,30)",
        },
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="PC1 loading (direction & magnitude)", row=2, col=1)
    fig.update_yaxes(title_text="Parameter", row=2, col=1)
    fig.update_xaxes(title_text="Sample index (time order)", row=3, col=1)
    fig.update_yaxes(title_text="Hotelling T²", row=3, col=1)
    fig.update_xaxes(title_text="Sample index (time order)", row=3, col=2)
    fig.update_yaxes(title_text="Q (SPE)", row=3, col=2)

    # Row 4–5 axes: panel order follows FAULT_PANEL_TORQUES
    for col_idx, torque in enumerate(FAULT_PANEL_TORQUES, start=1):
        fig.update_xaxes(title_text="T² contribution", row=4, col=col_idx)
        fig.update_yaxes(title_text="Parameter", row=4, col=col_idx)
        fig.update_xaxes(title_text="Q contribution", row=5, col=col_idx)
        fig.update_yaxes(title_text="Parameter", row=5, col=col_idx)

    # Subplot titles aligned to grid (row, col) — avoids colspan title drift
    _set_subplot_title(fig, 1, 1, f"Step 1 — 3D PCA + {svm_title}")
    _set_subplot_title(
        fig,
        2,
        1,
        f"Step 2 — PC1 ({evr[0]:.1f}% variance) Top {TOP_PC1_VARS} Contributors",
    )
    _set_subplot_title(fig, 3, 1, "Step 3 — Hotelling T² (MSPC)")
    _set_subplot_title(fig, 3, 2, "Step 3 — Q / SPE (MSPC)")

    for col_idx, torque in enumerate(FAULT_PANEL_TORQUES, start=1):
        condition = _condition_label(torque)
        t2_m, q_m = cond_stats[condition]
        label = _torque_short_label(torque)
        _set_subplot_title(
            fig,
            4,
            col_idx,
            f"{label} — T² Contribution (mean T²={t2_m:.2f}, mean Q={q_m:.1f})",
        )
        _set_subplot_title(
            fig,
            5,
            col_idx,
            f"{label} — Q Contribution (mean T²={t2_m:.2f}, mean Q={q_m:.1f})",
        )

    return fig


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    combined, feature_names = load_all_samples(base_dir)

    X = combined[feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Non-numeric values remain after coercion.")

    scaler, X_normal_scaled = fit_baseline_scaler(X, combined)
    X_scaled = transform_all_samples(X, scaler)
    pca = fit_baseline_pca(X_normal_scaled)

    scaler_path = base_dir / SCALER_PATH
    save_baseline_scaler(scaler, scaler_path)
    print_scaling_diagnostics(scaler, X_scaled, combined)

    normal_mask = _normal_sample_mask(combined)
    t2, q = compute_mspc(X_scaled, pca)
    t2_ucl = ucl_from_normal(t2, normal_mask)
    q_ucl = ucl_from_normal(q, normal_mask)

    pc1_contrib_df = pc1_loading_contributions(pca, feature_names)
    evr = pca.explained_variance_ratio_ * 100
    print_pc1_contributions(pc1_contrib_df, evr[0])

    fig = build_dashboard(combined, feature_names, scaler, pca, X_scaled)

    out_path = base_dir / OUTPUT_HTML
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    n_files_per_cond = len(next(iter(TORQUE_GROUPS.values())))
    print(
        f"Samples: {len(combined)} averaged points "
        f"({len(TORQUE_GROUPS)} torque conditions x {LAST_N_ROWS} rows, "
        f"each averaged over {n_files_per_cond} replicate files)"
    )
    print(f"Features: {len(feature_names)}")
    print(
        f"Baseline fit: {NORMAL_CONDITION} only "
        f"(StandardScaler fit_transform + PCA on scaled data)"
    )
    print(f"All conditions: transform-only with saved scaler -> {scaler_path}")
    print(
        "Explained variance (%): "
        + ", ".join(f"{v:.2f}" for v in evr)
        + f" (total {evr.sum():.2f})"
    )
    print(f"MSPC T² UCL ({NORMAL_CONDITION} mean+3σ): {t2_ucl:.4f}")
    print(f"MSPC Q UCL ({NORMAL_CONDITION} mean+3σ): {q_ucl:.4f}")
    print(
        f"SVM boundary: {NORMAL_CONDITION}=class 0 (normal), "
        "10/5/1 Nm=class 1 (fault) in PC1-PC2-PC3 space"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
