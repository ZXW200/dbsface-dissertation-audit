"""Image-similarity cluster sensitivity for class-specific AEV statistics.

The module groups highly similar test images through cosine connected
components. It then repeats the Class 0 versus Class 1 comparison with each
image-similarity cluster as the resampling unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_aev_class_statistics import bh_fdr, cliffs_delta, cohen_d  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from load_pd_dbs import load_pd_dbs  # noqa: E402


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    norm = np.sqrt((x * x).sum(axis=1)) + 1e-12
    xn = x / norm[:, None]
    return xn @ xn.T


def pseudo_clusters(cos: np.ndarray, threshold: float) -> np.ndarray:
    """Connected-component pseudo-clusters from a cosine-similarity graph."""

    n = cos.shape[0]
    adj = cos >= threshold
    np.fill_diagonal(adj, False)
    graph = csr_matrix(adj)
    n_comp, labels = connected_components(graph, directed=False)
    _ = n_comp
    return labels.astype(int)


def cluster_permutation_p(
    cluster_vals0: np.ndarray,
    cluster_vals1: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Permutation test with the (pure) cluster as the exchangeable unit."""

    pooled = np.concatenate([cluster_vals0, cluster_vals1])
    n1 = len(cluster_vals1)
    observed = abs(float(np.mean(cluster_vals1) - np.mean(cluster_vals0)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n1]) - np.mean(perm[n1:])))
        if diff >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def cluster_bootstrap_ci(
    image_vals: np.ndarray,
    image_labels: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Cluster bootstrap CI for the image-level Class1-Class0 mean difference.

    Resamples whole clusters with replacement and pools the member images, so
    the interval reflects the effective number of independent clusters rather
    than the raw image count.
    """

    unique_clusters = np.unique(cluster_ids)
    member_index = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([member_index[c] for c in drawn])
        lab = image_labels[idx]
        val = image_vals[idx]
        if (lab == 0).sum() == 0 or (lab == 1).sum() == 0:
            continue
        diffs.append(float(val[lab == 1].mean() - val[lab == 0].mean()))
    if not diffs:
        return float("nan"), float("nan")
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_robust_wald_p(
    y_drop: np.ndarray, x_class: np.ndarray, cluster_ids: np.ndarray
) -> tuple[float, float, float]:
    """Cluster-robust (CR1 sandwich) Wald p for OLS drop ~ 1 + class.

    Orthogonal cross-check to the cluster-mean permutation: uses every image
    (including mixed clusters) but corrects the standard error for clustering
    on the pseudo-cluster. beta_class equals the image-level Class1-Class0 mean
    difference; only its variance is cluster-adjusted.
    """

    n = len(y_drop)
    X = np.column_stack([np.ones(n), x_class.astype(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y_drop)
    resid = y_drop - X @ beta
    meat = np.zeros((2, 2))
    uniq = np.unique(cluster_ids)
    g = len(uniq)
    for c in uniq:
        idx = np.where(cluster_ids == c)[0]
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    k = X.shape[1]
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    var = correction * (XtX_inv @ meat @ XtX_inv)
    se = float(np.sqrt(var[1, 1]))
    if se <= 0:
        return 1.0, float(beta[1]), se
    z = float(beta[1]) / se
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return p, float(beta[1]), se


def perm_fdr_counts(
    df: pd.DataFrame,
    roi_cols: list[str],
    y: np.ndarray,
    labels: np.ndarray,
    pure_cluster_ids: list[int],
    pure_labels: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Seed-dependent FDR-significant counts (image and cluster-mean permutation)."""

    img_ps, cl_ps = [], []
    for col in roi_cols:
        vals = df[col].to_numpy(dtype=float)
        img_ps.append(_perm_p(vals[y == 0], vals[y == 1], n_perm, rng))
        cmeans = np.array([vals[labels == c].mean() for c in pure_cluster_ids], dtype=float)
        cl_ps.append(
            cluster_permutation_p(cmeans[pure_labels == 0], cmeans[pure_labels == 1], n_perm, rng)
        )
    img_fdr = bh_fdr(np.array(img_ps))
    cl_fdr = bh_fdr(np.array(cl_ps))
    return int((img_fdr < 0.05).sum()), int((cl_fdr < 0.05).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aev", default="outputs/aev/aev_test.csv")
    parser.add_argument("--data", default="data/raw/PD_DBS_Data.mat")
    parser.add_argument("--output-dir", default="outputs/aev/cluster_robust")
    parser.add_argument(
        "--thresholds",
        default="0.999,0.995,0.99,0.98",
        help="Comma-separated cosine thresholds (high = strict near-duplicate).",
    )
    parser.add_argument("--primary-threshold", type=float, default=0.99)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.aev).reset_index(drop=True)
    roi_cols = [c for c in df.columns if c.startswith("evidence_drop__")]
    y = df["y_true"].to_numpy(dtype=int)
    n_images = len(df)

    data = load_pd_dbs(args.data)
    x_test = data["x_test_flat"].astype(np.float64)
    if len(x_test) != n_images:
        raise ValueError(f"AEV rows ({n_images}) != test images ({len(x_test)}); ordering assumption broken.")

    cos = cosine_matrix(x_test)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    cluster_assignments = {"sample_id": df["sample_id"], "y_true": y}
    cluster_summary_rows = []
    roi_rows = []

    for tau in thresholds:
        labels = pseudo_clusters(cos, tau)
        cluster_assignments[f"cluster_tau_{tau}"] = labels
        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique)
        sizes = counts
        n_singletons = int((sizes == 1).sum())
        n_nonsingleton = int((sizes > 1).sum())
        max_size = int(sizes.max())
        mean_size = float(sizes.mean())

        # Purity: a cluster is pure if all member labels agree.
        pure_flags = {}
        pure_label = {}
        for c in unique:
            members = np.where(labels == c)[0]
            lbls = y[members]
            is_pure = bool((lbls == lbls[0]).all())
            pure_flags[c] = is_pure
            pure_label[c] = int(lbls[0]) if is_pure else -1
        n_pure = int(sum(pure_flags.values()))
        n_mixed = n_clusters - n_pure
        n_nonsingleton_pure = int(sum(pure_flags[c] for c in unique if (labels == c).sum() > 1))
        purity_nonsingleton = (
            float(n_nonsingleton_pure / n_nonsingleton) if n_nonsingleton else 1.0
        )
        design_effect = float(n_images / n_clusters)
        chaining_flag = bool(max_size > 0.05 * n_images)

        cluster_summary_rows.append(
            {
                "threshold": tau,
                "n_images": n_images,
                "n_clusters": n_clusters,
                "n_singletons": n_singletons,
                "n_nonsingleton_clusters": n_nonsingleton,
                "max_cluster_size": max_size,
                "mean_cluster_size": mean_size,
                "n_pure_clusters": n_pure,
                "n_mixed_clusters": n_mixed,
                "nonsingleton_purity": purity_nonsingleton,
                "design_effect_N_over_clusters": design_effect,
                "effective_N": float(n_clusters),
                "chaining_flag": chaining_flag,
            }
        )

        # Per-ROI image-level vs cluster-level statistics.
        pure_cluster_ids = [c for c in unique if pure_flags[c]]
        pure_labels = np.array([pure_label[c] for c in pure_cluster_ids])
        image_p_perm = []
        cluster_p_perm = []
        cluster_robust_p = []
        tmp_rows = []
        for col in roi_cols:
            roi = col.replace("evidence_drop__", "")
            vals = df[col].to_numpy(dtype=float)
            x0 = vals[y == 0]
            x1 = vals[y == 1]

            # image-level (reproduces analyze_aev_class_statistics)
            img_diff = float(x1.mean() - x0.mean())
            img_d = cohen_d(x0, x1)
            img_delta = cliffs_delta(x0, x1)
            img_p = _perm_p(x0, x1, args.n_permutations, rng)
            image_p_perm.append(img_p)

            # cluster-level aggregation over pure clusters
            cmeans = np.array(
                [vals[labels == c].mean() for c in pure_cluster_ids], dtype=float
            )
            c0 = cmeans[pure_labels == 0]
            c1 = cmeans[pure_labels == 1]
            cl_diff = float(c1.mean() - c0.mean())
            cl_d = cohen_d(c0, c1)
            cl_delta = cliffs_delta(c0, c1)
            cl_p = cluster_permutation_p(c0, c1, args.n_permutations, rng)
            cluster_p_perm.append(cl_p)

            cb_lo, cb_hi = cluster_bootstrap_ci(
                vals, y, labels, args.n_bootstrap, rng
            )

            # Orthogonal cross-check: cluster-robust sandwich SE over all images.
            cr_p, cr_beta, cr_se = cluster_robust_wald_p(vals, y, labels)
            cluster_robust_p.append(cr_p)

            tmp_rows.append(
                {
                    "threshold": tau,
                    "roi": roi,
                    "n_clusters_class0": int((pure_labels == 0).sum()),
                    "n_clusters_class1": int((pure_labels == 1).sum()),
                    "image_diff": img_diff,
                    "image_cohen_d": img_d,
                    "image_cliffs_delta": img_delta,
                    "image_p_perm": img_p,
                    "cluster_diff": cl_diff,
                    "cluster_cohen_d": cl_d,
                    "cluster_cliffs_delta": cl_delta,
                    "cluster_p_perm": cl_p,
                    "cluster_bootstrap_ci_low": cb_lo,
                    "cluster_bootstrap_ci_high": cb_hi,
                    "cluster_robust_diff": cr_beta,
                    "cluster_robust_se": cr_se,
                    "cluster_robust_p": cr_p,
                }
            )

        image_fdr = bh_fdr(np.array(image_p_perm))
        cluster_fdr = bh_fdr(np.array(cluster_p_perm))
        cluster_robust_fdr = bh_fdr(np.array(cluster_robust_p))
        for i, row in enumerate(tmp_rows):
            row["image_p_fdr"] = float(image_fdr[i])
            row["cluster_p_fdr"] = float(cluster_fdr[i])
            row["cluster_robust_p_fdr"] = float(cluster_robust_fdr[i])
            row["image_fdr_sig"] = bool(image_fdr[i] < 0.05)
            row["cluster_fdr_sig"] = bool(cluster_fdr[i] < 0.05)
            row["cluster_robust_fdr_sig"] = bool(cluster_robust_fdr[i] < 0.05)
            row["direction"] = "higher in Class 1" if row["cluster_diff"] > 0 else "higher in Class 0"
            row["sign_stable"] = bool(np.sign(row["image_diff"]) == np.sign(row["cluster_diff"]))
            roi_rows.append(row)

    clusters_df = pd.DataFrame(cluster_assignments)
    clusters_df.to_csv(out_dir / "test_pseudo_clusters.csv", index=False)

    summary_df = pd.DataFrame(cluster_summary_rows)
    summary_df.to_csv(out_dir / "cluster_summary.csv", index=False)

    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(out_dir / "roi_class_comparison_cluster_robust.csv", index=False)

    # Headline: image vs cluster FDR-significant counts per threshold.
    headline = []
    for tau in thresholds:
        sub = roi_df[roi_df["threshold"] == tau]
        headline.append(
            {
                "threshold": tau,
                "n_roi": int(len(sub)),
                "image_fdr_significant": int(sub["image_fdr_sig"].sum()),
                "cluster_fdr_significant": int(sub["cluster_fdr_sig"].sum()),
                "cluster_robust_fdr_significant": int(sub["cluster_robust_fdr_sig"].sum()),
                "sign_stable": int(sub["sign_stable"].sum()),
            }
        )
    headline_df = pd.DataFrame(headline)

    # F: multi-seed stability of the permutation results at the primary threshold.
    primary_labels = pseudo_clusters(cos, args.primary_threshold)
    p_unique = np.unique(primary_labels)
    p_pure_ids = [
        int(c) for c in p_unique if bool((y[primary_labels == c] == y[primary_labels == c][0]).all())
    ]
    p_pure_labels = np.array([int(y[primary_labels == c][0]) for c in p_pure_ids])
    seed_rows = []
    for s in [args.seed, 1, 2, 3, 4]:
        img_c, cl_c = perm_fdr_counts(
            df, roi_cols, y, primary_labels, p_pure_ids, p_pure_labels,
            args.n_permutations, np.random.default_rng(s),
        )
        seed_rows.append({"seed": int(s), "image_fdr_sig": img_c, "cluster_fdr_sig": cl_c})
    multiseed_df = pd.DataFrame(seed_rows)
    multiseed_df.to_csv(out_dir / "multiseed_stability.csv", index=False)

    primary = summary_df[summary_df["threshold"] == args.primary_threshold]
    primary_head = headline_df[headline_df["threshold"] == args.primary_threshold]
    primary_summary = cluster_summary_rows[thresholds.index(args.primary_threshold)]
    primary_threshold_rationale = (
        f"Cosine>={args.primary_threshold} was used as the primary operating point. It produced "
        f"{primary_summary['n_clusters']} compact image-similarity groups, with a largest group "
        f"of {primary_summary['max_cluster_size']} images. Lower thresholds are included as "
        f"broader sensitivity settings."
    )
    ms_cluster = [r["cluster_fdr_sig"] for r in seed_rows]
    ms_image = [r["image_fdr_sig"] for r in seed_rows]
    metrics = {
        "n_images": n_images,
        "thresholds": thresholds,
        "primary_threshold": args.primary_threshold,
        "primary_threshold_rationale": primary_threshold_rationale,
        "cluster_summary": cluster_summary_rows,
        "headline_image_vs_cluster_fdr": headline,
        "multiseed_stability": {
            "seeds": [r["seed"] for r in seed_rows],
            "image_fdr_sig_range": [int(min(ms_image)), int(max(ms_image))],
            "cluster_fdr_sig_range": [int(min(ms_cluster)), int(max(ms_cluster))],
        },
        "note": (
            "Cosine pseudo-clusters represent groups of highly similar test images. "
            "The cluster-mean permutation and CR1 sandwich estimator assess sensitivity "
            "to using those groups as the resampling unit."
        ),
    }
    (out_dir / "cluster_robust_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    md = [
        "# Near-Duplicate Cluster-Robust Sensitivity Analysis for Class-Specific AEV",
        "",
        "Resampling unit = within-test near-duplicate pseudo-cluster (cosine connected components).",
        "The analysis compares image-level estimates with results obtained when highly similar",
        "test images are grouped into cosine-connected components.",
        "",
        "## Cluster structure by threshold",
        "",
        "| tau | clusters | singletons | non-singleton | max size | design effect (N/clusters) | non-singleton purity | chaining |",
        "|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in cluster_summary_rows:
        md.append(
            f"| {r['threshold']} | {r['n_clusters']} | {r['n_singletons']} | "
            f"{r['n_nonsingleton_clusters']} | {r['max_cluster_size']} | "
            f"{r['design_effect_N_over_clusters']:.3f} | {r['nonsingleton_purity']:.3f} | "
            f"{'YES' if r['chaining_flag'] else 'no'} |"
        )
    md += [
        "",
        "## Image-level vs near-duplicate cluster-level FDR-significant ROIs (of 8)",
        "",
        "cluster = cluster-mean permutation (Method A); cluster-robust = CR1 sandwich (Method C).",
        "",
        "| tau | image FDR-sig | cluster FDR-sig | cluster-robust FDR-sig | sign-stable |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in headline:
        md.append(
            f"| {h['threshold']} | {h['image_fdr_significant']} | "
            f"{h['cluster_fdr_significant']} | {h['cluster_robust_fdr_significant']} | "
            f"{h['sign_stable']} |"
        )
    md += [
        "",
        f"Primary operating point: {primary_threshold_rationale}",
        "",
        f"Multi-seed stability (5 seeds) at cosine>={args.primary_threshold}: "
        f"cluster FDR-sig range {min(ms_cluster)}-{max(ms_cluster)}, "
        f"image FDR-sig range {min(ms_image)}-{max(ms_image)}.",
    ]
    (out_dir / "cluster_robust_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print()
    print(headline_df.to_string(index=False))
    print()
    print("multi-seed at primary:", multiseed_df.to_dict("records"))
    if len(primary):
        print()
        print(
            f"[primary tau={args.primary_threshold}] "
            f"design effect {float(primary['design_effect_N_over_clusters'].iloc[0]):.3f}, "
            f"image FDR-sig {int(primary_head['image_fdr_significant'].iloc[0])} -> "
            f"cluster(A) FDR-sig {int(primary_head['cluster_fdr_significant'].iloc[0])} -> "
            f"cluster-robust(C) FDR-sig {int(primary_head['cluster_robust_fdr_significant'].iloc[0])} of 8, "
            f"sign-stable {int(primary_head['sign_stable'].iloc[0])}/8"
        )
    return 0


def _perm_p(x0: np.ndarray, x1: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    observed = abs(float(x1.mean() - x0.mean()))
    pooled = np.concatenate([x0, x1])
    n1 = len(x1)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(float(perm[:n1].mean() - perm[n1:].mean())) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


if __name__ == "__main__":
    raise SystemExit(main())
