# %%
"""
Plotting utilities for EV Stag Hunt experiments.

All functions accept pandas DataFrames produced by ev_experiments
and save figures to disk, returning the output path.
"""
# %%
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# %%
def _default_plot_path(filename: str) -> str:
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)

# %%
def plot_fanchart(
    traces_df: pd.DataFrame,
    *,
    out_path: Optional[str] = None,
    fig_title: Optional[str] = None
) -> str:
    """Plot fan charts (quantile bands) for baseline vs subsidy using traces DF."""

    if traces_df.empty:
        raise ValueError("traces_df is empty")

    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        q = gdf.groupby("time")["X"].quantile([0.10, 0.25, 0.75, 0.90]).unstack(level=1)
        mean = gdf.groupby("time")["X"].mean()
        t = mean.index.to_numpy()

        ax = axes[0, j]
        ax.fill_between(t, q[0.10], q[0.90], color=("steelblue" if group=="baseline" else "darkorange"), alpha=0.15)
        ax.fill_between(t, q[0.25], q[0.75], color=("steelblue" if group=="baseline" else "darkorange"), alpha=0.30)
        trial_ids = gdf["trial"].unique()
        rng = np.random.default_rng(123)
        sample = rng.choice(trial_ids, size=min(100, len(trial_ids)), replace=False)
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(tr_df["time"], tr_df["X"], color=("steelblue" if group=="baseline" else "darkorange"), alpha=0.1, linewidth=0.8)
        ax.plot(t, mean, color=("steelblue" if group=="baseline" else "darkorange"), linewidth=2)
        ax.set_title(f"{group.capitalize()} adoption")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0,1)

        # Final histogram
        t_max = int(gdf["time"].max())
        final_vals = gdf[gdf["time"] == t_max].groupby("trial")["X"].mean().to_numpy()
        axes[1, j].hist(final_vals, bins=20, color=("steelblue" if group=="baseline" else "darkorange"), alpha=0.8)
        axes[1, j].set_title(f"{group.capitalize()} final X(T)")
        axes[1, j].set_xlabel("X(T)")
        axes[1, j].set_ylabel("Count")

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    if out_path is None:
        # default filename
        out_path = os.path.join(os.getcwd(), "plots", "ev_intervention_fanchart.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # ✅ THIS IS THE CRUCIAL LINE
    return out_path


# %%

def plot_spaghetti(
    traces_df: pd.DataFrame,
    *,
    max_traces: int = 100,
    alpha: float = 0.15,
    out_path: Optional[str] = None,
    fig_title: Optional[str] = None  # new argument
) -> str:
    """Spaghetti plot from traces DF for baseline vs subsidy."""

    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rng = np.random.default_rng(123)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        trial_ids = gdf["trial"].unique()
        sample = rng.choice(trial_ids, size=min(max_traces, len(trial_ids)), replace=False)
        ax = axes[j]
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(
                tr_df["time"],
                tr_df["X"],
                color=("steelblue" if group == "baseline" else "darkorange"),
                alpha=alpha,
                linewidth=0.8
            )
        ax.set_title(f"{group.capitalize()} traces")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0, 1)

    # ✅ figure-level title
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    if out_path is None:
        out_path = _default_plot_path("ev_spaghetti.png")

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# %%
from matplotlib.colors import LogNorm
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd

def plot_density(
    traces_df: pd.DataFrame,
    *,
    x_bins: int = 50,
    time_bins: Optional[int] = None,
    out_path: Optional[str] = None,
    fig_title: Optional[str] = None  # new argument
) -> str:
    """Time-evolving density plot (2D histogram) from traces DF with white background."""
    
    groups = ["baseline", "subsidy"]
    
    # Figure with white background
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, facecolor="white")

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        T = int(gdf["time"].max()) + 1
        bins_time = T if time_bins is None else time_bins
        
        # White axes background
        axes[j].set_facecolor("white")
        
        # 2D histogram with high-contrast colors
        hb = axes[j].hist2d(
            gdf["time"].to_numpy(),
            gdf["X"].to_numpy(),
            bins=[bins_time, x_bins],
            range=[[0, T - 1], [0.0, 1.0]],
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=50)
        )
        
        axes[j].set_title(f"{group.capitalize()} density: time vs X(t)")
        axes[j].set_xlabel("Time")
        axes[j].set_ylabel("X(t)")
        fig.colorbar(hb[3], ax=axes[j], label="count")
    
    # Set overall figure title if provided
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    if out_path is None:
        out_path = _default_plot_path("ev_density.png")
        
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path



# %%


def plot_ratio_sweep(sweep_df: pd.DataFrame, *, out_path: Optional[str] = None, fig_title: Optional[str] = None) -> str:
    """Plot X* vs ratio from a DataFrame with columns ['ratio','X_mean'].""" 
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(sweep_df["ratio"], sweep_df["X_mean"], color="C0", lw=2)
    ax.set_xlabel("a_I / b (ratio)")
    ax.set_ylabel("Final adoption X*")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    
    # Set overall figure title if provided
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    
    if out_path is None:
        out_path = _default_plot_path("ev_ratio_sweep.png")
    
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
def plot_phase_plot(
    phase_df: pd.DataFrame,
    *,
    out_path: Optional[str] = None,
    fig_title: Optional[str] = None
) -> str:
    """Plot heatmap from tidy DataFrame with columns ['X0','ratio','X_final'].""" 

    pivot = phase_df.pivot(
        index="ratio", columns="X0", values="X_final"
    ).sort_index().sort_index(axis=1)

    ratios = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    fig = plt.figure(figsize=(7, 4))

    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], ratios[0], ratios[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )

    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")

    # Threshold curve
    X_thresh = np.clip(1.0 / ratios, 0.0, 1.0)
    plt.plot(
        X_thresh,
        ratios,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="X = b / a_I"
    )
    plt.legend(loc="upper right")

    # ✅ figure-level title
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    if out_path is None:
        out_path = _default_plot_path("ev_phase_plot.png")

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# %%
def plot_phase_plot(
    phase_df: pd.DataFrame,
    *,
    out_path: Optional[str] = None,
    fig_title: Optional[str] = None
) -> str:
    """Plot heatmap from tidy DataFrame with columns ['X0','ratio','X_final'].""" 

    pivot = phase_df.pivot(
        index="ratio", columns="X0", values="X_final"
    ).sort_index().sort_index(axis=1)

    ratios = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    fig = plt.figure(figsize=(7, 4))

    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], ratios[0], ratios[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )

    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")

    # Threshold curve
    X_thresh = np.clip(1.0 / ratios, 0.0, 1.0)
    plt.plot(
        X_thresh,
        ratios,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="X = b / a_I"
    )
    plt.legend(loc="upper right")

    # ✅ figure-level title
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    if out_path is None:
        out_path = _default_plot_path("ev_phase_plot.png")

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path

# %%
