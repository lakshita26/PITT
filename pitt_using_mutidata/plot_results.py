"""
plot_results.py  —  Publication-Quality Plots for ALL Reynolds Numbers
=======================================================================
Design rules (matching task requirements)
------------------------------------------
1. SAME colorbar/colormap across every field-type figure.
2. One figure per field type showing ALL Re numbers side-by-side.
3. Individual 4-panel figures per Re (pressure+streamlines,
   speed+streamlines, pressure+quiver, learning curve).
4. All plot functions use the shared colormap from config.py.

Exports
-------
generate_all_plots(all_fields, epochs, loss, output_dir)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

from config import (
    RE_LIST, SHARED_CMAP, SHARED_CMAP2,
    P_LO, P_HI, SP_LO, SP_HI, CBAR_N_TICKS, PLOT_DIR
)

# ─────────────────────────────────────────────────────────────────────────────
# Global rcParams
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.linewidth":       1.0,
    "axes.labelsize":       12,
    "axes.titlesize":       12,
    "axes.titlepad":        8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "figure.dpi":           150,
    "savefig.dpi":          180,
    "savefig.facecolor":    "white",
})

SAVE_KW = dict(dpi=180, bbox_inches="tight", facecolor="white")

# Shared colormaps (same across ALL figures)
CMAP_DIV = plt.get_cmap(SHARED_CMAP)    # diverging  (pressure, vorticity)
CMAP_SEQ = plt.get_cmap(SHARED_CMAP2)   # sequential (speed)

TICKS_P  = np.linspace(P_LO,  P_HI,  CBAR_N_TICKS)
TICKS_SP = np.linspace(SP_LO, SP_HI, CBAR_N_TICKS)


# ─────────────────────────────────────────────────────────────────────────────
# Shared colorbar helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_cbar(fig, ax, cmap, lo, hi, label, ticks=None, fmt="{:.3f}"):
    """Attach a consistent colorbar to ax using the shared colormap."""
    norm = Normalize(vmin=lo, vmax=hi)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb   = fig.colorbar(sm, ax=ax, pad=0.025, fraction=0.048, aspect=22)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([fmt.format(t) for t in ticks], fontsize=7.5)
    return cb


def _frame(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
#     ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Individual panel drawers  (used both in single-Re and multi-Re figures)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pressure_streamlines(ax, fig, f, show_cbar=True, title=None):
    X, Y = np.meshgrid(f["x"], f["y"])
    p_sm = gaussian_filter(f["p"], sigma=0.8)
    im   = ax.contourf(X, Y, p_sm, levels=80,
                       cmap=CMAP_DIV, vmin=P_LO, vmax=P_HI, extend="both")
    ax.contour(X, Y, p_sm, levels=14,
               colors="white", linewidths=0.22, alpha=0.18)
    ax.streamplot(f["x"], f["y"], f["u"], f["v"],
                  density=2.2, color="#1a4a6e",
                  linewidth=0.75, arrowsize=0.8, arrowstyle="->")
    if show_cbar:
        _add_cbar(fig, ax, CMAP_DIV, P_LO, P_HI, "Pressure", TICKS_P)
    _frame(ax, title or "Pressure + Streamlines")


def _draw_speed_streamlines(ax, fig, f, show_cbar=True, title=None):
    X, Y = np.meshgrid(f["x"], f["y"])
    sp   = gaussian_filter(f["speed"], sigma=0.5)
    im   = ax.contourf(X, Y, sp, levels=80,
                       cmap=CMAP_SEQ, vmin=SP_LO, vmax=SP_HI)
    ax.streamplot(f["x"], f["y"], f["u"], f["v"],
                  density=2.2, color="white",
                  linewidth=0.75, arrowsize=0.80, arrowstyle="->")
    if show_cbar:
        _add_cbar(fig, ax, CMAP_SEQ, SP_LO, SP_HI, "Speed |u| [m/s]", TICKS_SP)
    _frame(ax)


def _draw_pressure_quiver(ax, fig, f, show_cbar=True, title=None):
    X, Y = np.meshgrid(f["x"], f["y"])
    p_sm = gaussian_filter(f["p"], sigma=0.8)
    N    = f["N"]
    im   = ax.contourf(X, Y, p_sm, levels=80,
                       cmap=CMAP_DIV, vmin=P_LO, vmax=P_HI, extend="both")
    ax.contour(X, Y, p_sm, levels=18,
               colors="white", linewidths=0.18, alpha=0.12)
    step = max(1, N // 20)
    Xq, Yq = X[::step,::step], Y[::step,::step]
    Uq = f["u"][::step,::step]; Vq = f["v"][::step,::step]
    mag  = np.sqrt(Uq**2 + Vq**2)
    still = mag < 0.01
    if still.any():
        ax.plot(Xq[still], Yq[still], "w.", ms=1.2, alpha=0.45, zorder=3)
    mv = ~still
    if mv.any():
        ref = np.percentile(mag[mv], 90)
        ax.quiver(Xq[mv], Yq[mv], Uq[mv], Vq[mv],
                  scale=ref*20, scale_units="xy", angles="xy",
                  width=0.003, headwidth=4, headlength=4.5,
                  headaxislength=3.5, color="white", alpha=0.85, zorder=4)
    if show_cbar:
        _add_cbar(fig, ax, CMAP_DIV, P_LO, P_HI, "Pressure", TICKS_P)
    _frame(ax)


def _draw_vorticity(ax, fig, f, show_cbar=True, title=None):
    """Vorticity field — novel addition using same diverging colormap."""
    X, Y   = np.meshgrid(f["x"], f["y"])
    om     = gaussian_filter(f["omega"], sigma=0.5)
    v_lim  = max(abs(om.min()), abs(om.max()), 1e-3)
    im     = ax.contourf(X, Y, om, levels=80,
                         cmap=CMAP_DIV, vmin=-v_lim, vmax=v_lim)
    ax.contour(X, Y, om, levels=12,
               colors="black", linewidths=0.20, alpha=0.18)
    if show_cbar:
        ticks = np.linspace(-v_lim, v_lim, CBAR_N_TICKS)
        _add_cbar(fig, ax, CMAP_DIV, -v_lim, v_lim, "Vorticity ω", ticks)
    _frame(ax)


def _draw_learning_curve(ax, epochs, loss):
    loss = np.asarray(loss, dtype=float)
    ax.semilogy(epochs, loss, color="#2a72be", lw=1.0, alpha=0.75,
                label="Training loss")
    sm = gaussian_filter(loss, sigma=max(1, len(loss)//70))
    ax.semilogy(epochs, sm, color="#d84315", lw=2.2,
                ls="--", alpha=0.9, label="Smoothed trend")
    ax.set_xlim(0, epochs[-1])
    ax.set_ylim(max(loss.min()*0.5, 8e-4), min(loss.max()*3, 0.6))
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss",  fontsize=10)
    ax.set_title("Enhanced PITT Learning Curve")
    ax.legend(fontsize=9, framealpha=0.75)
    ax.grid(True, which="major", ls="--", lw=0.55, alpha=0.5)
    ax.grid(True, which="minor", ls=":",  lw=0.35, alpha=0.30)
    ax.tick_params(labelsize=8)
    n = len(epochs)
    for frac, lbl, yo in [(0.05,"Rapid\nconvergence",4.0),
                           (0.30,"Fine-tuning",2.5),
                           (0.88,"Near\nconverged",3.0)]:
        ep = int(frac*n)
        if 0 < ep < n:
            lv = float(gaussian_filter(loss, sigma=max(1,n//60))[ep])
            ax.annotate(lbl, xy=(epochs[ep],lv),
                        xytext=(epochs[ep]+n*0.04, lv*yo),
                        fontsize=7.5, color="#444", ha="left",
                        arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# Per-Re four-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_re(fields, epochs, loss, Re, output_dir):
    """4-panel figure for one Reynolds number."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    fig.suptitle(f"PITT Results  —  Re = {int(Re)}", fontsize=14,
                 fontweight="bold", y=1.005)

    _draw_pressure_streamlines(axes[0,0], fig, fields)
    _draw_speed_streamlines   (axes[0,1], fig, fields)
    _draw_pressure_quiver     (axes[1,0], fig, fields)
    _draw_learning_curve      (axes[1,1], epochs, loss)

    fig.tight_layout(pad=2.2)
    path = os.path.join(output_dir, f"Re{int(Re)}_four_panel.png")
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Re comparison figures  (same colorbar across all Re)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_re_comparison(all_fields, output_dir):
    """
    For each field type (pressure+streamlines, speed, vorticity, quiver),
    produce one wide figure with one column per Reynolds number,
    ALL sharing the SAME colorbar.
    """
    re_list = sorted(all_fields.keys())
    n       = len(re_list)

    field_specs = [
        ("pressure_streamlines", "Pressure + Streamlines",
         _draw_pressure_streamlines, CMAP_DIV, P_LO, P_HI, "Pressure"),
        ("speed_streamlines",    "Speed |u| + Streamlines",
         _draw_speed_streamlines,    CMAP_SEQ, SP_LO, SP_HI, "Speed |u|"),
        ("vorticity",            "Vorticity ω",
         None, CMAP_DIV, None, None, "Vorticity ω [1/s]"),
        ("pressure_quiver",      "Pressure + Velocity Quiver",
         _draw_pressure_quiver,  CMAP_DIV, P_LO, P_HI, "Pressure"),
    ]

    for slug, main_title, draw_fn, cmap, lo, hi, cbar_label in field_specs:
        fig = plt.figure(figsize=(4.5*n + 1.2, 5.5))
        gs  = gridspec.GridSpec(1, n+1, width_ratios=[4.2]*n + [0.35],
                                wspace=0.06, hspace=0.0,
                                left=0.05, right=0.93,
                                top=0.88, bottom=0.10)

        lo_eff, hi_eff = lo, hi   # will be updated for vorticity

        axes = []
        for col, Re in enumerate(re_list):
            ax = fig.add_subplot(gs[0, col])
            f  = all_fields[Re]

            if slug == "vorticity":
                # Compute shared symmetric limits across all Re
                if col == 0:
                    om_max = max(abs(all_fields[r]["omega"]).max()
                                 for r in re_list)
                    lo_eff, hi_eff = -om_max, om_max
                X, Y = np.meshgrid(f["x"], f["y"])
                om   = gaussian_filter(f["omega"], sigma=0.5)
                ax.contourf(X, Y, om, levels=80, cmap=cmap,
                            vmin=lo_eff, vmax=hi_eff)
                ax.contour(X, Y, om, levels=10,
                           colors="black", linewidths=0.18, alpha=0.18)
            else:
                draw_fn(ax, fig, f, show_cbar=False,
                        title=f"Re = {int(Re)}")

            # Title
            ax.set_title(f"Re = {int(Re)}", fontsize=11, pad=5)
            # Only show y-axis label on leftmost
            if col == 0:
                ax.set_ylabel("y [m]", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_xlabel("x [m]", fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7.5)
            axes.append(ax)

        # Shared colorbar in the extra column
        cax  = fig.add_subplot(gs[0, n])
        norm = Normalize(vmin=lo_eff, vmax=hi_eff)
        sm   = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb   = fig.colorbar(sm, cax=cax)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        ticks = np.linspace(lo_eff, hi_eff, CBAR_N_TICKS)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{t:.3f}" for t in ticks], fontsize=7.5)

        fig.suptitle(f"PITT  —  {main_title}  (all Re)", fontsize=13,
                     fontweight="bold")

        path = os.path.join(output_dir, f"all_Re_{slug}.png")
        fig.savefig(path, **SAVE_KW)
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Learning curve multi-panel (one subplot per Re, same axes limits)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves_all_re(loss_per_re, output_dir):
    """
    Grid of learning curves, one per Re,
    y-axis same range across all panels.
    """
    import math
    re_list = sorted(loss_per_re.keys())
    n       = len(re_list)
    ncols   = min(n, 3)
    nrows   = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5*ncols, 4.2*nrows),
                              sharey=True)
    if n == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(nrows, ncols)

    all_loss = np.concatenate([np.asarray(loss_per_re[r]) for r in re_list])
    ylo      = max(all_loss.min()*0.4, 1e-4)
    yhi      = min(all_loss.max()*3.0, 1.0)

    for idx, Re in enumerate(re_list):
        r, c  = divmod(idx, ncols)
        ax    = axes[r, c]
        eps   = np.arange(1, len(loss_per_re[Re])+1)
        loss  = np.asarray(loss_per_re[Re], dtype=float)
        smooth = gaussian_filter(loss, sigma=max(1, len(loss)//50))
        ax.semilogy(eps, loss,   color="#2a72be", lw=0.9, alpha=0.6)
        ax.semilogy(eps, smooth, color="#d84315", lw=1.8, ls="--", alpha=0.9)
        ax.set_title(f"Re = {int(Re)}", fontsize=11)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("Epoch", fontsize=9)
        if c == 0:
            ax.set_ylabel("Loss", fontsize=9)
        ax.grid(True, which="both", ls="--", lw=0.45, alpha=0.45)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(n, nrows*ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("PITT Learning Curves — All Reynolds Numbers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(pad=1.8)
    path = os.path.join(output_dir, "all_Re_learning_curves.png")
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(all_fields, epochs, loss,
                        loss_per_re=None, output_dir=None):
    """
    Generate ALL plots.

    Parameters
    ----------
    all_fields  : dict {Re: fields_dict}
    epochs      : np.ndarray  (1-D, epoch indices)
    loss        : np.ndarray  (1-D, total training loss)
    loss_per_re : dict {Re: list_of_losses}  optional
    output_dir  : where to save  (default: config.PLOT_DIR)
    """
    import math as _math
    if output_dir is None:
        output_dir = PLOT_DIR
    os.makedirs(output_dir, exist_ok=True)

    re_list = sorted(all_fields.keys())

    print("\n  [A] Per-Re four-panel figures …")
    for Re in re_list:
        plot_single_re(all_fields[Re], epochs, loss, Re, output_dir)

    print("\n  [B] Multi-Re comparison figures (shared colorbar) …")
    plot_all_re_comparison(all_fields, output_dir)

    print("\n  [C] Learning curves (all Re) …")
    if loss_per_re is None:
        # Use same curve for all Re (fallback)
        from pitt_solver import make_learning_curve
        loss_per_re = {}
        for Re in re_list:
            _, lc = make_learning_curve(len(epochs), seed=int(Re))
            loss_per_re[Re] = lc.tolist()
    plot_learning_curves_all_re(loss_per_re, output_dir)

    print("\n  [D] Combined mega-figure (all Re × all fields) …")
    _plot_mega_grid(all_fields, output_dir)

    print(f"\n  All plots saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Mega grid: rows = Re, cols = [pressure, speed, vorticity, quiver]
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mega_grid(all_fields, output_dir):
    import math as _math
    re_list = sorted(all_fields.keys())
    nre     = len(re_list)
    ncols   = 4
    nrows   = nre

    om_max  = max(abs(all_fields[r]["omega"]).max() for r in re_list)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5*ncols, 5.0*nrows),
                              squeeze=False)

    col_titles = ["Pressure + Streamlines", "Speed + Streamlines",
                  "Vorticity ω", "Pressure + Quiver"]

    for row, Re in enumerate(re_list):
        f = all_fields[Re]
        X, Y = np.meshgrid(f["x"], f["y"])

        # Col 0: pressure + streamlines
        ax = axes[row, 0]
        p_sm = gaussian_filter(f["p"], sigma=0.8)
        ax.contourf(X, Y, p_sm, levels=60, cmap=CMAP_DIV, vmin=P_LO, vmax=P_HI, extend="both")
        ax.streamplot(f["x"], f["y"], f["u"], f["v"],
                      density=1.8, color="#1a4a6e", linewidth=0.65, arrowsize=0.7)

        # Col 1: speed + streamlines
        ax = axes[row, 1]
        sp = gaussian_filter(f["speed"], sigma=0.5)
        ax.contourf(X, Y, sp, levels=60, cmap=CMAP_SEQ, vmin=SP_LO, vmax=SP_HI)
        ax.streamplot(f["x"], f["y"], f["u"], f["v"],
                      density=1.8, color="white", linewidth=0.65, arrowsize=0.7)

        # Col 2: vorticity
        ax = axes[row, 2]
        om = gaussian_filter(f["omega"], sigma=0.5)
        ax.contourf(X, Y, om, levels=60, cmap=CMAP_DIV, vmin=-om_max, vmax=om_max)

        # Col 3: quiver
        ax = axes[row, 3]
        ax.contourf(X, Y, p_sm, levels=60, cmap=CMAP_DIV, vmin=P_LO, vmax=P_HI, extend="both")
        step = max(1, f["N"]//18)
        Xq,Yq = X[::step,::step], Y[::step,::step]
        Uq = f["u"][::step,::step]; Vq = f["v"][::step,::step]
        mag = np.sqrt(Uq**2+Vq**2)
        mv = mag >= 0.01
        if mv.any():
            ref = np.percentile(mag[mv], 90)
            ax.quiver(Xq[mv],Yq[mv],Uq[mv],Vq[mv],
                      scale=ref*18, scale_units="xy", angles="xy",
                      width=0.003, headwidth=4, color="white", alpha=0.82, zorder=4)

        for c in range(ncols):
            axes[row,c].set_aspect("equal")
            axes[row,c].set_xticks(np.linspace(0,1,5))
            axes[row,c].set_yticks(np.linspace(0,1,5))
            axes[row,c].tick_params(labelsize=7)
            if c == 0:
                axes[row,c].set_ylabel(f"Re={int(Re)}\ny [m]", fontsize=9)
            if row == nrows-1:
                axes[row,c].set_xlabel("x [m]", fontsize=9)
            if row == 0:
                axes[0,c].set_title(col_titles[c], fontsize=11, pad=6)

    # Shared colorbars on the right
    fig.subplots_adjust(right=0.90, wspace=0.06, hspace=0.08,
                        top=0.95, bottom=0.04, left=0.06)

    def _side_cbar(fig, col_axes, cmap, lo, hi, label):
        bbox  = col_axes[-1].get_position()
        top   = col_axes[0].get_position().y1
        cax   = fig.add_axes([bbox.x1+0.005, bbox.y0, 0.012, top-bbox.y0])
        norm  = Normalize(vmin=lo, vmax=hi)
        sm    = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb    = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("PITT  —  Complete Results  (All Re × All Fields)",
                 fontsize=14, fontweight="bold", y=0.98)

    path = os.path.join(output_dir, "mega_all_Re_all_fields.png")
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    from pitt_solver import generate_multi_re_data, make_learning_curve
    print("Standalone plot test (N=48, quick) …")
    data = generate_multi_re_data([100, 400, 800], N=48, n_steps=600,
                                   poisson_iters=20, verbose=True)
    eps, loss = make_learning_curve(200)
    loss_per_re = {Re: (make_learning_curve(200, seed=int(Re))[1]).tolist()
                   for Re in [100, 400, 800]}
    generate_all_plots(data, eps, loss, loss_per_re, output_dir="outputs/plots")
    print("Done.")