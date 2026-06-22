"""
plot_from_npz.py
================
Standalone script: reads results.npz produced by pitt_main.py
and saves all 14 plots as individual PDF files.

Usage
-----
    python plot_from_npz.py                       # uses default path
    python plot_from_npz.py circle_heat/results.npz
    python plot_from_npz.py results.npz --out my_plots/

Outputs
-------
    <out_dir>/01_pressure_quiver.pdf
    <out_dir>/02_pressure_streamlines.pdf
    ...
    <out_dir>/14_T_vs_speed.pdf
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_pdf import PdfPages

# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Plot PITT results from NPZ file.")
    p.add_argument("npz", nargs="?", default="circle_heat/results.npz",
                   help="Path to results.npz (default: circle_heat/results.npz)")
    p.add_argument("--out", default=None,
                   help="Output directory for PDF files "
                        "(default: same directory as the NPZ file)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Load NPZ
# ══════════════════════════════════════════════════════════════
def load_npz(path):
    d = np.load(path)
    fields = {
        "u":       d["u"],
        "v":       d["v"],
        "p":       d["p"],
        "T":       d["T"],
        "obs":     d["obs_mask"].astype(bool),
        "history": d["history"],
    }
    meta = {
        "Re":            float(d["Re"]),
        "Pr":            float(d["Pr"]),
        "alpha":         float(d["alpha"]),
        "T_hot":         float(d["T_hot"]),
        "T_cold":        float(d["T_cold"]),
        "L_dom":         float(d["L_dom"]),
        "dx":            float(d["dx"]),
        "dy":            float(d["dy"]),
        "dt":            float(d["dt"]),
        "tri_cx":        float(d["tri_cx"]),
        "tri_half_base": float(d["tri_half_base"]),
        "tri_height":    float(d["tri_height"]),
        "grid_size":     int(d["grid_size"]),
    }
    return fields, meta


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def tri_verts(meta):
    cx, hb = meta["tri_cx"], meta["tri_half_base"]
    ht = meta["tri_height"]
    return np.array([
        [cx - hb, 0.0],
        [cx + hb, 0.0],
        [cx,      ht ],
    ])


def obs_patch(ax, verts, alpha=1.0):
    tri = MplPolygon(verts, closed=True,
                     facecolor='#d0d0d0', edgecolor='#444444',
                     linewidth=1.0, zorder=5, alpha=alpha)
    ax.add_patch(tri)


def axis_style(ax, xlabel='x', ylabel='y', l_dom=1.0):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, l_dom)
    ax.set_ylim(0, l_dom)


def add_cbar(fig, ax, cf, label):
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def save_pdf(fig, path):
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Individual plot functions
# ══════════════════════════════════════════════════════════════
def plot_01(X, Y, x_arr, y_arr, um, vm, pm, speed, verts, meta, skip, out):
    LEVELS_P = np.linspace(np.nanmin(pm), np.nanmax(pm), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='turbo', extend='both')
    ax.contour(X, Y, pm, levels=12, colors='k', linewidths=0.4, alpha=0.35)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              um[::skip, ::skip], vm[::skip, ::skip],
              color='white', scale=15, width=0.003, alpha=0.85, zorder=4)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'p')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "01_pressure_quiver.pdf"))
    print("Saved 01_pressure_quiver.pdf")


def plot_02(X, Y, x_arr, y_arr, um, vm, pm, speed, verts, meta, out):
    LEVELS_P = np.linspace(np.nanmin(pm), np.nanmax(pm), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='coolwarm', extend='both')
    ax.streamplot(x_arr, y_arr, um, vm,
                  color=speed, cmap='Greens', linewidth=1.2, density=1.5, arrowsize=1.0)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'p')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "02_pressure_streamlines.pdf"))
    print("Saved 02_pressure_streamlines.pdf")


def plot_03(X, Y, um, vm, verts, meta, out):
    u_vmin, u_vmax = np.nanmin(um), np.nanmax(um)
    LEVELS_U = np.linspace(u_vmin, u_vmax, 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, um, levels=LEVELS_U, cmap='RdBu_r', extend='both')
    ax.contour(X, Y, um, levels=[0], colors='k', linewidths=1.2)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'u')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "03_u_velocity.pdf"))
    print("Saved 03_u_velocity.pdf")


def plot_04(X, Y, um, vm, verts, meta, out):
    v_vmin, v_vmax = np.nanmin(vm), np.nanmax(vm)
    LEVELS_V = np.linspace(v_vmin, v_vmax, 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, vm, levels=LEVELS_V, cmap='RdBu_r', extend='both')
    ax.contour(X, Y, vm, levels=[0], colors='k', linewidths=1.2)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'v')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "04_v_velocity.pdf"))
    print("Saved 04_v_velocity.pdf")


def plot_05(X, Y, x_arr, y_arr, pm, verts, meta, out):
    DX, DY = meta["dx"], meta["dy"]
    dp_dx  = np.gradient(np.nan_to_num(pm), DX, axis=1)
    dp_dy  = np.gradient(np.nan_to_num(pm), DY, axis=0)
    grad_mag = np.sqrt(dp_dx**2 + dp_dy**2)
    LEVELS_P = np.linspace(np.nanmin(pm), np.nanmax(pm), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf   = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='RdBu_r', extend='both')
    strm = ax.streamplot(x_arr, y_arr, -dp_dx, -dp_dy,
                         color=grad_mag, cmap='hot', linewidth=1.0, density=1.5)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'p')
    cb2 = fig.colorbar(strm.lines, ax=ax, fraction=0.03, pad=0.12)
    cb2.set_label('|∇p|', fontsize=9); cb2.ax.tick_params(labelsize=8)
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "05_pressure_gradient.pdf"))
    print("Saved 05_pressure_gradient.pdf")


def plot_06(y_arr, um, meta, out):
    mid_x = meta["grid_size"] // 2
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
    ax.plot(um[:, mid_x], y_arr, 'b-', lw=2.5, label='PITT prediction')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.fill_betweenx(y_arr, 0, np.nan_to_num(um[:, mid_x]), alpha=0.15, color='blue')
    ax.axhspan(0, meta["tri_height"], alpha=0.10, color='gray', label='obstacle region')
    ax.set_xlabel('u', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "06_u_profile_x05.pdf"))
    print("Saved 06_u_profile_x05.pdf")


def plot_07(x_arr, vm, meta, out):
    mid_y = meta["grid_size"] // 2
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    ax.plot(x_arr, vm[mid_y, :], 'r-', lw=2.5, label='PITT prediction')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.fill_between(x_arr, 0, np.nan_to_num(vm[mid_y, :]), alpha=0.15, color='red')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('v', fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "07_v_profile_y05.pdf"))
    print("Saved 07_v_profile_y05.pdf")


def plot_08(X, Y, x_arr, y_arr, pm, verts, meta, out):
    p_vmin, p_vmax = np.nanmin(pm), np.nanmax(pm)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=40, cmap='turbo', extend='both')
    ax.contour(X, Y, pm, levels=14, colors='white', linewidths=0.7, alpha=0.6)
    pf = np.nan_to_num(pm)
    pi = np.unravel_index(np.argmax(pf), pf.shape)
    pj = np.unravel_index(np.argmin(pf), pf.shape)
    ax.plot(x_arr[pi[1]], y_arr[pi[0]], 'w^', ms=10, zorder=7,
            label=f'p_max = {pf.max():.4f}')
    ax.plot(x_arr[pj[1]], y_arr[pj[0]], 'wv', ms=10, zorder=7,
            label=f'p_min = {pf.min():.4f}')
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'p')
    ax.legend(fontsize=9, loc='lower left', facecolor='#1a1a1a',
              labelcolor='white', framealpha=0.85)
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "08_pressure_map.pdf"))
    print("Saved 08_pressure_map.pdf")


def plot_09(history, out):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    epochs_arr = np.arange(1, len(history) + 1)
    ax.semilogy(epochs_arr, history, color='#1a5fa8', lw=1.8, label='Total Loss')
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.grid(which='major', alpha=0.35); ax.grid(which='minor', alpha=0.12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "09_learning_curve.pdf"))
    print("Saved 09_learning_curve.pdf")


def plot_10(X, Y, x_arr, y_arr, um, vm, Tm, speed, verts, meta, skip, out):
    LEVELS_T = np.linspace(np.nanmin(Tm), np.nanmax(Tm), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, Tm, levels=LEVELS_T, cmap='hot', extend='both')
    ax.contour(X, Y, Tm, levels=12, colors='k', linewidths=0.4, alpha=0.35)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              um[::skip, ::skip], vm[::skip, ::skip],
              color='white', scale=15, width=0.003, alpha=0.85, zorder=4)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'T')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "10_temperature_quiver.pdf"))
    print("Saved 10_temperature_quiver.pdf")


def plot_11(X, Y, x_arr, y_arr, um, vm, Tm, speed, verts, meta, out):
    LEVELS_T = np.linspace(np.nanmin(Tm), np.nanmax(Tm), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, Tm, levels=LEVELS_T, cmap='RdYlBu_r', extend='both')
    ax.streamplot(x_arr, y_arr, um, vm,
                  color=speed, cmap='Greens', linewidth=1.2, density=1.5, arrowsize=1.0)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, 'T')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "11_temperature_streamlines.pdf"))
    print("Saved 11_temperature_streamlines.pdf")


def plot_12(X, Y, x_arr, y_arr, Tm, obs, verts, meta, out):
    DX, DY = meta["dx"], meta["dy"]
    dT_dx  = np.gradient(np.nan_to_num(Tm), DX, axis=1)
    dT_dy  = np.gradient(np.nan_to_num(Tm), DY, axis=0)
    grad_T = np.sqrt(dT_dx**2 + dT_dy**2)
    grad_T = np.where(obs, np.nan, grad_T)
    gT_levels = np.linspace(np.nanmin(grad_T), np.nanmax(grad_T), 30)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf   = ax.contourf(X, Y, grad_T, levels=gT_levels, cmap='inferno', extend='both')
    strm = ax.streamplot(x_arr, y_arr, -dT_dx, -dT_dy,
                         color=grad_T, cmap='cool', linewidth=0.9, density=1.2)
    obs_patch(ax, verts)
    add_cbar(fig, ax, cf, '|∇T|')
    axis_style(ax, l_dom=meta["L_dom"])
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "12_temperature_gradient.pdf"))
    print("Saved 12_temperature_gradient.pdf")


def plot_13(y_arr, Tm, meta, out):
    mid_x  = meta["grid_size"] // 2
    T_cold = meta["T_cold"]
    T_hot  = meta["T_hot"]
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
    ax.plot(np.nan_to_num(Tm[:, mid_x]), y_arr, 'm-', lw=2.5, label='PITT prediction')
    ax.axvline(T_cold, color='blue', lw=0.8, ls='--', label=f'T_cold={T_cold}')
    ax.axvline(T_hot,  color='red',  lw=0.8, ls='--', label=f'T_hot={T_hot}')
    ax.fill_betweenx(y_arr, T_cold, np.nan_to_num(Tm[:, mid_x]),
                     alpha=0.12, color='magenta')
    ax.axhspan(0, meta["tri_height"], alpha=0.10, color='gray', label='obstacle region')
    ax.set_xlabel('T', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "13_T_profile_x05.pdf"))
    print("Saved 13_T_profile_x05.pdf")


def plot_14(Tm, speed, out):
    T_flat = Tm.ravel()
    s_flat = speed.ravel()
    valid  = ~(np.isnan(T_flat) | np.isnan(s_flat))
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    sc = ax.scatter(T_flat[valid][::5], s_flat[valid][::5],
                    c=T_flat[valid][::5], cmap='hot', s=2, alpha=0.6)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('T', fontsize=9); cb.ax.tick_params(labelsize=8)
    ax.set_xlabel('T', fontsize=10)
    ax.set_ylabel('Speed', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_pdf(fig, os.path.join(out, "14_T_vs_speed.pdf"))
    print("Saved 14_T_vs_speed.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    npz_path = args.npz

    if not os.path.isfile(npz_path):
        sys.exit(f"ERROR: NPZ file not found: {npz_path}")

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(npz_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {npz_path}")
    fields, meta = load_npz(npz_path)

    u_raw = fields["u"]
    v_raw = fields["v"]
    p_raw = fields["p"]
    T_raw = fields["T"]
    obs   = fields["obs"]
    hist  = fields["history"]

    GS      = meta["grid_size"]
    L_dom   = meta["L_dom"]
    x_arr   = np.linspace(0, L_dom, GS)
    y_arr   = np.linspace(0, L_dom, GS)
    X, Y    = np.meshgrid(x_arr, y_arr)

    um    = np.where(obs, np.nan, u_raw)
    vm    = np.where(obs, np.nan, v_raw)
    pm    = np.where(obs, np.nan, p_raw)
    Tm    = np.where(obs, np.nan, T_raw)
    speed = np.where(obs, np.nan, np.sqrt(u_raw**2 + v_raw**2))

    verts = tri_verts(meta)
    skip  = max(1, GS // 20)

    print(f"Output directory: {out_dir}\n")

    plot_01(X, Y, x_arr, y_arr, um, vm, pm, speed, verts, meta, skip, out_dir)
    plot_02(X, Y, x_arr, y_arr, um, vm, pm, speed, verts, meta,       out_dir)
    plot_03(X, Y, um, vm, verts, meta, out_dir)
    plot_04(X, Y, um, vm, verts, meta, out_dir)
    plot_05(X, Y, x_arr, y_arr, pm, verts, meta, out_dir)
    plot_06(y_arr, um, meta, out_dir)
    plot_07(x_arr, vm, meta, out_dir)
    plot_08(X, Y, x_arr, y_arr, pm, verts, meta, out_dir)
    plot_09(hist, out_dir)
    plot_10(X, Y, x_arr, y_arr, um, vm, Tm, speed, verts, meta, skip, out_dir)
    plot_11(X, Y, x_arr, y_arr, um, vm, Tm, speed, verts, meta,       out_dir)
    plot_12(X, Y, x_arr, y_arr, Tm, obs, verts, meta, out_dir)
    plot_13(y_arr, Tm, meta, out_dir)
    plot_14(Tm, speed, out_dir)

    print(f"\nAll 14 PDFs saved to: {out_dir}")


if __name__ == "__main__":
    main()