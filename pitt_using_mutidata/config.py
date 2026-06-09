"""
config.py  —  Central Configuration for PITT-v2
================================================
Every hyper-parameter, Re list, colormap choice, and path is
defined ONCE here. All other files import from this module.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Device  (auto GPU → CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Physics / CFD
# ─────────────────────────────────────────────────────────────────────────────
RE_LIST      = [100, 400, 800, 1600, 3200]   # Multi-Re curriculum
N_GRID       = 128                            # Spatial grid NxN
N_CFD_STEPS  = 10_000                         # CFD solver iterations per Re
POISSON_ITER = 50                             # Jacobi pressure iters per step

# ─────────────────────────────────────────────────────────────────────────────
# PITT Model
# ─────────────────────────────────────────────────────────────────────────────
C_IN         = 4      # channels: u, v, p, ω
C_OUT        = 4
FNO_MODES    = 16     # Fourier modes to keep (x and y)
FNO_WIDTH    = 32     # latent channel width
FNO_LAYERS   = 4
D_MODEL      = 128    # Transformer embedding dimension
N_HEADS      = 4
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF         = 256    # feed-forward hidden size
DROPOUT      = 0.1

# Novel components
USE_ADAPTIVE_TOKENS   = True   # adaptive PDE token weighting
USE_PHYSICS_ATTENTION = True   # physics-constrained attention mask
USE_SPECTRAL_LOSS     = True   # Fourier-space loss term
USE_CURL_LOSS         = True   # vorticity / curl-preservation loss
USE_DIV_LOSS          = True   # divergence-free constraint

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4
LR           = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 60
GRAD_CLIP    = 1.0
AMP          = True   # Automatic Mixed Precision (fp16, GPU only)

# Loss weights
W_DATA  = 1.0
W_VORT  = 0.6
W_DIV   = 0.5
W_FFT   = 0.4
W_CURL  = 0.3
W_CONS  = 0.2   # kinetic-energy conservation

# ─────────────────────────────────────────────────────────────────────────────
# Colormap  (shared across ALL figures — task requirement)
# ─────────────────────────────────────────────────────────────────────────────
SHARED_CMAP  = "RdBu_r"   # diverging, blue→white→red
SHARED_CMAP2 = "viridis"  # sequential  (speed magnitude, positive only)
CBAR_N_TICKS = 9

# Pressure field colour limits
P_LO  = -0.20
P_HI  =  0.55
# Speed colour limits
SP_LO =  0.00
SP_HI =  1.00

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR      = "outputs"
CKPT_DIR     = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR     = os.path.join(OUT_DIR, "plots")
DATA_DIR     = os.path.join(OUT_DIR, "cfd_data")

for _d in [OUT_DIR, CKPT_DIR, PLOT_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Quick-run overrides  (used by  main.py --quick)
# ─────────────────────────────────────────────────────────────────────────────
QUICK = dict(
    RE_LIST      = [100, 400, 800],
    N_GRID       = 48,
    N_CFD_STEPS  = 600,
    POISSON_ITER = 20,
    EPOCHS       = 8,
    BATCH_SIZE   = 2,
)
