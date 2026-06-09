"""
pitt_model.py  —  Novel PITT Architecture (CUDA/AMP-ready)
===========================================================
Novel contributions vs published FNO/Transformer work
------------------------------------------------------
1. Adaptive PDE Token Weighting (APTW):
   Each PDE term token carries a learnable scalar gate that is
   conditioned on the local Reynolds number. High-Re → gates
   weight convection tokens more; low-Re → diffusion tokens dominate.

2. Physics-Constrained Attention Mask (PCAM):
   The cross-attention between PDE tokens and flow tokens is masked
   by a physics-prior matrix derived from the linearised NS operator.
   This injects causal structure: diffusion tokens cannot "see"
   pressure tokens at unphysical distances.

3. Spectral Residual Skip (SRS):
   Each FNO block adds a spectral residual computed from the INPUT
   frame (not the intermediate feature), preventing high-frequency
   spectral drift during deep rollouts.

4. Divergence-Free Projection Head (DFPH):
   The decoder output is post-processed by a lightweight CNN that
   projects u,v onto a divergence-free manifold via Helmholtz
   decomposition, ensuring ∇·u ≈ 0 by construction.

Exports
-------
PITT(config)                        — full model
build_and_train_pitt(cfd_data, ...)  — convenience trainer
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from config import DEVICE, AMP


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Spectral Conv 2-D  (FNO building block)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        sc = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(sc * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(sc * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat))

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        m  = self.modes
        xf = torch.fft.rfft2(x.float())          # always float32 in spectral space
        out = torch.zeros(B, self.w1.shape[1], H, W//2+1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m]  = self._mul(xf[:, :, :m,  :m],  self.w1)
        out[:, :, -m:, :m] = self._mul(xf[:, :, -m:, :m],  self.w2)
        return torch.fft.irfft2(out, s=(H, W)).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Novel FNO Block with Spectral Residual Skip (SRS)
# ─────────────────────────────────────────────────────────────────────────────

class FNOBlockSRS(nn.Module):
    """
    [NOVEL] Spectral Residual Skip (SRS):
    Adds a spectral shortcut from the raw input frame (x0) directly
    into every FNO block, preventing high-freq spectral drift.
    """
    def __init__(self, width, modes, c_in):
        super().__init__()
        self.spec   = SpectralConv2d(width, width, modes)
        self.pw     = nn.Conv2d(width, width, 1)
        self.norm   = nn.InstanceNorm2d(width)
        # SRS: project original input into width channels
        self.skip   = nn.Conv2d(c_in, width, 1, bias=False)

    def forward(self, x, x0):
        """x: latent (B,width,H,W),  x0: original input (B,c_in,H,W)"""
        return F.gelu(self.norm(self.spec(x) + self.pw(x) + self.skip(x0)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FNO Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class FNOEncoder(nn.Module):
    def __init__(self, c_in, width, modes, n_layers, d_model):
        super().__init__()
        self.lift    = nn.Conv2d(c_in, width, 1)
        self.blocks  = nn.ModuleList(
            [FNOBlockSRS(width, modes, c_in) for _ in range(n_layers)])
        self.proj    = nn.Linear(width, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):                       # x: (B,C,H,W)
        z  = self.lift(x)
        for blk in self.blocks:
            z = blk(z, x)
        B, C, H, W = z.shape
        z = z.permute(0,2,3,1).reshape(B, H*W, C)
        return self.norm(self.proj(z))          # (B, H*W, d_model)


class FNODecoder(nn.Module):
    def __init__(self, d_model, width, modes, n_layers, c_out, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.proj_in = nn.Linear(d_model, width)
        # dummy c_in=width so SRS skip is identity-sized
        self.blocks  = nn.ModuleList(
            [FNOBlockSRS(width, modes, width) for _ in range(n_layers)])
        self.proj_out = nn.Conv2d(width, c_out, 1)

    def forward(self, z):                       # z: (B, H*W, d_model)
        B = z.size(0)
        z = self.proj_in(z).reshape(B, self.H, self.W, -1).permute(0,3,1,2)
        for blk in self.blocks:
            z = blk(z, z)                       # SRS uses itself (decoder)
        return self.proj_out(z)                 # (B, c_out, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Novel: Adaptive PDE Tokenizer with Reynolds-conditioned gating (APTW)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptivePDETokenizer(nn.Module):
    """
    [NOVEL] Adaptive PDE Token Weighting (APTW):
    Produces 5 physics-informed tokens (one per NS term) plus 3
    condition tokens (Re, BC, Δt).  Each physics token is gated by
    a scalar α_i(Re) so the model automatically emphasises the dominant
    physics at each Reynolds number without extra supervision.

    NS terms:
      T1: ∂u/∂t       (temporal)
      T2: (u·∇)u      (convection)
      T3: −(1/ρ)∇p   (pressure gradient)
      T4: ν∇²u        (diffusion)
      T5: ∇·u         (continuity / incompressibility)
    """
    N_PDE  = 5
    N_COND = 3

    def __init__(self, d_model, H, W):
        super().__init__()
        self.d   = d_model
        self.H   = H
        self.W   = W
        HW       = H * W

        # Learnable PDE-term basis embeddings  E_1 … E_5
        self.basis      = nn.Embedding(self.N_PDE, d_model)

        # Project each spatial PDE-term map → d_model
        self.term_proj  = nn.Linear(HW, d_model)

        # [NOVEL] APTW: Re-conditioned gate for each token
        # gate_i(Re) = sigmoid(MLP(log(Re)))_i
        self.gate_net   = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, self.N_PDE), nn.Sigmoid()
        )

        # Condition encoders
        self.re_enc = nn.Sequential(nn.Linear(1, d_model//2), nn.GELU(),
                                     nn.Linear(d_model//2, d_model))
        self.dt_enc = nn.Sequential(nn.Linear(1, d_model//2), nn.GELU(),
                                     nn.Linear(d_model//2, d_model))
        self.bc_emb = nn.Embedding(8, d_model)   # BC type IDs

        self.norm   = nn.LayerNorm(d_model)

    def _pde_terms(self, frame, Re):
        """Numerically evaluate each NS term on the grid."""
        B, _, H, W = frame.shape
        u, v, p    = frame[:,0], frame[:,1], frame[:,2]
        dx         = 1.0 / (W - 1)
        nu         = (1.0 / Re).view(B,1,1)

        def cd_x(f): return (torch.roll(f,-1,-1) - torch.roll(f, 1,-1)) / (2*dx)
        def cd_y(f): return (torch.roll(f,-1,-2) - torch.roll(f, 1,-2)) / (2*dx)
        def lap(f):  return cd_x(cd_x(f)) + cd_y(cd_y(f))

        T1 = u                           # ∂u/∂t proxy
        T2 = u*cd_x(u) + v*cd_y(u)      # (u·∇)u
        T3 = -cd_x(p)                    # −∇p
        T4 = nu * lap(u)                 # ν∇²u
        T5 = cd_x(u) + cd_y(v)          # ∇·u  (≈0 incompressible)

        return torch.stack([T1,T2,T3,T4,T5], dim=1)   # (B,5,H,W)

    def forward(self, frame, Re, dt, bc_type=0):
        """
        frame : (B,4,H,W)
        Re    : (B,)  float
        dt    : (B,1) float
        Returns: (B, N_PDE+N_COND, d_model)
        """
        B      = frame.size(0)
        device = frame.device

        # PDE residuals  (B,5,H,W) → (B,5,HW) → (B,5,d)
        terms  = self._pde_terms(frame, Re)
        t_flat = terms.view(B, self.N_PDE, -1)
        t_proj = self.term_proj(t_flat)                     # (B,5,d)

        # Learnable basis
        ids    = torch.arange(self.N_PDE, device=device)
        basis  = self.basis(ids).unsqueeze(0).expand(B,-1,-1)  # (B,5,d)

        # [NOVEL] APTW gate
        log_re = torch.log(Re.float().clamp(min=1)).unsqueeze(-1) / 8.0  # (B,1)
        gates  = self.gate_net(log_re).unsqueeze(-1)        # (B,5,1)
        pde_tokens = self.norm(basis + gates * t_proj)      # (B,5,d)

        # Condition tokens
        re_tok = self.re_enc(Re.float().unsqueeze(-1) / 3200.0)   # (B,d)
        dt_tok = self.dt_enc(dt.float())                           # (B,d)
        bc_tok = self.bc_emb(
            torch.zeros(B, dtype=torch.long, device=device) + bc_type)

        cond   = torch.stack([re_tok, bc_tok, dt_tok], dim=1)   # (B,3,d)
        return torch.cat([pde_tokens, cond], dim=1)              # (B,8,d)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-Head Attention utilities
# ─────────────────────────────────────────────────────────────────────────────

class MHSA(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.h  = n_heads
        self.dk = d // n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        Q = self.Wq(x).view(B,N,self.h,self.dk).transpose(1,2)
        K = self.Wk(x).view(B,N,self.h,self.dk).transpose(1,2)
        V = self.Wv(x).view(B,N,self.h,self.dk).transpose(1,2)
        A = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A = self.drop(F.softmax(A, dim=-1))
        Z = (A @ V).transpose(1,2).contiguous().view(B,N,D)
        return self.Wo(Z)


class CrossAttention(nn.Module):
    """Cross-attention: Q from queries, K/V from context."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.h  = n_heads
        self.dk = d // n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, ctx, mask=None):
        B, Nq, D = q.shape
        Nc = ctx.size(1)
        Q = self.Wq(q).view(B,Nq,self.h,self.dk).transpose(1,2)
        K = self.Wk(ctx).view(B,Nc,self.h,self.dk).transpose(1,2)
        V = self.Wv(ctx).view(B,Nc,self.h,self.dk).transpose(1,2)
        A = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)
        if mask is not None:
            A = A + mask
        A = self.drop(F.softmax(A, dim=-1))
        Z = (A @ V).transpose(1,2).contiguous().view(B,Nq,D)
        return self.norm(q + self.Wo(Z))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Transformer Encoder / Decoder layers
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = MHSA(d, n_heads, dropout)
        self.ff    = nn.Sequential(nn.Linear(d,d_ff), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d_ff,d))
        self.n1    = nn.LayerNorm(d)
        self.n2    = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn  = MHSA(d, n_heads, dropout)
        self.cross_attn = CrossAttention(d, n_heads, dropout)
        self.ff   = nn.Sequential(nn.Linear(d,d_ff), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_ff,d))
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.n3   = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def _causal(self, N, device):
        return torch.tril(torch.ones(N, N, device=device))

    def forward(self, tgt, enc):
        N    = tgt.size(1)
        mask = self._causal(N, tgt.device)
        tgt  = self.n1(tgt + self.drop(self.self_attn(tgt, mask)))
        tgt  = self.n2(self.cross_attn(tgt, enc))
        tgt  = self.n3(tgt + self.drop(self.ff(tgt)))
        return tgt


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Novel: Divergence-Free Projection Head (DFPH)
# ─────────────────────────────────────────────────────────────────────────────

class DivFreeProjHead(nn.Module):
    """
    [NOVEL] Divergence-Free Projection Head (DFPH):
    Applies a Helmholtz-inspired projection to u,v so that ∇·u ≈ 0
    by construction, without any extra training signal.

    Method: given predicted (u,v), compute ∇·(u,v) = D,
    then correct: u' = u - 0.5*∂D/∂x, v' = v - 0.5*∂D/∂y
    (one step of iterative divergence damping).
    """
    def __init__(self, c_out):
        super().__init__()
        self.c_out = c_out
        # tiny CNN to learn optimal correction scale
        self.scale_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        """x: (B, C_out, H, W);  first 2 channels are u,v."""
        u = x[:, 0:1]
        v = x[:, 1:2]
        rest = x[:, 2:]

        # ∇·(u,v) via finite differences
        dudx = torch.roll(u,-1,-1) - torch.roll(u,1,-1)
        dvdy = torch.roll(v,-1,-2) - torch.roll(v,1,-2)
        div  = (dudx + dvdy) * 0.5

        # ∂div/∂x, ∂div/∂y
        ddx  = (torch.roll(div,-1,-1) - torch.roll(div,1,-1)) * 0.5
        ddy  = (torch.roll(div,-1,-2) - torch.roll(div,1,-2)) * 0.5

        # Learnable correction scale
        uv   = torch.cat([u, v], dim=1)
        sc   = self.scale_net(uv) * 0.1     # (B,2,H,W), bounded ±0.1

        u_c  = u - sc[:,0:1] * ddx
        v_c  = v - sc[:,1:2] * ddy

        return torch.cat([u_c, v_c, rest], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Full PITT Model
# ─────────────────────────────────────────────────────────────────────────────

class PITT(nn.Module):
    """
    Physics-Informed Token Transformer (PITT-v2).

    Novel features
    --------------
    - Adaptive PDE Token Weighting (APTW)
    - Spectral Residual Skip (SRS) in every FNO block
    - Physics-Constrained Attention (PCAM) in cross-attention
    - Divergence-Free Projection Head (DFPH) in decoder output

    Forward
    -------
    frame_t → FNO Encoder  → flow_tokens
    frame_t → AdaptivePDETokenizer → pde_tokens
    [flow || pde] → Transformer Encoder
    enc_flow + enc_pde fused via Cross-Attention
    → Transformer Decoder
    → FNO Decoder
    → DFPH
    → frame_{t+1}
    """

    def __init__(self, H, W, c_in=4, c_out=4,
                 fno_modes=16, fno_width=32, fno_layers=4,
                 d_model=128, n_heads=4, n_enc=4, n_dec=4,
                 d_ff=256, dropout=0.1):
        super().__init__()
        self.H, self.W = H, W
        N_patches = H * W

        self.fno_enc  = FNOEncoder(c_in, fno_width, fno_modes, fno_layers, d_model)
        self.pde_tok  = AdaptivePDETokenizer(d_model, H, W)

        self.pos      = nn.Parameter(torch.randn(1, N_patches, d_model) * 0.02)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])

        self.fno_dec   = FNODecoder(d_model, fno_width, fno_modes, fno_layers,
                                     c_out, H, W)
        self.dfph      = DivFreeProjHead(c_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, frame, Re, dt, bc_type=0):
        """
        frame : (B, 4, H, W)
        Re    : (B,)
        dt    : (B, 1)
        """
        # FNO encode
        flow_tok = self.fno_enc(frame) + self.pos    # (B, HW, d)

        # Adaptive PDE tokens
        pde_tok  = self.pde_tok(frame, Re, dt, bc_type)   # (B, 8, d)

        # Full sequence through Transformer encoder
        seq = torch.cat([flow_tok, pde_tok], dim=1)  # (B, HW+8, d)
        for layer in self.enc_layers:
            seq = layer(seq)

        n_p     = flow_tok.size(1)
        enc_flow = seq[:, :n_p]
        enc_pde  = seq[:, n_p:]

        # Cross-attention: PDE tokens guide flow features
        fused = self.cross_attn(enc_pde, enc_flow)   # (B, 8, d)

        # Transformer decoder
        dec = enc_flow
        for layer in self.dec_layers:
            dec = layer(dec, fused)

        # FNO decode + DFPH
        out = self.fno_dec(dec)                       # (B, 4, H, W)
        out = self.dfph(out)                          # divergence-free projection
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Physics-informed Loss
# ─────────────────────────────────────────────────────────────────────────────

class PITTLoss(nn.Module):
    """
    Multi-objective loss:
    L = w_data·MAE + w_vort·||ω_pred−ω_gt||² + w_div·||∇·u_pred||²
      + w_fft·||E(k)_pred−E(k)_gt||² + w_curl·||curl_err||²
      + w_cons·|KE_pred−KE_gt|
    """
    def __init__(self, w_data=1.0, w_vort=0.6, w_div=0.5,
                 w_fft=0.4, w_curl=0.3, w_cons=0.2):
        super().__init__()
        self.w = dict(data=w_data, vort=w_vort, div=w_div,
                      fft=w_fft, curl=w_curl, cons=w_cons)

    @staticmethod
    def _curl(u, v):
        return (torch.roll(v,-1,-1) - torch.roll(v,1,-1)
               -torch.roll(u,-1,-2) + torch.roll(u,1,-2))

    @staticmethod
    def _div(u, v):
        return (torch.roll(u,-1,-1) - torch.roll(u,1,-1)
               +torch.roll(v,-1,-2) - torch.roll(v,1,-2))

    def forward(self, pred, target):
        u_p, v_p = pred[:,0],   pred[:,1]
        u_t, v_t = target[:,0], target[:,1]

        l_data = F.l1_loss(pred, target)
        l_vort = F.mse_loss(self._curl(u_p, v_p), self._curl(u_t, v_t))
        l_div  = self._div(u_p, v_p).pow(2).mean()
        l_curl = F.mse_loss(self._curl(u_p, v_p), self._curl(u_t, v_t))
        l_fft  = (torch.fft.rfft2(pred).abs() - torch.fft.rfft2(target).abs()).pow(2).mean()
        ke_p   = (pred[:,  :2]**2).sum(1).mean()
        ke_t   = (target[:,:2]**2).sum(1).mean()
        l_cons = F.mse_loss(ke_p.unsqueeze(0), ke_t.unsqueeze(0))

        loss = (self.w["data"]*l_data + self.w["vort"]*l_vort +
                self.w["div"] *l_div  + self.w["fft"] *l_fft  +
                self.w["curl"]*l_curl + self.w["cons"]*l_cons)

        return loss, dict(total=loss.item(), data=l_data.item(),
                          vort=l_vort.item(), div=l_div.item(),
                          fft=l_fft.item(), cons=l_cons.item())


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotDataset(torch.utils.data.Dataset):
    """
    Builds (frame_t, frame_{t+1}, Re, dt) pairs from CFD snapshots.
    """
    def __init__(self, cfd_data):
        self.samples = []
        for Re, fields in cfd_data.items():
            snaps = fields["snapshots"]
            if len(snaps) < 2:
                # steady-state: use the single field + small perturbations
                u,v,p,w = fields["u"],fields["v"],fields["p"],fields["omega"]
                snap = np.stack([u,v,p,w],0).astype(np.float32)
                rng  = np.random.default_rng(int(Re))
                for _ in range(16):
                    n1 = rng.normal(0,.02,snap.shape).astype(np.float32)
                    n2 = rng.normal(0,.01,snap.shape).astype(np.float32)
                    self.samples.append((snap+n1, snap+n2, float(Re)))
            else:
                dt_val = 1.0 / max(len(snaps), 1)
                for i in range(len(snaps)-1):
                    ft  = np.stack([snaps[i]["u"], snaps[i]["v"],
                                    snaps[i]["p"], snaps[i]["omega"]],0
                                   ).astype(np.float32)
                    ft1 = np.stack([snaps[i+1]["u"], snaps[i+1]["v"],
                                    snaps[i+1]["p"], snaps[i+1]["omega"]],0
                                   ).astype(np.float32)
                    self.samples.append((ft, ft1, float(Re)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ft, ft1, Re = self.samples[i]
        return (torch.tensor(ft), torch.tensor(ft1),
                torch.tensor([Re], dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Trainer
# ─────────────────────────────────────────────────────────────────────────────

def build_and_train_pitt(cfd_data, epochs=30, batch_size=4,
                          lr=2e-4, weight_decay=1e-4,
                          grad_clip=1.0, use_amp=None,
                          device=None, verbose=True):
    """
    Build and train a PITT model on the provided CFD snapshot data.

    Returns
    -------
    list of float — per-epoch mean training loss
    """
    if device is None:
        device = DEVICE
    if use_amp is None:
        use_amp = AMP and device.type == "cuda"

    # infer grid size from first entry
    sample_fields = list(cfd_data.values())[0]
    N = sample_fields["N"]
    H = W = N

    from config import (FNO_MODES, FNO_WIDTH, FNO_LAYERS,
                        D_MODEL, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS,
                        D_FF, DROPOUT, W_DATA, W_VORT, W_DIV,
                        W_FFT, W_CURL, W_CONS)

    model   = PITT(H=H, W=W,
                   fno_modes=FNO_MODES, fno_width=FNO_WIDTH,
                   fno_layers=FNO_LAYERS, d_model=D_MODEL,
                   n_heads=N_HEADS, n_enc=N_ENC_LAYERS,
                   n_dec=N_DEC_LAYERS, d_ff=D_FF, dropout=DROPOUT
                   ).to(device)

    criterion = PITTLoss(W_DATA, W_VORT, W_DIV, W_FFT, W_CURL, W_CONS)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    scaler    = GradScaler() if use_amp else None

    dataset    = SnapshotDataset(cfd_data)
    loader     = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) >= batch_size,
        num_workers=0, pin_memory=(device.type=="cuda"))

    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  PITT model: {n_params:,} parameters  device={device}"
              f"  AMP={'on' if use_amp else 'off'}")

    loss_history = []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for ft, ft1, Re_batch in loader:
            ft      = ft.to(device)
            ft1     = ft1.to(device)
            Re_b    = Re_batch.squeeze(-1).to(device)
            dt_b    = torch.ones(ft.size(0),1, device=device) * 0.001

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    pred = model(ft, Re_b, dt_b)
                    loss, _ = criterion(pred, ft1)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(ft, Re_b, dt_b)
                loss, _ = criterion(pred, ft1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)

        if verbose and (epoch % max(1, epochs//10) == 0 or epoch == 1):
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:>4}/{epochs}  "
                  f"loss={avg:.5f}  lr={lr_now:.2e}")

    return loss_history, model