"""
Physics Informed Token Transformer (PITT) for Burgers' Equation
================================================================
Full implementation following the methodology from the paper:
"Physics Informed Token Transformer for Solving Partial Differential Equations"

Burgers' Equation:
    ∂u/∂t + α * ∂(u²)/∂x - β * ∂²u/∂x² = δ(t,x)
    where δ(t,x) = Σ Aj * sin(ωj*t + 2π*lj*x/L + ϕj)

Usage:
    python pitt_burgers_cuda.py
    # Outputs saved to ./pitt_outputs/
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless / no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# OUTPUT DIRECTORY  (local, always writable)
# ─────────────────────────────────────────────
OUT_DIR = './burgers_invis_output'
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EQUATION TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class BurgersTokenizer:
    """
    Tokenizes Burgers' equation following Table 3.1 from the paper.

    Burgers' equation: ∂u/∂t + α*∂(u²)/∂x - β*∂²u/∂x² = δ(t,x)
    In the paper's unified form (eq 3.1): α≠0, β≠0, γ=0

    Each token gets an integer index from the master vocabulary list.
    """

    VOCAB = [
        # 0-17: Equation tokens
        '(', ')', 'partial', 'sum', 'j', 'Aj', 'lj', 'wj', 'phij',
        'sin', 't', 'u', 'x', 'y', '+', '-', '*', '/',
        # 18-20: Boundary condition tokens
        'Neumann', 'Dirichlet', 'None',
        # 21-33: Numerical digit tokens
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^', 'E', 'e',
        # 34-35: Delimiter tokens
        ',', '.',
        # 36: Separator token
        '&',
        # 37-41: Equation structure tokens
        'Derivative', '=', 'Delta', 'dot', 'nabla',
        # 42-46: Physics-specific tokens
        'alpha', 'beta', 'gamma', 'delta', 'nu',
        # 47-51: Additional tokens
        'PAD', 'UNK', 'dt', 'dx', 'dx2',
    ]

    def __init__(self, max_len=100):
        self.max_len   = max_len
        self.token2idx = {tok: i for i, tok in enumerate(self.VOCAB)}
        self.pad_idx   = self.token2idx['PAD']
        self.vocab_size = len(self.VOCAB)

    def _num_to_tokens(self, value):
        """Convert a float to a sequence of character tokens."""
        s = f"{value:.6f}"
        tokens = []
        for ch in s:
            if ch.isdigit():    tokens.append(ch)
            elif ch == '.':     tokens.append('.')
            elif ch == '-':     tokens.append('-')
            elif ch in ('e','E'): tokens.append('E')
        return tokens

    def tokenize(self, alpha, beta, forcing_params,
                 boundary='Dirichlet', target_time=1.0):
        """
        Tokenize one Burgers' equation instance.
        Structure: governing eq & forcing params & BC & target_time
        """
        tokens = []

        # ── Governing equation: ∂u/∂t + α∂(u²)/∂x - β∂²u/∂x² = δ(t,x) ──
        tokens += ['Derivative','(','u','(','x',',','t',')',',','t',')']
        tokens += ['+']
        tokens += self._num_to_tokens(alpha)
        tokens += ['*','Derivative','(','u','*','u',',','x',')']
        tokens += ['-']
        tokens += self._num_to_tokens(beta)
        tokens += ['*','Derivative','(','Derivative','(','u',',','x',')',',','x',')']
        tokens += ['=','delta','(','t',',','x',')']
        tokens += ['&']

        # ── Forcing term parameters ──
        for (Aj, wj, lj, phij) in forcing_params:
            tokens += self._num_to_tokens(Aj)  + [',']
            tokens += self._num_to_tokens(wj)  + [',']
            tokens += [str(int(lj)),             ',']
            tokens += self._num_to_tokens(phij) + ['&']

        # ── Boundary condition ──
        tokens += [boundary, '&']

        # ── Target time ──
        tokens += self._num_to_tokens(target_time)

        # Convert → indices
        idx = [self.token2idx.get(t, self.token2idx['UNK']) for t in tokens]

        # Pad / truncate to max_len
        if len(idx) < self.max_len:
            idx += [self.pad_idx] * (self.max_len - len(idx))
        else:
            idx = idx[:self.max_len]

        return torch.tensor(idx, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA GENERATION  (Section 3.2.1 of paper)
# ═══════════════════════════════════════════════════════════════════════════════

class BurgersDataGenerator:
    """
    Generates Burgers' equation trajectories via finite differences.

    Equation (from paper eq 3.1 with γ=0):
        ∂u/∂t + α*∂(u²/2)/∂x - β*∂²u/∂x² = δ(t,x)

    Paper parameters:
        J=5, L=16
        α ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
        β ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
        Aj ~ U(-0.5, 0.5),  ωj ~ U(-0.4, 0.4),
        lj ~ {1,2,3},        φj ~ U(0, 2π)
    """

    def __init__(self, nx=100, T=4.0, nt=100, J=1, L=16.0):
        self.nx = nx
        self.T  = T
        self.nt = nt
        self.J  = J
        self.L  = L
        self.dx = L / nx
        self.dt = T / nt
        self.x  = np.linspace(0, L, nx, endpoint=False)
        self.t  = np.linspace(0, T, nt + 1)

    def forcing(self, t, x, params):
        """δ(t,x) = Σ Aj sin(ωj t + 2π lj x/L + φj)"""
        out = np.zeros_like(x)
        for (Aj, wj, lj, phij) in params:
            out += Aj * np.sin(wj * t + 2*np.pi*lj*x/self.L + phij)
        return out

    def sample_forcing_params(self):
        params = []
        for _ in range(self.J):
            params.append((
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.4, 0.4),
                np.random.choice([1, 2, 3]),
                np.random.uniform(0, 2*np.pi),
            ))
        return params

    def solve_burgers(self, alpha, beta, forcing_params):
        """
        Finite-difference solver (upwind advection + central diffusion).
        Periodic boundary conditions (matches paper: 0 to 16 domain).
        IC = forcing term at t=0.
        """
        u    = self.forcing(0.0, self.x, forcing_params).copy()
        traj = [u.copy()]

        for n in range(self.nt):
            t_n    = self.t[n]
            f      = self.forcing(t_n, self.x, forcing_params)
            u_r    = np.roll(u, -1)   # u_{i+1}
            u_l    = np.roll(u,  1)   # u_{i-1}

            # Lax-Friedrichs flux for α u²/2
            flux   = alpha * u**2 / 2.0
            adv = (np.roll(flux, -1) - np.roll(flux, 1)) / (2 * self.dx)

            # Central difference diffusion
#             diff   = beta * (u_r - 2*u + u_l) / self.dx**2

#             u = u + self.dt * (-adv + diff + f)
            u = u + self.dt * (-adv + f)
            u = u
            traj.append(u.copy())

        return np.array(traj)   # (nt+1, nx)

    def generate_dataset(self, n_samples=500,
                         alpha_vals=None, beta_vals=None):
        if alpha_vals is None:
            alpha_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        if beta_vals is None:
            beta_vals  = [0.0]

        trajs, alphas, betas, fparams = [], [], [], []
        print(f"Generating {n_samples} Burgers simulations...")
        for _ in tqdm(range(n_samples)):
            a = np.random.choice(alpha_vals)
            b = np.random.choice(beta_vals)
            p = self.sample_forcing_params()
            trajs.append(self.solve_burgers(a, b, p))
            alphas.append(a); betas.append(b); fparams.append(p)

        return {
            'trajectories':  np.array(trajs),    # (N, nt+1, nx)
            'alphas':        np.array(alphas),
            'betas':         np.array(betas),
            'forcing_params': fparams,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class BurgersDataset(Dataset):
    """
    Next-step prediction: use N_INPUT_FRAMES consecutive frames → predict frame+1.
    Matches Section 3.3.1 (1D Next-Step Prediction).
    """

    def __init__(self, data, tokenizer, n_input=10, stride=5):
        self.samples = []
        trajs   = data['trajectories']
        alphas  = data['alphas']
        betas   = data['betas']
        forcing = data['forcing_params']
        T_total = 4.0

        for i in range(len(trajs)):
            traj   = trajs[i]           # (nt+1, nx)
            nt1    = traj.shape[0]
            for start in range(0, nt1 - n_input - 1, stride):
                inp    = traj[start : start + n_input]          # (n_input, nx)
                target = traj[start + n_input]                  # (nx,)
                t_tgt  = (start + n_input) / (nt1 - 1) * T_total
                tok    = tokenizer.tokenize(
                    alphas[i], betas[i], forcing[i], target_time=t_tgt)
                self.samples.append((
                    inp.astype(np.float32),
                    target.astype(np.float32),
                    tok,
                    np.float32(alphas[i]),
                    np.float32(betas[i]),
                ))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt, tok, a, b = self.samples[idx]
        return (torch.from_numpy(inp),
                torch.from_numpy(tgt),
                tok,
                torch.tensor(a), torch.tensor(b))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a. Softmax-free Linear Attention (paper Section 3.1.2) ──────────────────

class LinearAttention(nn.Module):
    """
    z = Q (K̃ᵀ Ṽ) / n
    K̃, Ṽ = instance-normalized K, V
    Feature map: φ(x) = elu(x) + 1
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)
        self.k_norm = nn.InstanceNorm1d(dim)
        self.v_norm = nn.InstanceNorm1d(dim)

    def forward(self, q, k, v):
        Q = F.elu(self.q_proj(q)) + 1                              # (B,Nq,D)
        K = F.elu(self.k_norm(self.k_proj(k).transpose(1,2)).transpose(1,2)) + 1
        V = self.v_norm(self.v_proj(v).transpose(1,2)).transpose(1,2)
        Nk = k.shape[1]
        KV = torch.bmm(K.transpose(1,2), V)                        # (B,D,D)
        z  = torch.bmm(Q, KV) / (Nk + 1e-6)                       # (B,Nq,D)
        return self.out(z)


# ── 4b. Token Transformer (Figure 3.1a) ──────────────────────────────────────

class TokenTransformer(nn.Module):
    """
    Equation tokens → latent embedding Th.

    Novel embedding (†):  T used as Q,K,V via W_T1, W_T2, W_T3
    Standard embedding(*): fixed sinusoidal pos-enc + lookup table
    """
    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 max_len=100, dropout=0.1, method='novel'):
        super().__init__()
        self.method    = method
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if method == 'standard':
            pe  = torch.zeros(max_len, embed_dim)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, embed_dim, 2).float()
                            * -(np.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pos_enc', pe.unsqueeze(0))
        else:
            self.W_T1 = nn.Linear(embed_dim, embed_dim)
            self.W_T2 = nn.Linear(embed_dim, embed_dim)
            self.W_T3 = nn.Linear(embed_dim, embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        T = self.token_embed(tokens)                    # (B, L, D)
        # Scale to [-1, 1]  (paper: "significantly boosts performance")
        Tmin = T.min(); Tmax = T.max()
        T = 2.0 * (T - Tmin) / (Tmax - Tmin + 1e-8) - 1.0

        if self.method == 'standard':
            T = T + self.pos_enc[:, :T.shape[1], :]
            Th, _ = self.self_attn(T, T, T)
        else:
            T1, T2, T3 = self.W_T1(T), self.W_T2(T), self.W_T3(T)
            Th, _ = self.self_attn(T1, T2, T3)

        Th = self.dropout(Th)
        Th = self.norm(Th + T)
        return Th                                       # (B, L, D)


# ── 4c. Spectral Conv + FNO backbone ─────────────────────────────────────────

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                               # x: (B,C,N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weights.shape[1],
                             N//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=N)             # (B,out_ch,N)


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, n_input=10, nx=100):
        super().__init__()
        self.fc0  = nn.Linear(n_input + 1, width)      # lift: frames+grid → width
        convs = [SpectralConv1d(width, width, modes) for _ in range(4)]
        ws    = [nn.Conv1d(width, width, 1)            for _ in range(4)]
        self.convs = nn.ModuleList(convs)
        self.ws    = nn.ModuleList(ws)
        self.fc1  = nn.Linear(width, 128)
        self.fc2  = nn.Linear(128, 1)

    def forward(self, x, grid):                         # x:(B,T,nx), grid:(B,nx,1)
        x_in = torch.cat([x.permute(0,2,1), grid], dim=-1)  # (B,nx,T+1)
        x_in = self.fc0(x_in).permute(0,2,1)           # (B,width,nx)
        for conv, w in zip(self.convs, self.ws):
            x_in = F.gelu(conv(x_in) + w(x_in))
        x_in = x_in.permute(0,2,1)                     # (B,nx,width)
        return self.fc2(F.gelu(self.fc1(x_in))).squeeze(-1)  # (B,nx)


# ── 4d. Linear Attention Update Block (Figure 3.1b) ──────────────────────────

class LinearAttentionUpdateBlock(nn.Module):
    """
    Numerical-method-like update over L layers:
        Xₗ = Dropout(LA(Th₁, Th₂, Vₗ₋₁))
        tₗ  = MLP(l·t / L)
        Vₗ  = Vₗ₋₁ + MLP([Xₗ, tₗ])
    """
    def __init__(self, dim, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.W_X   = nn.Linear(dim, dim)
        self.W_Th1 = nn.Linear(dim, dim)
        self.W_Th2 = nn.Linear(dim, dim)

        self.la_layers  = nn.ModuleList(
            [LinearAttention(dim, num_heads=4) for _ in range(n_layers)])
        self.t_mlps     = nn.ModuleList([
            nn.Sequential(nn.Linear(1, dim//2), nn.GELU(),
                          nn.Linear(dim//2, dim))
            for _ in range(n_layers)])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(),
                          nn.Linear(dim, dim))
            for _ in range(n_layers)])
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, V0, Th, t):
        """
        V0: (B,nx,dim)   neural operator output
        Th: (B,L,dim)    token embedding
        t:  (B,)         target time
        """
        # Pool token seq → nx spatial positions
        Th_p = Th.mean(dim=1, keepdim=True).expand(-1, V0.shape[1], -1)
        V    = self.W_X(V0)
        Th1  = self.W_Th1(Th_p)
        Th2  = self.W_Th2(Th_p)

        for l in range(self.n_layers):
            X_l   = self.dropout(self.la_layers[l](Th1, Th2, V))
            frac  = ((l+1) * t / self.n_layers).unsqueeze(-1)   # (B,1)
            t_l   = self.t_mlps[l](frac).unsqueeze(1)            # (B,1,dim)
            t_l   = t_l.expand(-1, V.shape[1], -1)
            V     = V + self.update_mlps[l](torch.cat([X_l, t_l], dim=-1))

        return self.out_proj(V)                                   # (B,nx,dim)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FULL PITT MODEL  (Figure 3.1c)
# ═══════════════════════════════════════════════════════════════════════════════

class PITT_FNO(nn.Module):
    """
    Physics Informed Token Transformer with FNO backbone.

    Forward pass:
        tokens → TokenTransformer          → Th  (equation embedding)
        [x, grid] → FNO                    → V0  (neural operator passthrough)
        Th + V0   → LinearAttentionUpdate  → correction
        output = V0 + correction           (numerical update rule)
    """
    def __init__(self, vocab_size, nx=100, n_input=10,
                 embed_dim=128, fno_modes=16, fno_width=64,
                 n_update_layers=4, dropout=0.1,
                 token_method='novel', max_len=100):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_transformer = TokenTransformer(
            vocab_size, embed_dim, num_heads=4,
            max_len=max_len, dropout=dropout, method=token_method)

        self.fno = FNO1d(fno_modes, fno_width, n_input, nx)

        self.fno_proj   = nn.Linear(1, embed_dim)
        self.update_block = LinearAttentionUpdateBlock(
            embed_dim, n_update_layers, dropout)
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, grid, tokens, t):
        """
        x:      (B, n_input, nx)
        grid:   (B, nx, 1)
        tokens: (B, seq_len)
        t:      (B,)
        Returns: pred (B,nx), fno_passthrough (B,nx), correction (B,nx)
        """
        Th      = self.token_transformer(tokens)                  # (B,L,D)
        V0_flat = self.fno(x, grid)                               # (B,nx)
        V0      = self.fno_proj(V0_flat.unsqueeze(-1))            # (B,nx,D)
        V_up    = self.update_block(V0, Th, t)                    # (B,nx,D)
        corr    = self.final_proj(V_up).squeeze(-1)               # (B,nx)
        pred    = V0_flat + corr
        return pred, V0_flat, corr


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_grid(nx, bsz):
    """Normalized spatial grid on current device."""
    return torch.linspace(-1, 1, nx).reshape(1, nx, 1).expand(bsz, -1, -1).to(device)


def train_pitt(model, train_loader, val_loader, n_epochs=50, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state = float('inf'), None

    print("\n" + "="*60)
    print("Training PITT for Burgers' Equation")
    print("="*60)

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for inp, tgt, tok, _, _ in train_loader:
            inp  = inp.to(device, non_blocking=True)
            tgt  = tgt.to(device, non_blocking=True)
            tok  = tok.to(device, non_blocking=True)
            t_v  = torch.ones(inp.shape[0], device=device)
            grid = make_grid(inp.shape[-1], inp.shape[0])

            optimizer.zero_grad(set_to_none=True)
            pred, _, _ = model(inp, grid, tok, t_v)
            loss = criterion(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        avg_tr = ep_loss / len(train_loader)
        train_losses.append(avg_tr)

        # ── Validate ──
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for inp, tgt, tok, _, _ in val_loader:
                inp  = inp.to(device, non_blocking=True)
                tgt  = tgt.to(device, non_blocking=True)
                tok  = tok.to(device, non_blocking=True)
                t_v  = torch.ones(inp.shape[0], device=device)
                grid = make_grid(inp.shape[-1], inp.shape[0])
                pred, _, _ = model(inp, grid, tok, t_v)
                vl += criterion(pred, tgt).item()

        avg_vl = vl / len(val_loader)
        val_losses.append(avg_vl)
        scheduler.step(avg_vl)

        if avg_vl < best_val:
            best_val   = avg_vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}]  "
                  f"Train: {avg_tr:.6f}  Val: {avg_vl:.6f}  Best: {best_val:.6f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"\nBest validation loss: {best_val:.6f}")
    return train_losses, val_losses


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: AUTOREGRESSIVE ROLLOUT  (Section 3.3.4)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout(model, initial_frames, tokens, n_steps):
    """
    Autoregressively predict n_steps into the future.
    initial_frames: (1, n_input, nx) on device
    tokens:         (1, seq_len) on device
    """
    model.eval()
    frames = initial_frames.clone()
    preds  = []
    for _ in range(n_steps):
        t_v  = torch.ones(1, device=device)
        grid = make_grid(frames.shape[-1], 1)
        pred, _, _ = model(frames, grid, tokens, t_v)   # (1,nx)
        preds.append(pred.squeeze(0).cpu().numpy())
        frames = torch.cat([frames[:, 1:, :], pred.unsqueeze(1)], dim=1)
    return np.array(preds)   # (n_steps, nx)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(train_losses, color='#00d4ff', lw=2, label='Train Loss')
    ax.plot(val_losses,   color='#ff6b35', lw=2, label='Val Loss')
    ax.set_xlabel('Epoch', color='#94a3b8')
    ax.set_ylabel('MSE Loss', color='#94a3b8')
    ax.set_title("PITT Training Curve — Burgers' Equation",
                 color='#f0f6fc', fontweight='bold')
    ax.legend(facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e2e8f0')
    ax.tick_params(colors='#94a3b8')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, 'pitt_training_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {path}")


def plot_results(model, test_sample, gen, out_dir):
    inp_t, tgt_t, tok_t, alpha_v, beta_v = test_sample
    inp_t = inp_t.unsqueeze(0).to(device)
    tok_t = tok_t.unsqueeze(0).to(device)
    nx    = inp_t.shape[-1]
    x     = np.linspace(0, gen.L, nx, endpoint=False)

    model.eval()
    with torch.no_grad():
        t_v      = torch.ones(1, device=device)
        grid     = make_grid(nx, 1)
        pred, passthrough, correction = model(inp_t, grid, tok_t, t_v)

    pred_np = pred.squeeze(0).cpu().numpy()
    pass_np = passthrough.squeeze(0).cpu().numpy()
    corr_np = correction.squeeze(0).cpu().numpy()
    inp_np  = inp_t.squeeze(0).cpu().numpy()   # (n_input, nx)
    tgt_np  = tgt_t.numpy()

    # ── Rollout ──
    print("Running autoregressive rollout...")
    roll_preds = rollout(model, inp_t, tok_t, n_steps=40)

    C = dict(pred='#00d4ff', target='#ff6b35', pass_='#a855f7',
             corr='#22c55e', inp='#94a3b8', err='#ef4444')

    fig = plt.figure(figsize=(21, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def sax(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='#e2e8f0', fontsize=10, fontweight='bold', pad=7)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.6)
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        return ax

    # 1. Input frames
    ax = sax(fig.add_subplot(gs[0,0]), "Input Frames (last 3 of 10)")
    for i, fi in enumerate([-3,-2,-1]):
        ax.plot(x, inp_np[fi], color=C['inp'], alpha=0.4+0.3*i,
                lw=1.5, label=f'Frame {10+fi}')
    ax.legend(fontsize=7, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#94a3b8')
    ax.set_xlabel('x'); ax.set_ylabel('u(x,t)')

    # 2. FNO passthrough
    ax = sax(fig.add_subplot(gs[0,1]), "FNO Neural Operator Output\n(Passthrough)")
    ax.plot(x, inp_np[-1], color=C['inp'],    lw=1.5, ls='--', label='Last input')
    ax.plot(x, pass_np,   color=C['pass_'],  lw=2,            label='FNO output')
    ax.plot(x, tgt_np,    color=C['target'], lw=1.5, ls=':',  label='Target')
    ax.legend(fontsize=7, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#94a3b8')
    ax.set_xlabel('x'); ax.set_ylabel('u(x,t)')

    # 3. Token correction
    ax = sax(fig.add_subplot(gs[0,2]), "Token Attention Correction\n(Physics-informed update)")
    ax.plot(x, corr_np, color=C['corr'], lw=2, label='Correction')
    ax.axhline(0, color='#475569', lw=0.8, ls='--')
    ax.legend(fontsize=7, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#94a3b8')
    ax.set_xlabel('x'); ax.set_ylabel('Correction Δu')

    # 4. PITT prediction vs target
    ax = sax(fig.add_subplot(gs[1,0]), "PITT Prediction vs Target")
    ax.plot(x, tgt_np,  color=C['target'], lw=2.5, label='Target',   zorder=3)
    ax.plot(x, pred_np, color=C['pred'],   lw=1.8, ls='--', label='PITT', zorder=4)
    ax.fill_between(x, tgt_np, pred_np, alpha=0.12, color=C['err'])
    ax.legend(fontsize=7, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#94a3b8')
    ax.set_xlabel('x'); ax.set_ylabel('u(x,t)')

    # 5. Pointwise error
    err = np.abs(tgt_np - pred_np)
    ax  = sax(fig.add_subplot(gs[1,1]),
              f"Pointwise Abs Error  (MAE={err.mean():.5f})")
    ax.fill_between(x, 0, err, color=C['err'], alpha=0.55)
    ax.plot(x, err, color=C['err'], lw=1.5)
    ax.set_xlabel('x'); ax.set_ylabel('|error|')

    # 6. Decomposition summary
    ax = sax(fig.add_subplot(gs[1,2]), "Prediction Decomposition\n(FNO + Token Correction)")
    ax.plot(x, pass_np, color=C['pass_'],  lw=1.8, label='FNO passthrough')
    ax.plot(x, corr_np, color=C['corr'],   lw=1.8, label='Token correction')
    ax.plot(x, pred_np, color=C['pred'],   lw=2.2, label='PITT output')
    ax.plot(x, tgt_np,  color=C['target'], lw=1.5, ls=':', label='Target')
    ax.legend(fontsize=7, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#94a3b8')
    ax.set_xlabel('x'); ax.set_ylabel('u(x,t)')

    # 7. Space-time rollout heatmap
    rollout_full = np.vstack([inp_np, roll_preds])    # (n_input+40, nx)
    ax = sax(fig.add_subplot(gs[2,:2]),
             "Space-Time Rollout  (first 10 frames = input, rest = autoregressive)")
    im = ax.imshow(rollout_full.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1.5, vmax=1.5, origin='lower',
                   extent=[0, rollout_full.shape[0], 0, gen.L])
    ax.axvline(x=inp_np.shape[0], color='#22c55e', lw=1.5, ls='--',
               label='Prediction start')
    ax.set_xlabel('Time step'); ax.set_ylabel('x')
    ax.legend(fontsize=8, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e2e8f0')
    plt.colorbar(im, ax=ax, label='u(x,t)', fraction=0.02, pad=0.02)

    # 8. Rollout error accumulation
    ax  = sax(fig.add_subplot(gs[2,2]), "Rollout Error Accumulation")
    rerr = [np.mean(np.abs(roll_preds[s] - tgt_np)) for s in range(len(roll_preds))]
    ax.plot(range(1, len(rerr)+1), rerr, color=C['err'], lw=2)
    ax.fill_between(range(1, len(rerr)+1), rerr, alpha=0.2, color=C['err'])
    ax.set_xlabel('Rollout Step'); ax.set_ylabel('MAE vs first target')

    a_v = float(alpha_v); b_v = float(beta_v)
    fig.suptitle(
        f"PITT — Burgers' Equation  |  α={a_v:.3f}  β={b_v:.3f}",
        color='#f0f6fc', fontsize=14, fontweight='bold', y=0.99)

    path = os.path.join(out_dir, 'pitt_burgers_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Hyperparameters ──────────────────────────────────────────────────────
    # Defaults: lighter config for quick runs.
    # Scale up for paper-level accuracy (comments show paper values).
    NX              = 100
    T               = 4.0
    NT              = 300
    N_INPUT         = 10
    EMBED_DIM       = 128     
    FNO_MODES       = 32    
    FNO_WIDTH       = 64      
    N_UPDATE_LAYERS = 4       # paper: 4
    N_SAMPLES       = 500     # paper: 2500 per (α,β) combo → ~90k total
    N_EPOCHS        = 500     # paper: until convergence with early stopping
    BATCH_SIZE      = 64
    LR              = 1e-3
    TOKEN_METHOD    = 'novel' # 'novel' (†) or 'standard' (*)
    MAX_TOKEN_LEN   = 100
    NUM_WORKERS     = 4 if device.type == 'cuda' else 0
    PIN_MEMORY      = device.type == 'cuda'
    # ─────────────────────────────────────────────────────────────────────────

    print("="*60)
    print("PITT — Burgers' Equation (CUDA-ready)")
    print("="*60)

    # 1. Generate data
    gen  = BurgersDataGenerator(nx=NX, T=T, nt=NT)
    data = gen.generate_dataset(n_samples=N_SAMPLES)

    # 2. Tokenizer
    tokenizer = BurgersTokenizer(max_len=MAX_TOKEN_LEN)
    print(f"\nVocabulary size : {tokenizer.vocab_size}")

    # 3. Split 60/20/20
    n   = len(data['trajectories'])
    idx = np.random.permutation(n)
    tr_idx = idx[:int(0.6*n)]
    va_idx = idx[int(0.6*n):int(0.8*n)]
    te_idx = idx[int(0.8*n):]

    def sub(d, ii):
        return {
            'trajectories':  d['trajectories'][ii],
            'alphas':         d['alphas'][ii],
            'betas':          d['betas'][ii],
            'forcing_params': [d['forcing_params'][i] for i in ii],
        }

    tr_ds = BurgersDataset(sub(data,tr_idx), tokenizer, N_INPUT, stride=5)
    va_ds = BurgersDataset(sub(data,va_idx), tokenizer, N_INPUT, stride=5)
    te_ds = BurgersDataset(sub(data,te_idx), tokenizer, N_INPUT, stride=5)
    print(f"Samples  Train={len(tr_ds)}  Val={len(va_ds)}  Test={len(te_ds)}")

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # 4. Model
    model = PITT_FNO(
        vocab_size      = tokenizer.vocab_size,
        nx              = NX,
        n_input         = N_INPUT,
        embed_dim       = EMBED_DIM,
        fno_modes       = FNO_MODES,
        fno_width       = FNO_WIDTH,
        n_update_layers = N_UPDATE_LAYERS,
        dropout         = 0.1,
        token_method    = TOKEN_METHOD,
        max_len         = MAX_TOKEN_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # 5. Train
    tr_loss, va_loss = train_pitt(model, tr_ld, va_ld,
                                   n_epochs=N_EPOCHS, lr=LR)

    # 6. Test MAE
    model.eval()
    criterion = nn.L1Loss()
    te_mae = 0.0
    with torch.no_grad():
        for inp, tgt, tok, _, _ in te_ld:
            inp  = inp.to(device, non_blocking=True)
            tgt  = tgt.to(device, non_blocking=True)
            tok  = tok.to(device, non_blocking=True)
            t_v  = torch.ones(inp.shape[0], device=device)
            grid = make_grid(NX, inp.shape[0])
            pred, _, _ = model(inp, grid, tok, t_v)
            te_mae += criterion(pred, tgt).item()
    te_mae /= len(te_ld)
    print(f"\nTest MAE : {te_mae:.6f}  (×10⁻³ : {te_mae*1e3:.4f})")

    # 7. Save plots
    plot_training_curve(tr_loss, va_loss, OUT_DIR)
    plot_results(model, te_ds[0], gen, OUT_DIR)

    # 8. Save checkpoint
    ckpt_path = os.path.join(OUT_DIR, 'pitt_burgers_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(
            vocab_size=tokenizer.vocab_size, nx=NX, n_input=N_INPUT,
            embed_dim=EMBED_DIM, fno_modes=FNO_MODES, fno_width=FNO_WIDTH,
            n_update_layers=N_UPDATE_LAYERS, token_method=TOKEN_METHOD,
            max_len=MAX_TOKEN_LEN),
        'test_mae':    te_mae,
        'train_losses': tr_loss,
        'val_losses':   va_loss,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    print("\n" + "="*60)
    print(f"All outputs in: {os.path.abspath(OUT_DIR)}/")
    print("  pitt_burgers_results.png  — prediction & rollout")
    print("  pitt_training_curve.png   — loss curves")
    print("  pitt_burgers_model.pt     — checkpoint")
    print("="*60)


if __name__ == '__main__':
    main()
