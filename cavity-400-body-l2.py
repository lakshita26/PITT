
"""
Complete Physics Informed Token Transformer (PITT)
with Semicircular Solid Obstacle — VECTORIZED BCs
==================================================
KEY FIX vs previous version:
  apply_obstacle_bc() no longer uses a Python for-loop over cells.
  It uses pure tensor scatter ops (zero-copy GPU kernels) so the
  20 Poisson iterations x 5000 timesteps run in microseconds, not hours.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

GRID_SIZE = 41
REYNOLDS  = 400
L_DOM     = 1.0
DX        = L_DOM / (GRID_SIZE - 1)
DY        = L_DOM / (GRID_SIZE - 1)
DT        = 0.001

OBSTACLE_RADIUS = 0.2 * L_DOM   # tweak here to resize

CFD_STEPS  = 5000
SAVE_EVERY = 25

IN_CHANNELS = 4   # u, v, p, obstacle_mask

FNO_MODES  = 12
FNO_WIDTH  = 64
D_MODEL    = 32
SEQ_LEN    = 100
EPOCHS     = 5000
BATCH_SIZE = 8
LR         = 1e-3


# ==========================================
# Obstacle: mask + precomputed neighbour avg
# ==========================================
def build_obstacle_mask(grid_size=GRID_SIZE, radius=OBSTACLE_RADIUS,
                        cx=0.5*L_DOM, device=DEVICE):
    """
    Boolean (H,W) tensor. True = solid cell (inside semicircle dome on y=0).
    """
    x_arr = torch.linspace(0, L_DOM, grid_size, device=device)
    y_arr = torch.linspace(0, L_DOM, grid_size, device=device)
    Y, X  = torch.meshgrid(y_arr, x_arr, indexing='ij')
    dist2 = (X - cx)**2 + Y**2
    return (dist2 <= radius**2) & (Y <= radius)


def build_neighbour_avg_weights(mask, device=DEVICE):
    """
    Precompute, ONCE before the time-loop, the four shifted fluid masks
    used to average neighbouring pressures into obstacle cells.

    Returns a tuple (shifts, fluid_shifted_masks) that apply_obstacle_bc_fast
    uses with pure tensor ops — no Python loops at call-time.

    Each element of `shifts` is a (H,W) bool tensor: True where
      - the cell is solid AND
      - the neighbour in that direction is a fluid cell.
    `counts` is the normalisation denominator (float, same shape).
    """
    fluid = ~mask   # (H,W)

    # Shifted fluid masks: "is my neighbour in direction d a fluid cell?"
    # We pad with False so boundary solid cells don't pick up out-of-range values.
    def shift(t, di, dj):
        """Roll tensor t by (di,dj) and zero-pad the wrapped edge."""
        t2 = torch.roll(t, shifts=(-di, -dj), dims=(0, 1))
        if di > 0:  t2[-di:, :]  = False
        if di < 0:  t2[:-di, :]  = False
        if dj > 0:  t2[:, -dj:]  = False
        if dj < 0:  t2[:, :-dj]  = False
        return t2

    # For each direction: neighbour_is_fluid AND current cell is solid
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    contrib = []          # bool masks — True where this neighbour contributes
    p_shift_fns = []      # how to fetch that neighbour's pressure

    for di, dj in directions:
        # shift fluid mask in OPPOSITE direction to "look" at the neighbour
        neighbour_is_fluid = shift(fluid, di, dj)
        contrib.append(mask & neighbour_is_fluid)
        p_shift_fns.append((di, dj))

    # Precompute count tensor (float) for normalisation; 0 where no fluid neighbours
    count = torch.zeros_like(mask, dtype=torch.float32)
    for c in contrib:
        count += c.float()
    count = torch.where(count == 0, torch.ones_like(count), count)  # avoid /0

    return contrib, p_shift_fns, count


def apply_obstacle_bc_fast(u, v, p, mask, contrib, p_shift_fns, count):
    """
    Vectorized obstacle BC — runs entirely as GPU tensor ops, no Python loops.

    1. Zero u,v inside solid (no-slip).
    2. Set p inside solid = average of neighbouring FLUID cell pressures (Neumann).
    """
    # --- velocity no-slip ---
    u = u.masked_fill(mask, 0.0)
    v = v.masked_fill(mask, 0.0)

    # --- pressure Neumann: neighbour average ---
    p_new = torch.zeros_like(p)
    for (di, dj), c in zip(p_shift_fns, contrib):
        # fetch shifted pressure (the neighbour's value)
        p_neighbour = torch.roll(p, shifts=(-di, -dj), dims=(0, 1))
        # zero out wrap-around edges so they don't contribute
        if di > 0:  p_neighbour[-di:, :]  = 0.0
        if di < 0:  p_neighbour[:-di, :]  = 0.0
        if dj > 0:  p_neighbour[:, -dj:]  = 0.0
        if dj < 0:  p_neighbour[:, :-dj]  = 0.0
        p_new += c.float() * p_neighbour

    p_avg = p_new / count                   # (H,W)
    p     = torch.where(mask, p_avg, p)     # only overwrite solid cells

    return u, v, p


# ==========================================
# 1. Ground Truth CFD Simulator
# ==========================================
def generate_real_cfd_data():
    print("Generating CFD data (vectorized obstacle BCs)...")

    obstacle = build_obstacle_mask()
    # ── precompute neighbour weights ONCE ──────────────────────────────────
    contrib, p_shift_fns, count = build_neighbour_avg_weights(obstacle)
    # ───────────────────────────────────────────────────────────────────────

    u = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    v = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    p = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    b = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)

    nu  = 1.0 / REYNOLDS
    rho = 1.0
    frames = []
    obs_channel = obstacle.float().cpu()   # constant 4th channel

    for step in range(CFD_STEPS):
        un = u.clone()
        vn = v.clone()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * DT/(2*DX) * (un[1:-1, 2:] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * DT/(2*DY) * (un[2:, 1:-1] - un[:-2, 1:-1])
            + nu*DT/DX**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu*DT/DY**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * DT/(2*DX) * (vn[1:-1, 2:] - vn[1:-1, :-2])
            - vn[1:-1, 1:-1] * DT/(2*DY) * (vn[2:, 1:-1] - vn[:-2, 1:-1])
            + nu*DT/DX**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])
            + nu*DT/DY**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

        # Wall BCs
        u[0,:]=0; u[-1,:]=1; u[:,0]=0; u[:,-1]=0
        v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0

        # ── fast vectorized obstacle BC ────────────────────────────────────
        u, v, p = apply_obstacle_bc_fast(u, v, p, obstacle, contrib, p_shift_fns, count)
        # ───────────────────────────────────────────────────────────────────

        b[1:-1, 1:-1] = rho/DT * (
            (u[1:-1, 2:] - u[1:-1, :-2])/(2*DX) +
            (v[2:, 1:-1] - v[:-2, 1:-1])/(2*DY))

        for _ in range(20):
            pn = p.clone()
            p[1:-1, 1:-1] = (
                ((pn[1:-1,2:]+pn[1:-1,:-2])*DY**2 +
                 (pn[2:,1:-1]+pn[:-2,1:-1])*DX**2) / (2*(DX**2+DY**2))
                - DX**2*DY**2/(2*(DX**2+DY**2)) * b[1:-1,1:-1])
            p[:,-1]=p[:,-2]; p[0,:]=p[1,:]
            p[:,0]=p[:,1];   p[-1,:]=0.0
            # ── fast obstacle BC inside Poisson loop ───────────────────────
            u, v, p = apply_obstacle_bc_fast(u, v, p, obstacle, contrib, p_shift_fns, count)
            # ──────────────────────────────────────────────────────────────

        u[1:-1,1:-1] -= DT/rho * (p[1:-1,2:]-p[1:-1,:-2])/(2*DX)
        v[1:-1,1:-1] -= DT/rho * (p[2:,1:-1]-p[:-2,1:-1])/(2*DY)

        u[0,:]=0; u[-1,:]=1; u[:,0]=0; u[:,-1]=0
        v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0

        # ── final obstacle BC ──────────────────────────────────────────────
        u, v, p = apply_obstacle_bc_fast(u, v, p, obstacle, contrib, p_shift_fns, count)
        # ───────────────────────────────────────────────────────────────────

        if step % SAVE_EVERY == 0:
            frame = torch.stack([u.cpu(), v.cpu(), p.cpu(), obs_channel])
            frames.append(frame)
            if step % 500 == 0:
                print(f"  step {step}/{CFD_STEPS} | "
                      f"u_max={u.abs().max():.4f}  p_max={p.abs().max():.4f}")

    print(f"Generated {len(frames)} frames.")
    return torch.stack(frames)   # (N, 4, H, W)


# ==========================================
# 2. Tokenizer  [unchanged]
# ==========================================
TOKEN_VOCAB = {
    '(':0,')':1,'partial':2,'Sigma':3,'j':4,'Aj':5,'lj':6,'omega_j':7,'phi_j':8,
    'sin':9,'t':10,'u':11,'x':12,'y':13,'+':14,'-':15,'*':16,'/':17,
    'Neumann':18,'Dirichlet':19,'None_bc':20,
    '0':21,'1':22,'2':23,'3':24,'4':25,'5':26,'6':27,'7':28,'8':29,'9':30,
    'exp':31,'E':32,'e':33,',':34,'.':35,'&':36,'nabla':37,'=':38,
    'Delta':39,'dot':40,'nu':41,'rho':42,'p':43,'v':44,'w':45,'PAD':46
}
VOCAB_SIZE = len(TOKEN_VOCAB)

def tokenize_equation(nu_val=0.01, Re=400, bc_type='Dirichlet', pad_len=SEQ_LEN):
    def encode_number(val):
        return [TOKEN_VOCAB[ch] for ch in f"{val:.5g}" if ch in TOKEN_VOCAB]
    seq = [
        TOKEN_VOCAB['partial'],TOKEN_VOCAB['u'],TOKEN_VOCAB['partial'],TOKEN_VOCAB['t'],
        TOKEN_VOCAB['+'],TOKEN_VOCAB['u'],TOKEN_VOCAB['dot'],TOKEN_VOCAB['nabla'],TOKEN_VOCAB['u'],
        TOKEN_VOCAB['='],TOKEN_VOCAB['nu'],TOKEN_VOCAB['Delta'],TOKEN_VOCAB['u'],
        TOKEN_VOCAB['&'],TOKEN_VOCAB[bc_type],TOKEN_VOCAB['&']
    ]
    seq += encode_number(nu_val) + [TOKEN_VOCAB['&']] + encode_number(Re)
    seq += [TOKEN_VOCAB['PAD']] * max(0, pad_len - len(seq))
    return torch.tensor(seq[:pad_len], dtype=torch.long)


# ==========================================
# 3. PITT Architecture  [unchanged except in_channels=4]
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.in_c=in_c; self.out_c=out_c; self.m1=m1; self.m2=m2
        s = 1/(in_c*out_c)
        self.w1 = nn.Parameter(s*torch.rand(in_c,out_c,m1,m2,dtype=torch.cfloat))
        self.w2 = nn.Parameter(s*torch.rand(in_c,out_c,m1,m2,dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B,self.out_c,x.size(-2),x.size(-1)//2+1,device=x.device,dtype=torch.cfloat)
        out[:,:,:self.m1,:self.m2]  = torch.einsum("bixy,ioxy->boxy",xf[:,:,:self.m1,:self.m2], self.w1)
        out[:,:,-self.m1:,:self.m2] = torch.einsum("bixy,ioxy->boxy",xf[:,:,-self.m1:,:self.m2],self.w2)
        return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)))

class FNO2d(nn.Module):
    def __init__(self, in_c, modes, width):
        super().__init__()
        self.p    = nn.Conv2d(in_c, width, 1)
        self.conv = SpectralConv2d(width, width, modes, modes)
        self.w    = nn.Conv2d(width, width, 1)
        self.q    = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x = self.p(x)
        x = F.gelu(self.conv(x) + self.w(x))
        return self.q(x)

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads=2):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, d_model)
        self.mha  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        x = self.emb(tokens)
        mn = x.min(dim=-1,keepdim=True)[0]; mx = x.max(dim=-1,keepdim=True)[0]
        x  = 2*(x-mn)/(mx-mn+1e-6)-1
        a, _ = self.mha(x,x,x)
        return self.norm(x+a)

def inst_norm(x, eps=1e-6):
    return (x - x.mean(-1,keepdim=True)) / (x.std(-1,keepdim=True)+eps)

class PITTModel(nn.Module):
    def __init__(self, in_c, vocab_size, d_model, fno_modes, fno_width):
        super().__init__()
        self.fno       = FNO2d(in_c, fno_modes, fno_width)
        self.tok_trans = TokenTransformer(vocab_size, d_model)
        self.q_proj    = nn.Linear(d_model, d_model)
        self.k_proj    = nn.Linear(d_model, d_model)
        self.v_proj    = nn.Linear(fno_width, d_model)
        self.out_proj  = nn.Linear(d_model, in_c)
        self.fno_out   = nn.Conv2d(fno_width, in_c, 1)

    def forward(self, grid, tokens):
        b,c,h,w = grid.shape; n = h*w
        ff  = self.fno(grid)
        base= self.fno_out(ff)
        eq  = self.tok_trans(tokens).mean(1,keepdim=True)
        Q   = self.q_proj(eq).expand(-1,n,-1)
        K   = self.k_proj(eq).expand(-1,n,-1)
        V   = self.v_proj(ff.view(b,-1,n).permute(0,2,1))
        KV  = torch.matmul(inst_norm(K).transpose(1,2), inst_norm(V))
        corr= self.out_proj(torch.matmul(Q,KV)/n).permute(0,2,1).view(b,c,h,w)
        return base + corr


# ==========================================
# 4. Training Pipeline
# ==========================================
def train_model(dataset):
    model     = PITTModel(IN_CHANNELS, VOCAB_SIZE, D_MODEL, FNO_MODES, FNO_WIDTH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.L1Loss()

    X_train = dataset[:-1].to(DEVICE)
    Y_train = dataset[1:].to(DEVICE)
    N       = len(X_train)
    tokens  = tokenize_equation(nu_val=1.0/REYNOLDS, Re=REYNOLDS).unsqueeze(0).to(DEVICE)
    history = []

    print(f"\n--- Training on {N} frames (4-channel) ---")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N)
        eloss = 0
        for i in range(0, N, BATCH_SIZE):
            idx  = perm[i:i+BATCH_SIZE]
            bx, by = X_train[idx], Y_train[idx]
            bt   = tokens.repeat(len(bx), 1)

            optimizer.zero_grad()
            pred = model(bx, bt)

            loss_mae = criterion(pred, by)
            u_p = pred[:,0]; v_p = pred[:,1]
            div = ((u_p[:,1:-1,2:]-u_p[:,1:-1,:-2])/(2*DX) +
                   (v_p[:,2:,1:-1]-v_p[:,:-2,1:-1])/(2*DY))
            loss = loss_mae + 0.1*div.abs().mean()
            loss.backward()
            optimizer.step()
            eloss += loss.item()

        scheduler.step()
        avg = eloss / (N / BATCH_SIZE)
        history.append(avg)
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    return model, history, dataset[-1].unsqueeze(0).to(DEVICE)


# ==========================================
# 5. Plotting  [obstacle overlay on every panel]
# ==========================================
def evaluate_and_plot(model, history, eval_input):
    print("\n--- Plotting ---")
    model.eval()
    tokens = tokenize_equation().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(eval_input, tokens).cpu().numpy()[0]

    u, v, p = pred[0], pred[1], pred[2]
    x_arr = np.linspace(0, L_DOM, GRID_SIZE)
    y_arr = np.linspace(0, L_DOM, GRID_SIZE)
    X, Y  = np.meshgrid(x_arr, y_arr)

    obs   = build_obstacle_mask().cpu().numpy()
    um    = np.where(obs, np.nan, u)
    vm    = np.where(obs, np.nan, v)
    pm    = np.where(obs, np.nan, p)
    speed = np.where(obs, np.nan, np.sqrt(u**2+v**2))

    def obs_patch(ax):
        theta = np.linspace(0, np.pi, 300)
        cx = 0.5
        px = np.concatenate([cx + OBSTACLE_RADIUS*np.cos(theta),
                              [cx+OBSTACLE_RADIUS, cx-OBSTACLE_RADIUS]])
        py = np.concatenate([OBSTACLE_RADIUS*np.sin(theta), [0, 0]])
        ax.fill(px, py, color='#bbbbbb', zorder=5)
        ax.plot(cx + OBSTACLE_RADIUS*np.cos(theta),
                OBSTACLE_RADIUS*np.sin(theta),
                'k-', lw=0.8, zorder=6)

    fig = plt.figure(figsize=(18,12), facecolor='white')
    gs  = gridspec.GridSpec(2,3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0,:2])
    cf1 = ax1.contourf(X,Y,pm,levels=25,cmap='turbo',alpha=0.85)
    plt.colorbar(cf1,ax=ax1,label='Pressure [Pa]')
    ax1.contour(X,Y,pm,levels=12,colors='k',linewidths=0.5,alpha=0.4)
    ax1.quiver(X[::2,::2],Y[::2,::2],um[::2,::2],vm[::2,::2],
               color='white',scale=12,width=0.003,alpha=0.9)
    obs_patch(ax1)
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title(f'Pressure + Quiver  (Re={REYNOLDS})', fontsize=11)

    ax2 = fig.add_subplot(gs[0,2])
    cf2 = ax2.contourf(X,Y,speed,levels=25,cmap='plasma')
    plt.colorbar(cf2,ax=ax2,label='Speed [m/s]')
    ax2.streamplot(x_arr,y_arr,um,vm,color='white',linewidth=0.7,density=2.0)
    obs_patch(ax2)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    ax2.set_title('Speed + Streamlines')

    ax3 = fig.add_subplot(gs[1,0])
    mid = GRID_SIZE//2
    ax3.plot(um[:,mid], y_arr,'b-',lw=2.5,label='PITT')
    ax3.axvline(0,color='gray',lw=0.8,ls='--')
    ax3.fill_betweenx(y_arr,0,np.nan_to_num(um[:,mid]),alpha=0.12,color='blue')
    ax3.set_xlabel('u [m/s]'); ax3.set_ylabel('y [m]')
    ax3.set_title('u-profile  x=0.5'); ax3.grid(alpha=0.3); ax3.legend()

    ax4 = fig.add_subplot(gs[1,1])
    ax4.plot(x_arr, vm[mid,:],'r-',lw=2.5,label='PITT')
    ax4.axhline(0,color='gray',lw=0.8,ls='--')
    ax4.fill_between(x_arr,0,np.nan_to_num(vm[mid,:]),alpha=0.12,color='red')
    ax4.set_xlabel('x [m]'); ax4.set_ylabel('v [m/s]')
    ax4.set_title('v-profile  y=0.5'); ax4.grid(alpha=0.3); ax4.legend()

    ax5 = fig.add_subplot(gs[1,2])
    ax5.semilogy(history,color='#185FA5',lw=2,label='Loss')
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss')
    ax5.set_title('Learning Curve'); ax5.grid(alpha=0.3); ax5.legend()

    plt.savefig("pitt_learned_diagnostics.png",dpi=150,bbox_inches='tight')
    print("Saved -> pitt_learned_diagnostics.png")

    # Pressure analysis
    dp_dx = np.gradient(np.nan_to_num(pm),DX,axis=1)
    dp_dy = np.gradient(np.nan_to_num(pm),DY,axis=0)
    fig2, axes = plt.subplots(1,3,figsize=(18,6),facecolor='white')
    fig2.suptitle(f'Pressure Analysis | Re={REYNOLDS}', fontsize=13)

    ax = axes[0]
    ax.contourf(X,Y,pm,levels=30,cmap='RdBu_r',alpha=0.80)
    plt.colorbar(ax.collections[0],ax=ax,label='Pressure [Pa]')
    ax.streamplot(x_arr,y_arr,-dp_dx,-dp_dy,
                  color=np.sqrt(dp_dx**2+dp_dy**2),cmap='hot',linewidth=1.2)
    obs_patch(ax); ax.set_title('Pressure + −∇p streamlines')

    ax = axes[1]
    ax.contourf(X,Y,pm,levels=30,cmap='coolwarm',alpha=0.75)
    plt.colorbar(ax.collections[0],ax=ax,label='Pressure [Pa]')
    ax.streamplot(x_arr,y_arr,um,vm,color=speed,cmap='Greens',linewidth=1.2)
    obs_patch(ax); ax.set_title('Pressure + velocity streamlines')

    ax = axes[2]
    ax.contourf(X,Y,pm,levels=40,cmap='turbo')
    plt.colorbar(ax.collections[0],ax=ax,label='Pressure [Pa]')
    ax.contour(X,Y,pm,levels=12,colors='white',linewidths=1.0)
    pf = np.nan_to_num(pm)
    pi = np.unravel_index(np.argmax(pf),pf.shape)
    pj = np.unravel_index(np.argmin(pf),pf.shape)
    ax.plot(x_arr[pi[1]],y_arr[pi[0]],'w^',ms=9,label=f'p_max={pf.max():.3f}')
    ax.plot(x_arr[pj[1]],y_arr[pj[0]],'wv',ms=9,label=f'p_min={pf.min():.3f}')
    ax.legend(loc='lower left',facecolor='#222',labelcolor='white')
    obs_patch(ax); ax.set_title('Pressure map')

    plt.savefig("pitt_learned_pressure.png",dpi=150,bbox_inches='tight')
    print("Saved -> pitt_learned_pressure.png")


# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    dataset = generate_real_cfd_data()
    model, history, eval_frame = train_model(dataset)
    evaluate_and_plot(model, history, eval_frame)
    print("Pipeline completed successfully.")