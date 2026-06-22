"""
PITT — Lid-Driven Cavity with Triangular Obstacle
====================================================
Changes from semicircular version:
  • Obstacle is an isosceles triangle, base on bottom wall, apex pointing up
  • Triangle base half-width = OBSTACLE_RADIUS (same parameter reused)
  • Triangle height         = OBSTACLE_RADIUS (equilateral-like proportions)
  • obs_patch() replaced with triangle polygon fill
  • build_obstacle_mask() uses barycentric/edge test for triangle interior
  • Everything else (grid, Re, FNO, PITT, training, plots) unchanged
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon as MplPolygon
import os

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

GRID_SIZE       = 81
REYNOLDS        = 400
L_DOM           = 1.0
DX              = L_DOM / (GRID_SIZE - 1)
DY              = L_DOM / (GRID_SIZE - 1)
DT              = 0.001
OBSTACLE_RADIUS = 0.3 * L_DOM   # reused as half-base width and height of triangle

CFD_STEPS  = 5000
SAVE_EVERY = 25
IN_CHANNELS = 4     # u, v, p, obstacle_mask

FNO_MODES  = 12
FNO_WIDTH  = 64
D_MODEL    = 32
SEQ_LEN    = 100
EPOCHS     = 5000
BATCH_SIZE = 8
LR         = 1e-3

OUT_DIR = "pitt_plots_triangle"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Triangle geometry helpers
# ══════════════════════════════════════════════════════════════
# Triangle vertices (in domain coordinates):
#   Bottom-left : (cx - half_base, 0)
#   Bottom-right: (cx + half_base, 0)
#   Apex        : (cx, height)
TRI_CX        = 0.5 * L_DOM
TRI_HALF_BASE = OBSTACLE_RADIUS          # half base width
TRI_HEIGHT    = OBSTACLE_RADIUS          # vertical height

def _tri_vertices():
    """Return the three (x, y) vertices of the triangle."""
    return np.array([
        [TRI_CX - TRI_HALF_BASE, 0.0],
        [TRI_CX + TRI_HALF_BASE, 0.0],
        [TRI_CX,                 TRI_HEIGHT],
    ])


def build_obstacle_mask(grid_size=GRID_SIZE, device=DEVICE):
    """
    Boolean mask (H x W) that is True inside the triangular obstacle.
    Uses the sign-of-cross-product test (point-in-triangle).
    """
    x = torch.linspace(0, L_DOM, grid_size, device=device)
    y = torch.linspace(0, L_DOM, grid_size, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')   # shape (H, W)

    # Vertices
    ax_, ay_ = TRI_CX - TRI_HALF_BASE, 0.0
    bx_, by_ = TRI_CX + TRI_HALF_BASE, 0.0
    cx_, cy_ = TRI_CX,                 TRI_HEIGHT

    def sign(px, py, x1, y1, x2, y2):
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)

    d1 = sign(X, Y, ax_, ay_, bx_, by_)
    d2 = sign(X, Y, bx_, by_, cx_, cy_)
    d3 = sign(X, Y, cx_, cy_, ax_, ay_)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    inside = ~(has_neg & has_pos)   # True if point is inside triangle
    return inside


def build_neighbour_weights(mask, device=DEVICE):
    fluid = ~mask
    def shift_bool(t, di, dj):
        t2 = torch.roll(t, (-di, -dj), (0, 1))
        if di > 0: t2[-di:, :] = False
        if di < 0: t2[:-di, :] = False
        if dj > 0: t2[:, -dj:] = False
        if dj < 0: t2[:, :-dj] = False
        return t2
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    contrib = [mask & shift_bool(fluid, di, dj) for di, dj in directions]
    count   = sum(c.float() for c in contrib)
    count   = torch.where(count == 0, torch.ones_like(count), count)
    return contrib, directions, count


def apply_bc_fast(u, v, p, mask, contrib, dirs, count):
    u = u.masked_fill(mask, 0.0)
    v = v.masked_fill(mask, 0.0)
    p_new = torch.zeros_like(p)
    for (di, dj), c in zip(dirs, contrib):
        pn = torch.roll(p, (-di, -dj), (0, 1))
        if di > 0: pn[-di:, :] = 0.0
        if di < 0: pn[:-di, :] = 0.0
        if dj > 0: pn[:, -dj:] = 0.0
        if dj < 0: pn[:, :-dj] = 0.0
        p_new += c.float() * pn
    p = torch.where(mask, p_new / count, p)
    return u, v, p


# ══════════════════════════════════════════════════════════════
# 1. CFD Solver
# ══════════════════════════════════════════════════════════════
def generate_cfd_data():
    print(f"CFD: grid={GRID_SIZE}x{GRID_SIZE}, Re={REYNOLDS}, "
          f"triangle half-base={TRI_HALF_BASE:.2f}, height={TRI_HEIGHT:.2f}")
    obstacle = build_obstacle_mask()
    contrib, dirs, count = build_neighbour_weights(obstacle)
    obs_ch = obstacle.float().cpu()

    u = torch.zeros(GRID_SIZE, GRID_SIZE, device=DEVICE)
    v = torch.zeros(GRID_SIZE, GRID_SIZE, device=DEVICE)
    p = torch.zeros(GRID_SIZE, GRID_SIZE, device=DEVICE)
    b = torch.zeros(GRID_SIZE, GRID_SIZE, device=DEVICE)
    nu, rho = 1.0/REYNOLDS, 1.0
    frames = []

    for step in range(CFD_STEPS):
        un, vn = u.clone(), v.clone()

        u[1:-1,1:-1] = (un[1:-1,1:-1]
            - un[1:-1,1:-1]*DT/(2*DX)*(un[1:-1,2:]-un[1:-1,:-2])
            - vn[1:-1,1:-1]*DT/(2*DY)*(un[2:,1:-1]-un[:-2,1:-1])
            + nu*DT/DX**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2])
            + nu*DT/DY**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1]))

        v[1:-1,1:-1] = (vn[1:-1,1:-1]
            - un[1:-1,1:-1]*DT/(2*DX)*(vn[1:-1,2:]-vn[1:-1,:-2])
            - vn[1:-1,1:-1]*DT/(2*DY)*(vn[2:,1:-1]-vn[:-2,1:-1])
            + nu*DT/DX**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2])
            + nu*DT/DY**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1]))

        u[0,:]=0; u[-1,:]=1; u[:,0]=0; u[:,-1]=0
        v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0
        u, v, p = apply_bc_fast(u, v, p, obstacle, contrib, dirs, count)

        b[1:-1,1:-1] = rho/DT*(
            (u[1:-1,2:]-u[1:-1,:-2])/(2*DX) +
            (v[2:,1:-1]-v[:-2,1:-1])/(2*DY))

        for _ in range(20):
            pn = p.clone()
            p[1:-1,1:-1] = (
                ((pn[1:-1,2:]+pn[1:-1,:-2])*DY**2 +
                 (pn[2:,1:-1]+pn[:-2,1:-1])*DX**2)/(2*(DX**2+DY**2))
                - DX**2*DY**2/(2*(DX**2+DY**2))*b[1:-1,1:-1])
            p[:,-1]=p[:,-2]; p[0,:]=p[1,:]
            p[:,0]=p[:,1];   p[-1,:]=0.0
            u, v, p = apply_bc_fast(u, v, p, obstacle, contrib, dirs, count)

        u[1:-1,1:-1] -= DT/rho*(p[1:-1,2:]-p[1:-1,:-2])/(2*DX)
        v[1:-1,1:-1] -= DT/rho*(p[2:,1:-1]-p[:-2,1:-1])/(2*DY)
        u[0,:]=0; u[-1,:]=1; u[:,0]=0; u[:,-1]=0
        v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0
        u, v, p = apply_bc_fast(u, v, p, obstacle, contrib, dirs, count)

        if step % SAVE_EVERY == 0:
            frames.append(torch.stack([u.cpu(), v.cpu(), p.cpu(), obs_ch]))
            if step % 500 == 0:
                print(f"  step {step}/{CFD_STEPS} | u_max={u.abs().max():.4f} | p_max={p.abs().max():.4f}")

    print(f"Generated {len(frames)} frames.")
    return torch.stack(frames)


# ══════════════════════════════════════════════════════════════
# 2. Tokenizer
# ══════════════════════════════════════════════════════════════
TOKEN_VOCAB = {
    '(':0,')':1,'partial':2,'Sigma':3,'j':4,'Aj':5,'lj':6,'omega_j':7,'phi_j':8,
    'sin':9,'t':10,'u':11,'x':12,'y':13,'+':14,'-':15,'*':16,'/':17,
    'Neumann':18,'Dirichlet':19,'None_bc':20,
    '0':21,'1':22,'2':23,'3':24,'4':25,'5':26,'6':27,'7':28,'8':29,'9':30,
    'exp':31,'E':32,'e':33,',':34,'.':35,'&':36,'nabla':37,'=':38,
    'Delta':39,'dot':40,'nu':41,'rho':42,'p':43,'v':44,'w':45,'PAD':46
}
VOCAB_SIZE = len(TOKEN_VOCAB)

def tokenize_equation(nu_val=1.0/REYNOLDS, Re=REYNOLDS, bc_type='Dirichlet', pad_len=SEQ_LEN):
    def enc(val):
        return [TOKEN_VOCAB[ch] for ch in f"{val:.5g}" if ch in TOKEN_VOCAB]
    seq = [TOKEN_VOCAB['partial'],TOKEN_VOCAB['u'],TOKEN_VOCAB['partial'],TOKEN_VOCAB['t'],
           TOKEN_VOCAB['+'],TOKEN_VOCAB['u'],TOKEN_VOCAB['dot'],TOKEN_VOCAB['nabla'],TOKEN_VOCAB['u'],
           TOKEN_VOCAB['='],TOKEN_VOCAB['nu'],TOKEN_VOCAB['Delta'],TOKEN_VOCAB['u'],
           TOKEN_VOCAB['&'],TOKEN_VOCAB[bc_type],TOKEN_VOCAB['&']]
    seq += enc(nu_val) + [TOKEN_VOCAB['&']] + enc(Re)
    seq += [TOKEN_VOCAB['PAD']] * max(0, pad_len - len(seq))
    return torch.tensor(seq[:pad_len], dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# 3. Model
# ══════════════════════════════════════════════════════════════
class SpectralConv2d(nn.Module):
    def __init__(self, ic, oc, m1, m2):
        super().__init__()
        s = 1/(ic*oc)
        self.m1,self.m2 = m1,m2
        self.w1 = nn.Parameter(s*torch.rand(ic,oc,m1,m2,dtype=torch.cfloat))
        self.w2 = nn.Parameter(s*torch.rand(ic,oc,m1,m2,dtype=torch.cfloat))
    def forward(self,x):
        B=x.shape[0]; xf=torch.fft.rfft2(x)
        out=torch.zeros(B,self.w1.shape[1],x.size(-2),x.size(-1)//2+1,device=x.device,dtype=torch.cfloat)
        out[:,:,:self.m1,:self.m2]  = torch.einsum("bixy,ioxy->boxy",xf[:,:,:self.m1,:self.m2], self.w1)
        out[:,:,-self.m1:,:self.m2] = torch.einsum("bixy,ioxy->boxy",xf[:,:,-self.m1:,:self.m2],self.w2)
        return torch.fft.irfft2(out,s=(x.size(-2),x.size(-1)))

class FNO2d(nn.Module):
    def __init__(self,ic,modes,width):
        super().__init__()
        self.p=nn.Conv2d(ic,width,1); self.sc=SpectralConv2d(width,width,modes,modes)
        self.w=nn.Conv2d(width,width,1); self.q=nn.Conv2d(width,width,1)
    def forward(self,x):
        x=self.p(x); return self.q(F.gelu(self.sc(x)+self.w(x)))

class TokenTransformer(nn.Module):
    def __init__(self,vs,dm,nh=2):
        super().__init__()
        self.emb=nn.Embedding(vs,dm); self.mha=nn.MultiheadAttention(dm,nh,batch_first=True)
        self.norm=nn.LayerNorm(dm)
    def forward(self,t):
        x=self.emb(t); mn=x.min(-1,keepdim=True)[0]; mx=x.max(-1,keepdim=True)[0]
        x=2*(x-mn)/(mx-mn+1e-6)-1; a,_=self.mha(x,x,x); return self.norm(x+a)

def inorm(x,eps=1e-6): return (x-x.mean(-1,keepdim=True))/(x.std(-1,keepdim=True)+eps)

class PITTModel(nn.Module):
    def __init__(self,ic,vs,dm,fm,fw):
        super().__init__()
        self.fno=FNO2d(ic,fm,fw); self.tt=TokenTransformer(vs,dm)
        self.qp=nn.Linear(dm,dm); self.kp=nn.Linear(dm,dm)
        self.vp=nn.Linear(fw,dm); self.op=nn.Linear(dm,ic)
        self.fo=nn.Conv2d(fw,ic,1)
    def forward(self,g,t):
        b,c,h,w=g.shape; n=h*w
        ff=self.fno(g); base=self.fo(ff)
        eq=self.tt(t).mean(1,keepdim=True)
        Q=self.qp(eq).expand(-1,n,-1); K=self.kp(eq).expand(-1,n,-1)
        V=self.vp(ff.view(b,-1,n).permute(0,2,1))
        corr=self.op(torch.matmul(Q,torch.matmul(inorm(K).transpose(1,2),inorm(V)))/n)
        return base + corr.permute(0,2,1).view(b,c,h,w)


# ══════════════════════════════════════════════════════════════
# 4. Training
# ══════════════════════════════════════════════════════════════
def train_model(dataset):
    model=PITTModel(IN_CHANNELS,VOCAB_SIZE,D_MODEL,FNO_MODES,FNO_WIDTH).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.L1Loss()

    X=dataset[:-1].to(DEVICE); Y=dataset[1:].to(DEVICE); N=len(X)
    tok=tokenize_equation().unsqueeze(0).to(DEVICE)
    hist=[]

    print(f"\nTraining on {N} frames, logging every 250 epochs")
    for ep in range(EPOCHS):
        model.train(); perm=torch.randperm(N); eloss=0
        for i in range(0,N,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]; bx,by=X[idx],Y[idx]
            bt=tok.repeat(len(bx),1)
            opt.zero_grad(); pred=model(bx,bt)
            up,vp=pred[:,0],pred[:,1]
            div=((up[:,1:-1,2:]-up[:,1:-1,:-2])/(2*DX)+
                 (vp[:,2:,1:-1]-vp[:,:-2,1:-1])/(2*DY))
            loss=crit(pred,by)+0.1*div.abs().mean()
            loss.backward(); opt.step(); eloss+=loss.item()
        sch.step()
        avg=eloss/(N/BATCH_SIZE); hist.append(avg)
        if (ep+1)%250==0:
            print(f"  Epoch {ep+1:5d}/{EPOCHS} | Loss: {avg:.6f} | LR: {sch.get_last_lr()[0]:.2e}")

    return model, hist, dataset[-1].unsqueeze(0).to(DEVICE)


# ══════════════════════════════════════════════════════════════
# 5. Plotting helpers
# ══════════════════════════════════════════════════════════════
def obs_patch(ax, alpha=1.0):
    """Draw filled triangle obstacle on axes."""
    verts = _tri_vertices()
    tri = MplPolygon(verts, closed=True,
                     facecolor='#d0d0d0', edgecolor='#444444',
                     linewidth=1.0, zorder=5, alpha=alpha)
    ax.add_patch(tri)

def axis_style(ax, title, xlabel='x [m]', ylabel='y [m]'):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, L_DOM); ax.set_ylim(0, L_DOM)

def add_cbar(fig, ax, cf, label):
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


# ══════════════════════════════════════════════════════════════
# 6. Evaluate & Save Individual Plots
# ══════════════════════════════════════════════════════════════
def evaluate_and_plot(model, history, eval_input):
    print("\nGenerating plots...")
    model.eval()
    tok = tokenize_equation().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(eval_input, tok).cpu().numpy()[0]

    u, v, p = pred[0], pred[1], pred[2]
    x_arr = np.linspace(0, L_DOM, GRID_SIZE)
    y_arr = np.linspace(0, L_DOM, GRID_SIZE)
    X, Y  = np.meshgrid(x_arr, y_arr)

    obs   = build_obstacle_mask().cpu().numpy()
    um    = np.where(obs, np.nan, u)
    vm    = np.where(obs, np.nan, v)
    pm    = np.where(obs, np.nan, p)
    speed = np.where(obs, np.nan, np.sqrt(u**2 + v**2))

    # ── Shared colorbar limits ────────────────────────────────────────────────
    p_vmin, p_vmax     = np.nanmin(pm), np.nanmax(pm)
    s_vmin, s_vmax     = 0.0,           np.nanmax(speed)
    u_vmin, u_vmax     = np.nanmin(um), np.nanmax(um)
    v_vmin, v_vmax     = np.nanmin(vm), np.nanmax(vm)
    LEVELS_P  = np.linspace(p_vmin, p_vmax, 30)
    LEVELS_S  = np.linspace(s_vmin, s_vmax, 30)
    LEVELS_U  = np.linspace(u_vmin, u_vmax, 30)
    LEVELS_V  = np.linspace(v_vmin, v_vmax, 30)
    # ─────────────────────────────────────────────────────────────────────────

    skip = max(1, GRID_SIZE // 20)   # quiver stride scaled to grid

    # ── Plot 1: Pressure contourf + quiver ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='turbo',
                     vmin=p_vmin, vmax=p_vmax, extend='both')
    ax.contour(X, Y, pm, levels=12, colors='k', linewidths=0.4, alpha=0.35)
    ax.quiver(X[::skip,::skip], Y[::skip,::skip],
              um[::skip,::skip], vm[::skip,::skip],
              color='white', scale=15, width=0.003, alpha=0.85, zorder=4)
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'Pressure [Pa]')
    axis_style(ax, f'Pressure + Quiver  (Re={REYNOLDS})')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_pressure_quiver.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 01_pressure_quiver.png")

    # ── Plot 2: Pressure contourf + velocity streamlines ─────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='coolwarm',
                     vmin=p_vmin, vmax=p_vmax, extend='both')
    ax.streamplot(x_arr, y_arr, um, vm,
                  color=speed, cmap='Greens', linewidth=1.2, density=1.5, arrowsize=1.0)
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'Pressure [Pa]')
    axis_style(ax, 'Pressure + velocity streamlines')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_pressure_streamlines.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 02_pressure_streamlines.png")

    # ── Plot 3: u-velocity contourf ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, um, levels=LEVELS_U, cmap='RdBu_r',
                     vmin=u_vmin, vmax=u_vmax, extend='both')
    ax.contour(X, Y, um, levels=[0], colors='k', linewidths=1.2)
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'u-velocity [m/s]')
    axis_style(ax, 'u-Velocity Field')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_u_velocity.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 03_u_velocity.png")

    # ── Plot 4: v-velocity contourf ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, vm, levels=LEVELS_V, cmap='RdBu_r',
                     vmin=v_vmin, vmax=v_vmax, extend='both')
    ax.contour(X, Y, vm, levels=[0], colors='k', linewidths=1.2)
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'v-velocity [m/s]')
    axis_style(ax, 'v-Velocity Field')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_v_velocity.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 04_v_velocity.png")

    # ── Plot 5: Pressure + −∇p streamlines ───────────────────────────────────
    dp_dx = np.gradient(np.nan_to_num(pm), DX, axis=1)
    dp_dy = np.gradient(np.nan_to_num(pm), DY, axis=0)
    grad_mag = np.sqrt(dp_dx**2 + dp_dy**2)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=LEVELS_P, cmap='RdBu_r',
                     vmin=p_vmin, vmax=p_vmax, extend='both')
    strm = ax.streamplot(x_arr, y_arr, -dp_dx, -dp_dy,
                         color=grad_mag, cmap='hot', linewidth=1.0, density=1.5)
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'Pressure [Pa]')
    cb2 = fig.colorbar(strm.lines, ax=ax, fraction=0.03, pad=0.12)
    cb2.set_label('|∇p|', fontsize=9); cb2.ax.tick_params(labelsize=8)
    axis_style(ax, 'Pressure Field + −∇p Streamlines')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_pressure_gradient.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 05_pressure_gradient.png")

    # ── Plot 6: u-profile at x=0.5 ───────────────────────────────────────────
    mid_x = GRID_SIZE // 2
    fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
    ax.plot(um[:, mid_x], y_arr, 'b-', lw=2.5, label='PITT prediction')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.fill_betweenx(y_arr, 0, np.nan_to_num(um[:, mid_x]),
                     alpha=0.15, color='blue')
    # shade obstacle region on profile (triangle height at centreline)
    ax.axhspan(0, TRI_HEIGHT, alpha=0.10, color='gray', label='obstacle region')
    ax.set_xlabel('u [m/s]', fontsize=10); ax.set_ylabel('y [m]', fontsize=10)
    ax.set_title('u-Velocity Profile at x = 0.5 m', fontsize=11)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_u_profile_x05.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 06_u_profile_x05.png")

    # ── Plot 7: v-profile at y=0.5 ───────────────────────────────────────────
    mid_y = GRID_SIZE // 2
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    ax.plot(x_arr, vm[mid_y, :], 'r-', lw=2.5, label='PITT prediction')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.fill_between(x_arr, 0, np.nan_to_num(vm[mid_y, :]),
                    alpha=0.15, color='red')
    ax.set_xlabel('x [m]', fontsize=10); ax.set_ylabel('v [m/s]', fontsize=10)
    ax.set_title('v-Velocity Profile at y = 0.5 m', fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_v_profile_y05.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 07_v_profile_y05.png")

    # ── Plot 8: Pressure map with min/max markers ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    cf = ax.contourf(X, Y, pm, levels=40, cmap='turbo',
                     vmin=p_vmin, vmax=p_vmax, extend='both')
    ax.contour(X, Y, pm, levels=14, colors='white', linewidths=0.7, alpha=0.6)
    pf = np.nan_to_num(pm)
    pi = np.unravel_index(np.argmax(pf), pf.shape)
    pj = np.unravel_index(np.argmin(pf), pf.shape)
    ax.plot(x_arr[pi[1]], y_arr[pi[0]], 'w^', ms=10, zorder=7,
            label=f'p_max = {pf.max():.4f} Pa')
    ax.plot(x_arr[pj[1]], y_arr[pj[0]], 'wv', ms=10, zorder=7,
            label=f'p_min = {pf.min():.4f} Pa')
    obs_patch(ax)
    add_cbar(fig, ax, cf, 'Pressure [Pa]')
    leg = ax.legend(fontsize=9, loc='lower left', facecolor='#1a1a1a',
                    labelcolor='white', framealpha=0.85)
    axis_style(ax, 'Pressure Map — Extrema Marked')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_pressure_map.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 08_pressure_map.png")

    # ── Plot 9: Learning curve (x-axis every 250 epochs) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    epochs_arr = np.arange(1, len(history)+1)
    ax.semilogy(epochs_arr, history, color='#1a5fa8', lw=1.8, label='Total Loss (MAE + div)')
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('PITT Training — Learning Curve', fontsize=12)
    ax.grid(which='major', alpha=0.35); ax.grid(which='minor', alpha=0.12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/09_learning_curve.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved 09_learning_curve.png")

    print(f"\nAll plots saved to ./{OUT_DIR}/")


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    dataset = generate_cfd_data()
    model, history, eval_frame = train_model(dataset)
    evaluate_and_plot(model, history, eval_frame)
    print("Done.")