"""
Complete Physics Informed Token Transformer (PITT) 
==================================================
Includes:
1. Ground Truth CFD Data Generator (Lid-Driven Cavity)
2. Tokenization & FNO Neural Operator
3. Token Transformer & Linear Attention
4. Actual Training Pipeline
5. Diagnostic Plotting
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

# Grid & Physics
GRID_SIZE = 41
REYNOLDS = 400
L_DOM = 1.0
DX = L_DOM / (GRID_SIZE - 1)
DY = L_DOM / (GRID_SIZE - 1)
DT = 0.001

# --- INCREASED DATA VOLUME ---
CFD_STEPS = 5000       # Allows the vortex to fully develop into steady-state
SAVE_EVERY = 25        # Generates 200 high-quality frames

# --- INCREASED NETWORK CAPACITY & TRAINING ---
IN_CHANNELS = 3        
FNO_MODES = 12
FNO_WIDTH = 64         # Increased from 32 (Gives the model a "bigger brain" for details)
D_MODEL = 32
SEQ_LEN = 100
EPOCHS = 5000           # Increased from 50 (More time to smooth out errors)
BATCH_SIZE = 8
LR = 1e-3

# ==========================================
# 1. Ground Truth CFD Simulator (Data Generator)
# ==========================================
def generate_real_cfd_data():
    print("Generating physical ground truth data (Classical CFD)...")
    u = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    v = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    p = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    b = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.float32)
    
    nu = 1.0 / REYNOLDS
    rho = 1.0
    frames = []

    for step in range(CFD_STEPS):
        un = u.clone()
        vn = v.clone()
        
        # Tentative Velocity
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                         un[1:-1, 1:-1] * DT / (2*DX) * (un[1:-1, 2:] - un[1:-1, :-2]) - 
                         vn[1:-1, 1:-1] * DT / (2*DY) * (un[2:, 1:-1] - un[:-2, 1:-1]) + 
                         nu * DT / DX**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) + 
                         nu * DT / DY**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                         un[1:-1, 1:-1] * DT / (2*DX) * (vn[1:-1, 2:] - vn[1:-1, :-2]) - 
                         vn[1:-1, 1:-1] * DT / (2*DY) * (vn[2:, 1:-1] - vn[:-2, 1:-1]) + 
                         nu * DT / DX**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) + 
                         nu * DT / DY**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

        # Boundary Conditions (Lid Driven)
        u[0, :] = 0.0; u[-1, :] = 1.0
        u[:, 0] = 0.0; u[:, -1] = 0.0
        v[0, :] = 0.0; v[-1, :] = 0.0
        v[:, 0] = 0.0; v[:, -1] = 0.0

        # Pressure Poisson setup
        b[1:-1, 1:-1] = rho / DT * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*DX) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*DY))
        
        for _ in range(20): # Poisson iterations
            pn = p.clone()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * DY**2 + 
                              (pn[2:, 1:-1] + pn[:-2, 1:-1]) * DX**2) / (2 * (DX**2 + DY**2)) - 
                              DX**2 * DY**2 / (2 * (DX**2 + DY**2)) * b[1:-1, 1:-1])
            p[:, -1] = p[:, -2]; p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]; p[-1, :] = 0.0

        # Projection
        u[1:-1, 1:-1] -= DT / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2*DX)
        v[1:-1, 1:-1] -= DT / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*DY)
        
        # Re-apply BCs
        u[0, :] = 0.0; u[-1, :] = 1.0
        u[:, 0] = 0.0; u[:, -1] = 0.0
        v[0, :] = 0.0; v[-1, :] = 0.0
        v[:, 0] = 0.0; v[:, -1] = 0.0

        if step % SAVE_EVERY == 0:
            frames.append(torch.stack([u, v, p]).cpu())
            
    print(f"Generated {len(frames)} frames of physical data.")
    return torch.stack(frames)

# ==========================================
# 2. Tokenizer & Equations
# ==========================================
TOKEN_VOCAB = {
    '(':0, ')':1, 'partial':2, 'Sigma':3, 'j':4, 'Aj':5, 'lj':6, 'omega_j':7, 'phi_j':8,
    'sin':9, 't':10, 'u':11, 'x':12, 'y':13, '+':14, '-':15, '*':16, '/':17,
    'Neumann':18, 'Dirichlet':19, 'None_bc':20,
    '0':21, '1':22, '2':23, '3':24, '4':25, '5':26, '6':27, '7':28, '8':29, '9':30, 'exp':31, 'E':32, 'e':33,
    ',':34, '.':35, '&':36, 'nabla':37, '=':38, 'Delta':39, 'dot':40,
    'nu':41, 'rho':42, 'p':43, 'v':44, 'w':45, 'PAD':46
}
VOCAB_SIZE = len(TOKEN_VOCAB)

def tokenize_equation(nu_val=0.01, Re=400, bc_type='Dirichlet', pad_len=SEQ_LEN):
    def encode_number(val):
        s = f"{val:.5g}"
        return [TOKEN_VOCAB[ch] for ch in s if ch in TOKEN_VOCAB or ch == '-']
    seq = [
        TOKEN_VOCAB['partial'], TOKEN_VOCAB['u'], TOKEN_VOCAB['partial'], TOKEN_VOCAB['t'],
        TOKEN_VOCAB['+'], TOKEN_VOCAB['u'], TOKEN_VOCAB['dot'], TOKEN_VOCAB['nabla'], TOKEN_VOCAB['u'],
        TOKEN_VOCAB['='], TOKEN_VOCAB['nu'], TOKEN_VOCAB['Delta'], TOKEN_VOCAB['u'],
        TOKEN_VOCAB['&'], TOKEN_VOCAB[bc_type], TOKEN_VOCAB['&']
    ]
    seq += encode_number(nu_val) + [TOKEN_VOCAB['&']] + encode_number(Re)
    seq += [TOKEN_VOCAB['PAD']] * max(0, pad_len - len(seq))
    return torch.tensor(seq[:pad_len], dtype=torch.long)

# ==========================================
# 3. PITT Architecture (FNO + Transformer)
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNO2d(nn.Module):
    def __init__(self, in_channels, modes, width):
        super(FNO2d, self).__init__()
        self.p = nn.Conv2d(in_channels, width, 1)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.w1 = nn.Conv2d(width, width, 1)
        self.q = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x = self.p(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        return self.q(x)

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads=2):
        super(TokenTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = 2.0 * (x - x_min) / (x_max - x_min + 1e-6) - 1.0
        attn_out, _ = self.mha(x, x, x)
        return self.layer_norm(x + attn_out)

def manual_instance_norm(x, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

class PITTModel(nn.Module):
    def __init__(self, in_channels, vocab_size, d_model, fno_modes, fno_width):
        super(PITTModel, self).__init__()
        self.fno = FNO2d(in_channels, fno_modes, fno_width)
        self.token_transformer = TokenTransformer(vocab_size, d_model)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(fno_width, d_model)
        
        self.out_proj = nn.Linear(d_model, in_channels)
        self.fno_out_proj = nn.Conv2d(fno_width, in_channels, 1)

    def forward(self, grid, tokens):
        b, c, h, w = grid.shape
        n_points = h * w
        
        fno_features = self.fno(grid) 
        fno_base_pred = self.fno_out_proj(fno_features)
        
        eq_latent = self.token_transformer(tokens) 
        eq_pooled = eq_latent.mean(dim=1, keepdim=True) 
        
        Q = self.q_proj(eq_pooled).expand(-1, n_points, -1) 
        K = self.k_proj(eq_pooled).expand(-1, n_points, -1) 
        
        V_input = fno_features.view(b, -1, n_points).permute(0, 2, 1) 
        V = self.v_proj(V_input) 
        
        K_norm = manual_instance_norm(K)
        V_norm = manual_instance_norm(V)
        
        K_V = torch.matmul(K_norm.transpose(1, 2), V_norm) 
        attention_out = torch.matmul(Q, K_V) / n_points    
        
        correction = self.out_proj(attention_out).permute(0, 2, 1).view(b, c, h, w)
        return fno_base_pred + correction

# ==========================================
# 4. Training Pipeline
# ==========================================
# def train_model(dataset):
#     model = PITTModel(IN_CHANNELS, VOCAB_SIZE, D_MODEL, FNO_MODES, FNO_WIDTH).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
#     # --- ADDED: Learning Rate Scheduler ---
#     # This smoothly lowers the learning rate as epochs go on, removing "wobble"
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
#     criterion = nn.L1Loss() # MAE Loss
    
#     # Prepare Inputs (X: frame t, Y: frame t+1)
#     X_train = dataset[:-1].to(DEVICE)
#     Y_train = dataset[1:].to(DEVICE)
#     num_samples = len(X_train)
    
#     tokens = tokenize_equation(nu_val=1.0/REYNOLDS, Re=REYNOLDS).unsqueeze(0).to(DEVICE)
#     loss_history = []

#     print(f"\n--- Training PITT Model on {num_samples} Real Physics Frames ---")
#     for epoch in range(EPOCHS):
#         model.train()
#         permutation = torch.randperm(num_samples)
#         epoch_loss = 0
        
#         for i in range(0, num_samples, BATCH_SIZE):
#             indices = permutation[i:i+BATCH_SIZE]
#             batch_x, batch_y = X_train[indices], Y_train[indices]
#             batch_tokens = tokens.repeat(len(batch_x), 1)

#             optimizer.zero_grad()
#             predictions = model(batch_x, batch_tokens)
#             loss = criterion(predictions, batch_y)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         # Step the scheduler to adjust learning rate
#         scheduler.step()

#         avg_loss = epoch_loss / (num_samples / BATCH_SIZE)
#         loss_history.append(avg_loss)
        
#         if (epoch+1) % 25 == 0:
#             current_lr = scheduler.get_last_lr()[0]
#             print(f"Epoch [{epoch+1}/{EPOCHS}] | MAE Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
            
#     return model, loss_history, dataset[-1].unsqueeze(0).to(DEVICE)
# ==========================================
# 4. Training Pipeline (With Physics-Informed Loss)
# ==========================================
def train_model(dataset):
    model = PITTModel(IN_CHANNELS, VOCAB_SIZE, D_MODEL, FNO_MODES, FNO_WIDTH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = nn.L1Loss() # Standard MAE Loss
    
    X_train = dataset[:-1].to(DEVICE)
    Y_train = dataset[1:].to(DEVICE)
    num_samples = len(X_train)
    
    tokens = tokenize_equation(nu_val=1.0/REYNOLDS, Re=REYNOLDS).unsqueeze(0).to(DEVICE)
    loss_history = []

    print(f"\n--- Training PITT Model on {num_samples} Real Physics Frames ---")
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            batch_tokens = tokens.repeat(len(batch_x), 1)

            optimizer.zero_grad()
            predictions = model(batch_x, batch_tokens)
            
            # 1. Standard Data-Driven Loss (Match the pixels)
            loss_mae = criterion(predictions, batch_y)
            
            # 2. Physics-Informed Constraint (Penalize non-zero divergence)
            # Extract u and v predictions [Batch, Height, Width]
            u_pred = predictions[:, 0, :, :]
            v_pred = predictions[:, 1, :, :]
            
            # Compute central difference derivatives (matching the classical solver)
            du_dx = (u_pred[:, 1:-1, 2:] - u_pred[:, 1:-1, :-2]) / (2 * DX)
            dv_dy = (v_pred[:, 2:, 1:-1] - v_pred[:, :-2, 1:-1]) / (2 * DY)
            
            # Divergence = du/dx + dv/dy. We want this to be exactly 0.
            divergence = du_dx + dv_dy
            loss_div = torch.mean(torch.abs(divergence))
            
            # Combine the losses (lambda weight determines how strictly to enforce physics)
            lambda_physics = 0.1 
            loss = loss_mae + (lambda_physics * loss_div)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / (num_samples / BATCH_SIZE)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Total Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
            
    return model, loss_history, dataset[-1].unsqueeze(0).to(DEVICE)

# ==========================================
# 5. Diagnostic Plotting (Exactly as requested)
# ==========================================
def evaluate_and_plot(model, loss_history, eval_input):
    print("\n--- Generating CFD Plots from Trained Network ---")
    model.eval()
    tokens = tokenize_equation().unsqueeze(0).to(DEVICE)
    
    # The neural net predicts the fluid state
    with torch.no_grad():
        prediction = model(eval_input, tokens).cpu().numpy()[0]
    
    u, v, p = prediction[0], prediction[1], prediction[2]
    
    x_arr = np.linspace(0, L_DOM, GRID_SIZE)
    y_arr = np.linspace(0, L_DOM, GRID_SIZE)
    X, Y = np.meshgrid(x_arr, y_arr)

    # ── 1: Main Diagnostic Plot ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    cf1 = ax1.contourf(X, Y, p, levels=25, cmap='turbo', alpha=0.85)
    plt.colorbar(cf1, ax=ax1, label='Pressure [Pa]')
    ct1 = ax1.contour(X, Y, p, levels=12, colors='k', linewidths=0.5, alpha=0.4)
    ax1.clabel(ct1, inline=True, fontsize=7)
    ax1.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color='white', scale=12, width=0.003, alpha=0.9)
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title(f'Learned Pressure + Velocity Quiver (Re={REYNOLDS})', fontsize=11)

    ax2 = fig.add_subplot(gs[0, 2])
    speed = np.sqrt(u**2 + v**2)
    cf2  = ax2.contourf(X, Y, speed, levels=25, cmap='plasma')
    plt.colorbar(cf2, ax=ax2, label='Speed |u| [m/s]')
    ax2.streamplot(x_arr, y_arr, u, v, color='white', linewidth=0.7, density=2.0)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    ax2.set_title('Learned Speed + Streamlines')

    ax3 = fig.add_subplot(gs[1, 0])
    mid_x = GRID_SIZE // 2
    ax3.plot(u[:, mid_x], y_arr, 'b-', lw=2.5, label='PITT (PyTorch)')
    ax3.axvline(0, color='gray', lw=0.8, ls='--')
    ax3.fill_betweenx(y_arr, 0, u[:, mid_x], alpha=0.12, color='blue')
    ax3.set_xlabel('u-velocity [m/s]'); ax3.set_ylabel('y [m]')
    ax3.set_title('u-profile at x = 0.5')
    ax3.grid(alpha=0.3); ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    mid_y = GRID_SIZE // 2
    ax4.plot(x_arr, v[mid_y, :], 'r-', lw=2.5, label='PITT (PyTorch)')
    ax4.axhline(0, color='gray', lw=0.8, ls='--')
    ax4.fill_between(x_arr, 0, v[mid_y, :], alpha=0.12, color='red')
    ax4.set_xlabel('x [m]'); ax4.set_ylabel('v-velocity [m/s]')
    ax4.set_title('v-profile at y = 0.5')
    ax4.grid(alpha=0.3); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.semilogy(loss_history, color='#185FA5', lw=2, label='Training Loss (MAE)')
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss')
    ax5.set_title('PITT PyTorch Learning Curve')
    ax5.grid(alpha=0.3); ax5.legend()

    plt.savefig("pitt_learned_diagnostics.png", dpi=150, bbox_inches='tight')
    print("Saved -> pitt_learned_diagnostics.png")

    # ── 2: Pressure Analysis ──────────────────────────────────────────────────
    dp_dx = np.gradient(p, DX, axis=1)
    dp_dy = np.gradient(p, DY, axis=0)
    
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'Learned Pressure Analysis | Grid={GRID_SIZE}x{GRID_SIZE}', fontsize=13)

    ax = axes[0]
    cf = ax.contourf(X, Y, p, levels=30, cmap='RdBu_r', alpha=0.80)
    plt.colorbar(cf, ax=ax, label='Pressure [Pa]')
    strm = ax.streamplot(x_arr, y_arr, -dp_dx, -dp_dy, color=np.sqrt(dp_dx**2 + dp_dy**2), cmap='hot', linewidth=1.2)
    plt.colorbar(strm.lines, ax=ax, label='|∇p|')
    ax.set_title('Pressure field + −∇p streamlines')

    ax = axes[1]
    cf2 = ax.contourf(X, Y, p, levels=30, cmap='coolwarm', alpha=0.75)
    plt.colorbar(cf2, ax=ax, label='Pressure [Pa]')
    strm2 = ax.streamplot(x_arr, y_arr, u, v, color=np.sqrt(u**2 + v**2), cmap='Greens', linewidth=1.2)
    plt.colorbar(strm2.lines, ax=ax, label='Speed |u| [m/s]')
    ax.set_title('Pressure contours + velocity streamlines')

    ax = axes[2]
    cf3 = ax.contourf(X, Y, p, levels=40, cmap='turbo')
    plt.colorbar(cf3, ax=ax, label='Pressure [Pa]')
    ct3 = ax.contour(X, Y, p, levels=12, colors='white', linewidths=1.0)
    ax.clabel(ct3, inline=True, fontsize=8, fmt='%.3f', colors='white')
    p_max, p_min = np.unravel_index(np.argmax(p), p.shape), np.unravel_index(np.argmin(p), p.shape)
    ax.plot(x_arr[p_max[1]], y_arr[p_max[0]], 'w^', ms=9, label=f'p_max={p.max():.3f}')
    ax.plot(x_arr[p_min[1]], y_arr[p_min[0]], 'wv', ms=9, label=f'p_min={p.min():.3f}')
    ax.legend(loc='lower left', facecolor='#222', labelcolor='white')
    ax.set_title('Learned Pressure map')

    plt.savefig("pitt_learned_pressure.png", dpi=150, bbox_inches='tight')
    print("Saved -> pitt_learned_pressure.png")

# ==========================================
# Run Execution
# ==========================================
if __name__ == "__main__":
    dataset = generate_real_cfd_data()
    trained_model, loss_history, eval_frame = train_model(dataset)
    evaluate_and_plot(trained_model, loss_history, eval_frame)
    print("Pipeline completed successfully.")