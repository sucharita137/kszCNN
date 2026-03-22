# In[1]:


import os, time, random
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# In[2]:


GRID_FILE = os.path.join(DATA_DIR,"Grids_Mcdm_IllustrisTNG_1P_128_z=0.0.npy")

TRAIN_REAL_IDX = 0
TEST_REAL_IDX  = 14

TRAIN_HALO_FILE = os.path.join(DATA_DIR,"groups_090_1P_0.hdf5")
TEST_HALO_FILE  = os.path.join(DATA_DIR,"groups_090_1P_p2_n1.hdf5")


# In[3]:


PATCH = 32
BATCH = 16
EPOCHS = 100
LR = 3e-4
BOXSIZE = 25.0
MASS_CUT = 1e11
MAX_HALOS = 5000
SMOOTH_SCALE = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints_improved_v"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# In[4]:


def memmap_grid_slices(grid_file, idxs):
    if isinstance(idxs, int): idxs = [idxs]
    arr = np.load(grid_file, allow_pickle=False, mmap_mode='r')
    return [np.asarray(arr[i], dtype=np.float32) for i in idxs]


# In[5]:


def load_halos(hfile):
    with h5py.File(hfile, "r") as f:
        pos = np.array(f["Group/GroupPos"]) / 1000.0  # ckpc/h -> Mpc/h
        vel = np.array(f["Group/GroupVel"])
        mass = np.array(f["Group/Group_M_Mean200"]) * 1e10
    return pos, vel, mass


# In[6]:


def smooth_density_kspace(rho_cdm, R_smooth, boxsize=BOXSIZE):
    rho = np.asarray(rho_cdm, dtype=np.float32)
    delta = rho / rho.mean() - 1.0
    if R_smooth == 0.0:
        return delta.astype(np.float32)
    N = rho.shape[0]
    dk = fftn(delta)
    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq, 2*np.pi*kfreq, 2*np.pi*kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    W = np.exp(-0.5 * k2 * (R_smooth**2))
    return ifftn(dk * W).real.astype(np.float32)


# In[7]:


def compute_vlin_from_density(rho_cdm, z_snap=0.0, boxsize=BOXSIZE,
                              H0=67.66, Omega_m0=0.3, R_smooth=None):
    N = rho_cdm.shape[0]
    a = 1.0 / (1.0 + z_snap)
    Hz = H0 * np.sqrt(Omega_m0 * (1+z_snap)**3 + 1.0 - Omega_m0)
    f = (Omega_m0*(1+z_snap)**3 / (Omega_m0*(1+z_snap)**3 + 1.0 - Omega_m0))**0.55
    delta_x = rho_cdm / np.mean(rho_cdm) - 1.0
    dk = fftn(delta_x)
    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq, 2*np.pi*kfreq, 2*np.pi*kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_nozero = np.where(k2 == 0, 1.0, k2)
    pref = 1j * a * Hz * f
    vz_k = pref * (kz / k2_nozero) * dk
    vz_k[k2 == 0] = 0.0
    if R_smooth is not None and R_smooth > 0.0:
        W = np.exp(-0.5 * k2 * (R_smooth**2))
        vz_k *= W
    vz_x = ifftn(vz_k).real
    return vz_x.astype(np.float32)


# In[8]:


def build_vlin_interpolator(vlin_grid):
    N = vlin_grid.shape[0]
    cell = BOXSIZE / N
    coords = (np.arange(N) + 0.5) * cell
    return RegularGridInterpolator((coords, coords, coords), vlin_grid,
                                   bounds_error=False, fill_value=0.0)


# In[9]:


def smooth_velocity_grid(vz_grid, R_smooth):
    if R_smooth == 0:
        return vz_grid.astype(np.float32)

    N = vz_grid.shape[0]
    dk = fftn(vz_grid)

    kfreq = fftfreq(N, d=BOXSIZE/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq, 2*np.pi*kfreq, 2*np.pi*kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2

    W = np.exp(-0.5 * k2 * R_smooth**2)

    return ifftn(dk * W).real.astype(np.float32)


# In[10]:


def vlin_at_halos(vlin_grid, halo_pos):
    interp = build_vlin_interpolator(vlin_grid)
    pos_wrapped = (halo_pos % BOXSIZE)
    vals = [np.asarray(interp(tuple(p))).item() for p in pos_wrapped]
    return np.array(vals, dtype=np.float32)


# In[11]:


def extract_patch(grid, center, patch, boxsize):

    N = grid.shape[0]
    cell = boxsize / N

    idx = (center / cell - 0.5).astype(int)
    r = patch // 2

    xs = [(idx[0] + i) % N for i in range(-r, r)]
    ys = [(idx[1] + i) % N for i in range(-r, r)]
    zs = [(idx[2] + i) % N for i in range(-r, r)]

    return grid[np.ix_(xs, ys, zs)]


def redshift_err(pos, sigma_z=1e-4, H0=67.7, h=0.677, c=3e5):

    # convert to Mpc
    x_mpc = pos[:,2] / h

    # compute redshift
    z_true = H0 * x_mpc / c

    z_obs = z_true + np.random.normal(0, sigma_z, size=len(z_true))

    # convert back to Mpc
    x_err_mpc = c * z_obs / H0

    # convert back to Mpc/h
    x_err = x_err_mpc * h
    pos_err = pos.copy()
    pos_err[:,2] = x_err

    return pos_err


# In[12]:


class HaloDataset(Dataset):

    def __init__(self, density_grid, pos, vel,mass, 
                 patch=PATCH, boxsize=BOXSIZE,
                 mass_cut=MASS_CUT, max_n=MAX_HALOS,rng=None): # vel replaced by vz_smooth

        if rng is None:
            rng = np.random.RandomState(SEED)
        mask = mass > mass_cut
        pos, vel = pos[mask], vel[mask]

        
        if max_n is not None and len(pos) > max_n:
            sel = rng.choice(len(pos), max_n, replace=False)
            pos, vel = pos[sel], vel[sel]


        self.grid = density_grid
        self.pos = pos
        self.vz = vel[:,2].astype(np.float32) 

        self.patch = patch
        self.boxsize = boxsize

        print(f"Selected halos: {len(self.pos)}")

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):

        # coordinate channels only 
        Ngrid = self.grid.shape[0]
        cell = self.boxsize / Ngrid

        center_idx = (self.pos[i] / cell - 0.5)
        r = self.patch // 2
        coords = np.arange(-r, r)

        xg, yg, zg = np.meshgrid(coords, coords, coords, indexing='ij')

        xg = (xg + center_idx[0]) % Ngrid
        yg = (yg + center_idx[1]) % Ngrid
        zg = (zg + center_idx[2]) % Ngrid

        # normalize to [0,1]
        xg = (xg * cell) / self.boxsize
        yg = (yg * cell) / self.boxsize
        zg = (zg * cell) / self.boxsize

        #stack
        x_all = np.stack([xg, yg, zg], axis=0)

        x = torch.tensor(x_all, dtype=torch.float32)
        y = torch.tensor(self.vz[i], dtype=torch.float32)

        return x, y


# In[13]:


class CNN(nn.Module):

    def __init__(self, in_ch=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x).view(-1)


# In[14]:


def pearson_corr_torch(x, y, eps=1e-6):
    # x, y: tensors (batch,)
    xm = torch.mean(x)
    ym = torch.mean(y)
    xm0 = x - xm
    ym0 = y - ym
    cov = torch.mean(xm0 * ym0)
    sx = torch.sqrt(torch.mean(xm0 * xm0) + eps)
    sy = torch.sqrt(torch.mean(ym0 * ym0) + eps)
    corr = cov / (sx * sy + eps)
    return corr


# In[15]:


def _compute_stats(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return mask, np.nan, np.nan, np.nan, n

    t = true[mask]
    p = pred[mask]

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rms_true = float(np.std(t))
    rms_pred = float(np.std(p))

    return mask, rmse, rms_true, rms_pred, n


# In[16]:


def hexbin_panel(ax, true, pred, title, cmap="viridis"):
    mask, rmse, rms_true, rms_pred, n = _compute_stats(true, pred)
    
    # residuals 
    residual = pred[mask] - true[mask]
    abs_res = np.abs(residual)

    #  percentile errors 
    p68 = np.percentile(abs_res, 68)
    p95 = np.percentile(abs_res, 95)
    p99 = np.percentile(abs_res, 99)
    
    
    vmax = max(np.max(np.abs(true[mask])), np.max(np.abs(pred[mask]))) if n > 0 else 1.0
    lims = [-vmax, vmax]

    hb = ax.hexbin(
        true[mask], pred[mask],
        gridsize=150,
        cmap=cmap,
        bins='log',
        mincnt=1,
        extent=(lims[0], lims[1], lims[0], lims[1])
    )

    ax.plot(lims, lims, 'r--', lw=1.2, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("True LOS velocity (km/s)")
    ax.set_ylabel("Predicted LOS velocity (km/s)")
    ax.set_title(title)

    stats_text = (
        f"RMSE = {rmse:.2f}\n"
        f"p68 = {p68:.2f}\n"
        f"p95 = {p95:.2f}\n"
        f"p99 = {p99:.2f}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85)
    )
    return hb




# In[26]:


def train_model(model, tr_loader, val_loader, y_mean, y_std):

    model = model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    best_val = np.inf

    for epoch in range(EPOCHS):

        #  train 
        model.train()
        train_loss = 0
        train_pred, train_true = [], []

        for xb, y in tr_loader:

            xb = xb.to(DEVICE)
            y = y.to(DEVICE)

            y_n = (y - y_mean) / y_std

            pred_n = model(xb)
            loss = loss_fn(pred_n, y_n)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)

            pred = pred_n * y_std + y_mean
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(y.cpu().numpy())

        train_loss /= len(tr_loader.dataset)
        train_pred = np.concatenate(train_pred)
        train_true = np.concatenate(train_true)

        #train_rho = pearsonr(train_true, train_pred)[0]

        #  validation
        model.eval()
        val_loss = 0
        val_pred, val_true = [], []

        with torch.no_grad():
            for xb, y in val_loader:

                xb = xb.to(DEVICE)
                y = y.to(DEVICE)

                y_n = (y - y_mean) / y_std
                pred_n = model(xb)

                val_loss += ((pred_n - y_n)**2).mean().item() * xb.size(0)

                pred = pred_n * y_std + y_mean
                val_pred.append(pred.cpu().numpy())
                val_true.append(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        val_pred = np.concatenate(val_pred)
        val_true = np.concatenate(val_true)

        #val_rho = pearsonr(val_true, val_pred)[0]

        print(
            f"Epoch {epoch+1}: "
            f"Train={train_loss:.3e} | "
            f"Val={val_loss:.3e}"
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR + "/best.pth")

    return model


# In[27]:


def evaluate_model(model, te_loader, test_ds,
                   y_mean, y_std,
                   vlin_baseline, rho_vlin):

    model.load_state_dict(
        torch.load(CHECKPOINT_DIR + "/best.pth")
    )

    model.eval()

    preds = []

    with torch.no_grad():
        for xb, y in te_loader:

            xb = xb.to(DEVICE)

            pred = model(xb)
            pred = pred.cpu().numpy() * y_std + y_mean

            preds.append(pred)

    preds = np.concatenate(preds)
    true = test_ds.vz 
    
    # Residual histogram
    residual_cnn = preds - true
    residual_lin = vlin_baseline - true

    plt.figure(figsize=(6,5))
    plt.hist(residual_cnn, bins=50, alpha=0.6, label="CNN")
    plt.hist(residual_lin, bins=50, alpha=0.6, label="Linear")
    plt.xlabel("Velocity residual (km/s)")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("Velocity residual distribution")
    plt.show()

    #Residual vs true
    plt.figure(figsize=(6,5))
    plt.scatter(true, residual_cnn, s=5, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True velocity")
    plt.ylabel("Residual (pred - true)")
    plt.title("Residual vs True")
    plt.show()

    #Percentile errors
    abs_res = np.abs(residual_cnn)
    p68 = np.percentile(abs_res, 68)
    p95 = np.percentile(abs_res, 95)
    p99 = np.percentile(abs_res, 99)
    print("p68 =", p68)
    print("p95 =", p95)
    print("p99 =", p99)

    #Tail points
    threshold = np.percentile(abs_res, 95)
    tail_mask = abs_res > threshold
    tail_true = true[tail_mask]
    tail_pred = preds[tail_mask]

    plt.figure(figsize=(6,5))
    plt.scatter(true, preds, s=5, alpha=0.5)
    plt.scatter(tail_true, tail_pred, color='red', s=10)
    plt.plot([-500,500], [-500,500], 'k--')
    plt.xlabel("True velocity")
    plt.ylabel("Predicted velocity")
    plt.title("High scatter tail points")
    plt.show()
    
    
    rmse = np.sqrt(np.mean((preds - true)**2))
    rms_true = np.std(true)
    rms_pred = np.std(preds)
    N = len(true)

    print("\n===== TEST RESULTS =====")
    print(f"N = {N}")
    print(f"RMSE = {rmse:.3f}")
    print(f"RMS(true) = {rms_true:.3f}")
    print(f"RMS(pred) = {rms_pred:.3f}")
    # plots
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    hb0 = hexbin_panel(ax[0], true, vlin_baseline, "Linear")
    hb1 = hexbin_panel(ax[1], true, preds, "CNN")

    cb0 = fig.colorbar(hb0, ax=ax[0])
    cb0.set_label("Counts")

    cb1 = fig.colorbar(hb1, ax=ax[1])
    cb1.set_label("Counts")

    plt.tight_layout()
    plt.show()

