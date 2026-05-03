#!/usr/bin/env python
# coding: utf-8

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


# Cosmology + Experiment Handling


# Fiducial values
FIDUCIAL = {
    "Omega_m": 0.3,
    "sigma8": 0.8
}

# Step sizes
STEP = {
    "Omega_m": 0.1,
    "sigma8": 0.1
}

# Mapping of parameter IDs
PARAM_MAP = {
    "p1": "Omega_m",
    "p2": "sigma8"
}


# Experiment configurations


EXPERIMENTS = {
    "same_real": {
        "train": ["1P_0"],
        "test": ["1P_0"],
        "label": "Same Realization"
    },
    "diff_real": {
        "train": ["1P_0"],
        "test": [
            "1P_p1_n1", "1P_p1_n2", "1P_p1_1",
            "1P_p2_n1", "1P_p2_n2"
        ],
        "label": "Different Cosmology"
    },
    "diff_subgrid": {
        "train": ["1P_0"],
        "test": ["SIMBA_1P_0"],
        "label": "Different Subgrid"
    }
}

# Input types


INPUT_TYPES = {
    "position": "Galaxy Position (Number Density)",
    "overdensity": "Galaxy Overdensity"
}


# Cosmology parser (ROBUST)


def parse_cosmology(set_name):
    params = FIDUCIAL.copy()
    
    parts = set_name.split("_")
    
    # Handle names like SIMBA_1P_0, IllustrisTNG_1P_0
    if "1P" not in parts:
        return params
    
    idx = parts.index("1P")
    
    # Fiducial case
    if parts[idx + 1] == "0":
        return params
    
    # Example: 1P_p2_n1
    param_id = parts[idx + 1]
    shift_str = parts[idx + 2]
    
    param_name = PARAM_MAP[param_id]
    
    if shift_str.startswith("n"):
        sign = -1
        magnitude = int(shift_str[1])
    else:
        sign = +1
        magnitude = int(shift_str)
    
    params[param_name] = (
        FIDUCIAL[param_name]
        + sign * STEP[param_name] * magnitude
    )
    
    return params


# File handling


def get_group_file(set_name, snapshot=90):
    return f"groups_{snapshot:03d}_{set_name}.hdf5"


def load_cosmology(set_name):
    file_path = get_group_file(set_name)
    params = parse_cosmology(set_name)
    return file_path, params


# Experiment helper


def get_experiment_sets(exp_type):
    exp = EXPERIMENTS[exp_type]
    return exp["train"], exp["test"]



# Title generator


def make_title(set_name, input_type, exp_type):
    
    params = parse_cosmology(set_name)
    
    if input_type == "position":
        label = r"$n_g$"
    elif input_type == "overdensity":
        label = r"$\delta_g$"
    else:
        label = input_type
    
    return (
        f"{label} | {set_name} | "
        + r"$\Omega_m={:.2f}, \sigma_8={:.2f}$".format(
            params["Omega_m"], params["sigma8"]
        )
    )



# In[3]:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# In[22]:

set_name = "1P_0"
input_type = "overdensity"
exp_type = "same_real"

halo_filename, params = load_cosmology(set_name)

HALO_FILE = os.path.join(DATA_DIR, halo_filename)

# In[3]:


PATCH = 32
BATCH = 16
EPOCHS = 100
LR = 3e-4
MASS_CUT = 1e11
BOXSIZE = 25.0
SMOOTH_SCALE = 2.0


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints_improved_v"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# In[4]:


def load_groups_and_subhalos(hfile):
    with h5py.File(hfile, "r") as f:
        #  halos (targets)
        halo_pos  = np.array(f["Group/GroupPos"]) / 1000.0   # Mpc/h
        halo_vel  = np.array(f["Group/GroupVel"])
        halo_mass = np.array(f["Group/Group_M_Mean200"]) * 1e10

        # subhalos (galaxies) 
        sub_pos   = np.array(f["Subhalo/SubhaloPos"]) / 1000.0
        sub_mstar = np.array(f["Subhalo/SubhaloMassType"])[:, 4] * 1e10

    return halo_pos, halo_vel, halo_mass, sub_pos, sub_mstar



def build_galaxy_density_grid(subhalo_pos, mstar, Ngrid=128,
                             boxsize=BOXSIZE, mstar_cut=1e8):

    # selection 
    mask = mstar > mstar_cut
    pos = subhalo_pos[mask]

    print(f"Selected galaxies: {len(pos)}")

    # grid 
    grid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)

    # indices 
    cell = boxsize / Ngrid
    idx = (pos / cell).astype(int) % Ngrid

    # FAST counting
    np.add.at(grid, (idx[:,0], idx[:,1], idx[:,2]), 1)

    # overdensity 
    mean = grid.mean()
    delta_g = (grid - mean) / (mean + 1e-6)

    return delta_g


# In[6]:

def smooth_delta_field(delta, R_smooth, boxsize=BOXSIZE):

    if R_smooth == 0.0:
        return delta.astype(np.float32)

    N = delta.shape[0]

    dk = fftn(delta)

    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(
        2*np.pi*kfreq,
        2*np.pi*kfreq,
        2*np.pi*kfreq,
        indexing='ij'
    )

    k2 = kx**2 + ky**2 + kz**2

    W = np.exp(-0.5 * k2 * (R_smooth**2))

    return ifftn(dk * W).real.astype(np.float32)
# In[7]:


def compute_vlin_from_delta(delta_x, z_snap=0.0, boxsize=BOXSIZE,
                           H0=67.66, Omega_m0=0.3, R_smooth=None):

    N = delta_x.shape[0]
    a = 1.0 / (1.0 + z_snap)
    Hz = H0 * np.sqrt(Omega_m0 * (1+z_snap)**3 + 1.0 - Omega_m0)
    f = (Omega_m0*(1+z_snap)**3 /
         (Omega_m0*(1+z_snap)**3 + 1.0 - Omega_m0))**0.55

    dk = fftn(delta_x)

    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq,
                             2*np.pi*kfreq,
                             2*np.pi*kfreq,
                             indexing='ij')

    k2 = kx**2 + ky**2 + kz**2
    k2_nozero = np.where(k2 == 0, 1.0, k2)

    pref = 1j * a * Hz * f

    vz_k = pref * (kz / k2_nozero) * dk
    vz_k[k2 == 0] = 0.0

    if R_smooth is not None and R_smooth > 0:
        W = np.exp(-0.5 * k2 * R_smooth**2)
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



# In[13]:


class HaloDataset(Dataset):

    def __init__(self, density_grid, pos, vel,mass, 
                 patch=PATCH, boxsize=BOXSIZE,
                 mass_cut=MASS_CUT,rng=None): 

        if rng is None:
            rng = np.random.RandomState(SEED)
        mask = mass > mass_cut
        pos, vel = pos[mask], vel[mask]

        self.grid = density_grid
        self.pos = pos
        self.vz = vel[:,2].astype(np.float32) 

        self.patch = patch
        self.boxsize = boxsize

        print(f"Number of halos above mass 1e11: {len(self.pos)}")

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):

        patch = extract_patch(
            self.grid,
            self.pos[i],
            self.patch,
            self.boxsize
        )

        # local normalization
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        x = torch.tensor(patch[None], dtype=torch.float32)
        y = torch.tensor(self.vz[i], dtype=torch.float32)

        return x, y


# In[14]:


class CNN(nn.Module):

    def __init__(self, in_ch=1):
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
        # return self.fc(x).squeeze()


# In[15]:


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


# In[16]:


def _compute_stats(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return mask, np.nan, np.nan, np.nan, n

    t = true[mask]
    p = pred[mask]

    try:
        corr = float(pearsonr(t, p)[0])
    except Exception:
        corr = np.nan
        
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rms_true = float(np.std(t))
    rms_pred = float(np.std(p))
    
    m, b = np.polyfit(true, pred, 1)
    sigma_true = np.std(true)
    term_m = max(0, 1 - abs(m - 1))
    term_b = max(0, 1 - abs(b) / sigma_true)
    metric_val = corr * term_m * term_b
    
    return mask, corr, rmse, m,b, metric_val, rms_true, rms_pred, n

# In[17]:


def hexbin_panel(ax, true, pred, title, lims=None, cmap="viridis"):

    mask, corr, rmse, m, b, metric_val, rms_true, rms_pred, n = _compute_stats(true, pred)
    

    vmax = max(np.max(np.abs(true[mask])), np.max(np.abs(pred[mask])))
    lims = [-vmax, vmax]

    hb = ax.hexbin(
        true[mask], pred[mask],
        gridsize=150,
        cmap=cmap,
        bins='log',
        mincnt=1,
        extent=(lims[0], lims[1], lims[0], lims[1])
    )

    ax.plot(lims, lims, 'r--', lw=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    ax.set_xlabel("True LOS velocity (km/s)")
    ax.set_ylabel("Predicted LOS velocity (km/s)")
    ax.set_title(title)

    stats_text = (
        f"RMSE = {rmse:.2f}\n"
        f"r = {corr:.2f}\n"
        f"slope = {m:.2f}\n"
        #f"b = {b:.2f}\n"
        f"M_V = {metric_val:.2f}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9)
    )

    return hb





# In[46]:


def train_model(model, tr_loader, val_loader, y_mean, y_std):

    model = model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn =  nn.HuberLoss(delta=50)   #nn.HuberLoss(delta=50)nn.SmoothL1Loss()   

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

        # train_rho = pearsonr(train_true, train_pred)[0]

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


# In[47]:

def evaluate_model(model, te_loader, test_ds,
                   y_mean, y_std,
                   vlin_baseline, rho_vlin):

    model.load_state_dict(
        torch.load(CHECKPOINT_DIR + "/best.pth")
    )

    model.eval()

    preds = []
    true = []
    with torch.no_grad():
        for xb, y in te_loader:

            xb = xb.to(DEVICE)

            pred = model(xb)
            pred = pred.cpu().numpy() * y_std + y_mean

            preds.append(pred)
            true.append(y.numpy())

    preds = np.concatenate(preds)
    true = np.concatenate(true)
    corr = np.corrcoef(true, preds)[0,1]
    rmse = np.sqrt(np.mean((preds - true)**2))
    rms_true = np.std(true)
    rms_pred = np.std(preds)
    # linear fit
    m, b = np.polyfit(true, preds, 1)

    sigma_true = np.std(true)

    term_m = max(0, 1 - abs(m - 1))
    term_b = max(0, 1 - abs(b) / sigma_true)

    metric_val = corr * term_m * term_b

    # plots
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    hb0 = hexbin_panel(ax[0], true, vlin_baseline, "Linear")
    hb1 = hexbin_panel(ax[1], true, preds, "CNN")

    cb0 = fig.colorbar(hb0, ax=ax[0])
    cb0.set_label("Counts")

    cb1 = fig.colorbar(hb1, ax=ax[1])
    cb1.set_label("Counts")

    plt.title(make_title(set_name, input_type, exp_type))
    plt.tight_layout()
    plt.savefig("same_real_cnn_density.png")
    plt.show()










