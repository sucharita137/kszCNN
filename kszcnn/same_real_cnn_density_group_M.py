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

# Realization index mapping 
REAL_IDX_MAP = {
    "1P_0": 0,
    "1P_p1_n1": 4,
    "1P_p1_n2": 1,
    "1P_p1_1": 8,
    "1P_p1_2": 10,
    "1P_p2_n1": 14,
    "1P_p2_n2": 11,
    "1P_p2_1": 18,
    "1P_p2_2": 20
}

def parse_cosmology(set_name):
    params = FIDUCIAL.copy()
    
    parts = set_name.split("_")
    
    # Fiducial case
    if set_name == "1P_0":
        return params
    
    # Example: 1P_p2_n1
    param_id = parts[1]     # p2
    shift_str = parts[2]    # n1 
    
    param_name = PARAM_MAP[param_id]
    
    # Determine sign and magnitude
    if shift_str.startswith("n"):
        sign = -1
        magnitude = int(shift_str[1])
    else:
        sign = +1
        magnitude = int(shift_str)
    
    # Apply shift
    params[param_name] = FIDUCIAL[param_name] + sign * STEP[param_name] * magnitude
    
    return params


def get_group_file(set_name, snapshot=90):
    return f"groups_{snapshot:03d}_{set_name}.hdf5"


def load_cosmology(set_name):
    file_path = get_group_file(set_name)
    real_idx = REAL_IDX_MAP[set_name]
    params = parse_cosmology(set_name)
    
    return file_path, real_idx, params


def make_title(set_name):
    params = parse_cosmology(set_name)
    
    return ( r"Density+ GroupMass |"
        + f" {set_name} | "
        + r"$\Omega_m={:.2f}, \sigma_8={:.2f}$".format(
            params["Omega_m"], params["sigma8"]
        )
    )



# In[3]:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

GRID_FILE = os.path.join(DATA_DIR, "Grids_Mcdm_IllustrisTNG_1P_128_z=0.0.npy")


set_name = "1P_0"

halo_filename, REAL_IDX, params = load_cosmology(set_name)

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

def load_feature_file(hfile, feature_type="group_M"):

    with h5py.File(hfile, "r") as f:
        if feature_type == "group_M":
            feature = np.array(f["Group/GroupMass"]) * 1e10

        elif feature_type == "group_Mean":
            feature = np.array(f["Group/Group_M_Mean200"]) * 1e10

        elif feature_type == "nsubs":
            feature = np.array(f["Group/GroupNsubs"])

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    return feature

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


# In[39]:


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


# In[13]:

class HaloDataset(Dataset):

    def __init__(self, density_grid, pos, vel, mass, feature,
             feature_type="group_M",
             patch=PATCH, boxsize=BOXSIZE,
             mass_cut=MASS_CUT, rng=None):

        if rng is None:
            rng = np.random.RandomState(SEED)

        mask = mass > mass_cut
        pos, vel, mass = pos[mask], vel[mask], mass[mask]
        feature = feature[mask]

        self.grid = density_grid
        self.pos = pos
        self.mass = mass
        self.feature = feature
        self.feature_type = feature_type
        
        self.vz = vel[:, 2].astype(np.float32)

        self.patch = patch
        self.boxsize = boxsize

        
        if self.feature_type == "group_M":
            log_feat = np.log10(self.feature + 1e-6)

        elif self.feature_type == "nsubs":
            log_feat = np.log10(self.feature + 1)

        else:
            raise ValueError("Unknown feature_type")

        self.feat_mean = log_feat.mean()
        self.feat_std = log_feat.std() + 1e-6

        print(f"Selected halos: {len(self.pos)}")

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):

        # density patch
        patch = extract_patch(
            self.grid,
            self.pos[i],
            self.patch,
            self.boxsize
        )

        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        
        # halo mass channel 
        val = self.feature[i]

        if self.feature_type == "group_M":
            logv = np.log10(val + 1e-6)

        elif self.feature_type == "nsubs":
            logv = np.log10(val + 1)

        logv = (logv - self.feat_mean) / self.feat_std
        mchan = np.full_like(patch, logv)

        # stack channels 
        x_all = np.stack([patch,  mchan], axis=0)

        x = torch.tensor(x_all, dtype=torch.float32)
        y = torch.tensor(self.vz[i], dtype=torch.float32)

        return x, y


# In[14]:

class CNN(nn.Module):

    def __init__(self, in_ch=2):
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
        f"corr.coef r = {corr:.2f}\n"
        f"m = {m:.2f}\n"
        f"b = {b:.2f}\n"
        f"metric_val = {metric_val:.2f}"
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
            #feat = feat.to(DEVICE)
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
                #feat = feat.to(DEVICE)
                
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
        for xb,  y in te_loader:

            xb = xb.to(DEVICE)
            #feat = feat.to(DEVICE)
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

    plt.title(make_title(set_name))
    plt.tight_layout()
    plt.savefig("same_real_cnn_density_GroupMass.png")
    plt.show()










