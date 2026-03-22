# In[27]:


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


# In[29]:


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


# In[30]:


def memmap_grid_slices(grid_file, idxs):
    if isinstance(idxs, int): idxs = [idxs]
    arr = np.load(grid_file, allow_pickle=False, mmap_mode='r')
    return [np.asarray(arr[i], dtype=np.float32) for i in idxs]


# In[31]:


def load_halos(hfile):
    with h5py.File(hfile, "r") as f:
        pos = np.array(f["Group/GroupPos"]) / 1000.0  # ckpc/h -> Mpc/h
        vel = np.array(f["Group/GroupVel"])
        mass = np.array(f["Group/Group_M_Mean200"]) * 1e10
    return pos, vel, mass


# In[32]:


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


# In[33]:


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


# In[34]:


def build_vlin_interpolator(vlin_grid):
    N = vlin_grid.shape[0]
    cell = BOXSIZE / N
    coords = (np.arange(N) + 0.5) * cell
    return RegularGridInterpolator((coords, coords, coords), vlin_grid,
                                   bounds_error=False, fill_value=0.0)


# In[35]:


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


# In[36]:


def vlin_at_halos(vlin_grid, halo_pos):
    interp = build_vlin_interpolator(vlin_grid)
    pos_wrapped = (halo_pos % BOXSIZE)
    vals = [np.asarray(interp(tuple(p))).item() for p in pos_wrapped]
    return np.array(vals, dtype=np.float32)


# In[37]:


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

# In[38]:


class HaloDataset(Dataset):

    def __init__(self, density_grid, pos, vel, mass,
                 mode="density",
                 patch=PATCH, boxsize=BOXSIZE,
                 mass_cut=MASS_CUT, max_n=MAX_HALOS, rng=None):
        if rng is None:
            rng = np.random.RandomState(SEED)
        mask = mass > mass_cut
        pos, vel, mass = pos[mask], vel[mask], mass[mask]

        if max_n is not None and len(pos) > max_n:
            sel = rng.choice(len(pos), max_n, replace=False)
            pos, vel, mass = pos[sel], vel[sel], mass[sel]
            
        self.mode = mode
        self.grid = density_grid
        self.pos = pos
        self.mass = mass

        self.vz = vel[:, 2].astype(np.float32)

        self.patch = patch
        self.boxsize = boxsize

        # Normalize log mass globally 
        log_mass = np.log10(self.mass)
        self.mass_mean = log_mass.mean()
        self.mass_std = log_mass.std() + 1e-6

        print(f"Selected halos: {len(self.pos)}")

    def __len__(self):
        return len(self.pos)
    def __getitem__(self, i):
        inputs = []

        #  Density 
        if "density" in self.mode:
            patch = extract_patch(self.grid, self.pos[i], self.patch, self.boxsize)
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)
            inputs.append(patch)

        #  Position 
        if "pos" in self.mode:
            Ngrid = self.grid.shape[0]
            cell = self.boxsize / Ngrid

            center_idx = (self.pos[i] / cell - 0.5)
            r = self.patch // 2
            coords = np.arange(-r, r)

            xg, yg, zg = np.meshgrid(coords, coords, coords, indexing='ij')

            xg = (xg + center_idx[0]) % Ngrid
            yg = (yg + center_idx[1]) % Ngrid
            zg = (zg + center_idx[2]) % Ngrid

            xg = (xg * cell) / self.boxsize
            yg = (yg * cell) / self.boxsize
            zg = (zg * cell) / self.boxsize

            inputs.extend([xg, yg, zg])

        #  Halo mass 
        if "mass" in self.mode:
            logm = (np.log10(self.mass[i]) - self.mass_mean) / self.mass_std
            mchan = np.full_like(inputs[0], logm)
            inputs.append(mchan)

        x_all = np.stack(inputs, axis=0)

        x = torch.tensor(x_all, dtype=torch.float32)
        y = torch.tensor(self.vz[i], dtype=torch.float32)

        return x, y


# In[39]:


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


# In[40]:


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


# In[41]:


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


# In[51]:


def hexbin_panel(ax, true, pred, title, lims=None, cmap="viridis"):

    mask, rmse, rms_true, rms_pred, n = _compute_stats(true, pred)
    
    # residuals 
    residual = pred[mask] - true[mask]
    abs_res = np.abs(residual)

    #  percentile errors 
    p68 = np.percentile(abs_res, 68)
    p95 = np.percentile(abs_res, 95)
    p99 = np.percentile(abs_res, 99)

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
        f"p68 = {p68:.2f}\n"
        f"p95 = {p95:.2f}\n"
        f"p99 = {p99:.2f}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9)
    )

    return hb


# In[43]:


INPUT_MODES = {
    "density": 1,
    "density_pos": 4,
    "density_pos_mass": 5,
    "pos_mass": 4,
    "pos": 3,
}


# In[44]:


MODEL_TITLES = {
    "density": "Density",
    "density_pos": "Density + Position",
    "density_pos_mass": "Density + Position + Mass",
    "pos_mass": "Position + Mass",
    "pos": "Position"
}


# In[45]:


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


# In[46]:


#print(f"\nTrain realization: {TRAIN_REAL_IDX}")
#print(f"Test realization:  {TEST_REAL_IDX}")


# In[47]:


def run_experiment(mode, train_real, test_real):

    #  FIX RANDOMNESS
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\nRunning model: {mode}")

    # load grids
    grid_train, grid_test = memmap_grid_slices(
        GRID_FILE, [train_real, test_real]
    )

    delta_train = smooth_density_kspace(grid_train, SMOOTH_SCALE)
    delta_test  = smooth_density_kspace(grid_test, SMOOTH_SCALE)

    # halos
    pos_tr, vel_tr, mass_tr = load_halos(TRAIN_HALO_FILE)
    pos_te, vel_te, mass_te = load_halos(TEST_HALO_FILE)
    

    pos_tr = redshift_err(pos_tr, sigma_z=1e-4)
    pos_te = redshift_err(pos_te, sigma_z=1e-4)


    train_ds = HaloDataset(delta_train, pos_tr, vel_tr, mass_tr, mode=mode)
    test_ds  = HaloDataset(delta_test, pos_te, vel_te, mass_te, mode=mode)

    # split
    idxs = np.arange(len(train_ds))
    tr_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=SEED)

    tr_ds = Subset(train_ds, tr_idx)
    val_ds = Subset(train_ds, val_idx)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    te_loader  = DataLoader(test_ds, batch_size=BATCH)

    # model
    model = CNN(in_ch=INPUT_MODES[mode]).to(DEVICE)

    # velocity normalization
    y_train = np.array([train_ds[i][1] for i in tr_idx])
    y_mean = y_train.mean()
    y_std  = y_train.std() + 1e-12

    train_model(model, tr_loader, val_loader, y_mean, y_std)

    # Linear baseline 
    vlin_test = compute_vlin_from_density(grid_test, R_smooth=SMOOTH_SCALE)
    vlin_baseline = vlin_at_halos(vlin_test, test_ds.pos)

    # CNN prediction 
    preds = []
    with torch.no_grad():
        for xb, _ in te_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy() * y_std + y_mean
            preds.append(pred)

    preds = np.concatenate(preds)
    true = test_ds.vz

    #  Metrics 
    def metrics(true, pred):
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        rms_true = np.std(true)
        rms_pred = np.std(pred)
        return rmse, rms_true, rms_pred


    rmse_lin, rms_true, rms_lin = metrics(true, vlin_baseline)
    rmse_cnn, _, rms_cnn = metrics(true, preds)

    # residuals
    residual_lin = vlin_baseline - true
    residual_cnn = preds - true

    # percentile errors
    p68_lin = np.percentile(np.abs(residual_lin), 68)
    p95_lin = np.percentile(np.abs(residual_lin), 95)

    p68_cnn = np.percentile(np.abs(residual_cnn), 68)
    p95_cnn = np.percentile(np.abs(residual_cnn), 95)

    p99_cnn = np.percentile(np.abs(residual_cnn), 99)


    return (
        rms_true,
        rms_lin,
        rms_cnn,
        rmse_lin,
        rmse_cnn,
        p68_lin,
        p95_lin,
        p68_cnn,
        p95_cnn,
        p99_cnn,
        true,
        vlin_baseline,
        preds
        )


