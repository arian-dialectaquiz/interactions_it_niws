"""
extract_region.py  (full_v5)
----------------------------
A cache of u, v and w over the whole 1 km domain in blocks of SUB km, for the
maps of Figures 2 to 5. The slab caches of fig_06_extract.py hold only the
tiles around the 200 to 2000 m band, which is where the statistics are formed.
The maps of the paper show the whole bight, so the quantities they draw are
formed again from this cache by region_compute.py.

Each run gets ../new_full/region_cache_<RUN>/ with geometry.npz and three
memory-mapped arrays u.npy, v.npy and w.npy of shape (nt, ny, nx, nz), float32,
velocities on the grid axes (as the slab caches), w averaged to the rho levels,
block means weighted by the land mask. The records are those of the slab cache
of FULL (18 March to 9 April, hourly), matched on stamps rounded to the second.

Size. With SUB = 6 and every level, 232 x 109 x 40 x 529 x 4 bytes = 2.1 GB per
variable and run, 26 GB in all. LEVEL_STRIDE = 2 halves it.

Run under nohup, one run at a time or all four:
    nohup python -u extract_region.py > log_region.txt 2>&1 &
    nohup python -u extract_region.py FULL TIDE > log_region.txt 2>&1 &
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
RUNS = {'BACK': ROOT + 'avg_GLORYS_ONLY_HC.nc',
        'TIDE': ROOT + 'avg_TIDES_ONLY_HC.nc',
        'WIND': ROOT + 'avg_1km_NIWS_HC.nc',
        'FULL': ROOT + 'avg_FULL_HC.nc'}
SLAB_GEOMETRY = '../new_full/fig6_cache_FULL/geometry.npz'   # the records to keep
OUTROOT = '../new_full/'
SUB = 6                     # block size in cells
LEVEL_STRIDE = 1            # 1 keeps every level, 2 every other level
REPORT_EVERY = 50
FILL_LIMIT = 1.0e30

TODO = [a.upper() for a in sys.argv[1:]] or list(RUNS)
for r in TODO:
    if r not in RUNS:
        raise SystemExit(f'{r} is not one of {list(RUNS)}')


def to_second(x):
    return np.asarray(x).astype('datetime64[s]')


# --------------------------------------------------------------------------
# the records of the slab caches, matched in each run
# --------------------------------------------------------------------------
if not os.path.exists(SLAB_GEOMETRY):
    for alt in ('../new_full/fig6_cache/geometry.npz',):
        if os.path.exists(alt):
            SLAB_GEOMETRY = alt
geo_slab = np.load(SLAB_GEOMETRY, allow_pickle=True)
t_keep = to_second(geo_slab['time'])
KEEP = {}
for r, p in RUNS.items():
    tr = to_second(xr.open_dataset(p).ocean_time.values)
    k = np.searchsorted(tr, t_keep)
    k = np.clip(k, 0, tr.size - 1)
    assert np.array_equal(tr[k], t_keep), f'{r} does not hold every slab record'
    KEEP[r] = k
nt = t_keep.size
print(f'{nt} records, {str(t_keep[0])[:16]} to {str(t_keep[-1])[:16]}', flush=True)

# --------------------------------------------------------------------------
# the grid and its block average
# --------------------------------------------------------------------------
ref = xr.open_dataset(RUNS['FULL'])
h, lon, lat = ref.h.values, ref.lon_rho.values, ref.lat_rho.values
mask, f = ref.mask_rho.values, ref.f.values
pm, pn = ref.pm.values, ref.pn.values
angle = ref.angle.values if 'angle' in ref else np.zeros_like(h)
neta, nxi = h.shape
ny, nx = neta//SUB, nxi//SUB
cut = (slice(0, ny*SUB), slice(0, nx*SUB))
w_mask = mask[cut].astype(np.float64)
wsum = w_mask.reshape(ny, SUB, nx, SUB).sum(axis=(1, 3))
ocean = wsum >= 0.5*SUB*SUB
levels = np.arange(0, ref.sizes['s_rho'], LEVEL_STRIDE)
nz = levels.size


def block2(a, weight=True):
    a = a[cut].astype(np.float64)
    if not weight:
        return a.reshape(ny, SUB, nx, SUB).mean(axis=(1, 3))
    s = (a*w_mask).reshape(ny, SUB, nx, SUB).sum(axis=(1, 3))
    return np.where(wsum > 0, s/np.maximum(wsum, 1.0), np.nan)


def block3(a):
    """Block mean of (nz, eta, xi) on rho points, weighted by the land mask."""
    a = a[:, cut[0], cut[1]].astype(np.float32)
    s = (a*w_mask[None]).reshape(a.shape[0], ny, SUB, nx, SUB).sum(axis=(2, 4))
    return np.where(wsum[None] > 0, s/np.maximum(wsum[None], 1.0), 0.0).astype(np.float32)


def clean(a):
    return np.where(np.abs(a) > FILL_LIMIT, 0.0, a)


grid = dict(lon=block2(lon, False), lat=block2(lat, False), h=block2(h, False), f=block2(f, False),
            pm=1.0/(SUB*block2(1.0/pm, False)), pn=1.0/(SUB*block2(1.0/pn, False)),
            angle=block2(angle, False), mask=ocean.astype(np.int8))
print(f'grid {neta} x {nxi} in blocks of {SUB} to {ny} x {nx}, {nz} levels', flush=True)

for run in TODO:
    outdir = OUTROOT + f'region_cache_{run}/'
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(outdir + 'w.npy'):
        print(f'{outdir} is already on disk, nothing to do for {run}')
        continue
    print(f'--- {run} ---', flush=True)
    nc = Dataset(RUNS[run])
    nc.set_auto_mask(False)
    vu, vv, vw = nc.variables['u'], nc.variables['v'], nc.variables['w']
    U = np.lib.format.open_memmap(outdir + 'u.npy', mode='w+', dtype=np.float32, shape=(nt, ny, nx, nz))
    V = np.lib.format.open_memmap(outdir + 'v.npy', mode='w+', dtype=np.float32, shape=(nt, ny, nx, nz))
    W = np.lib.format.open_memmap(outdir + 'w.npy', mode='w+', dtype=np.float32, shape=(nt, ny, nx, nz))
    ur = np.zeros((nz, neta, nxi), np.float32)
    vr = np.zeros((nz, neta, nxi), np.float32)
    t0 = pd.Timestamp.now()
    for n, k in enumerate(KEEP[run]):
        u = clean(vu[k][levels])                       # (nz, eta, xi - 1)
        v = clean(vv[k][levels])                       # (nz, eta - 1, xi)
        ur[:, :, 1:-1] = 0.5*(u[:, :, :-1] + u[:, :, 1:])
        ur[:, :, 0], ur[:, :, -1] = u[:, :, 0], u[:, :, -1]
        vr[:, 1:-1, :] = 0.5*(v[:, :-1, :] + v[:, 1:, :])
        vr[:, 0, :], vr[:, -1, :] = v[:, 0, :], v[:, -1, :]
        ww = clean(vw[k])                              # (nz + 1 all levels, eta, xi) on the w levels
        wr = 0.5*(ww[levels] + ww[levels + 1])         # to the rho levels kept
        U[n] = np.moveaxis(block3(ur), 0, -1)
        V[n] = np.moveaxis(block3(vr), 0, -1)
        W[n] = np.moveaxis(block3(wr), 0, -1)
        if n == 9 or (n + 1) % REPORT_EVERY == 0:
            el = (pd.Timestamp.now() - t0).total_seconds()
            print(f'{n + 1:4d} of {nt} records, {el/60:.1f} min, {(nt - n - 1)*el/(n + 1)/60:.0f} min left',
                  flush=True)
    for a in (U, V, W):
        a.flush()
    del U, V, W
    nc.close()
    np.savez(outdir + 'geometry.npz', SUB=SUB, levels=levels, nz=nz, ny=ny, nx=nx, time=t_keep,
             eta_c=np.arange(ny)*SUB + SUB//2, xi_c=np.arange(nx)*SUB + SUB//2, source=RUNS[run], **grid)
    print(f'wrote {outdir} in {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min', flush=True)
print('done')