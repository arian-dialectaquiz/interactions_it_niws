"""
extract_surface.py
------------------
One pass per experiment over the two-dimensional fields of the history files,
cached on a 3 km grid for Figure 1 and for the wind work of the later figures.

Why this exists. The three-dimensional variables of the history files cost a
145 MB slab per record whatever is read from them, while the two-dimensional
ones cost a few megabytes, so the sea surface elevation and the surface stress
of the whole record can be held for all four experiments at the price of a few
minutes each. The elevation gives the balanced vorticity at every hour of every
experiment through geostrophy, where the epoch means of the model vorticity used
before gave three windows of one experiment and two of another, and the stress
gives the forcing in space and time.

Each field is averaged over blocks of SUB by SUB cells with the land mask as
weight, which removes the aliasing a plain subsample would leave and brings the
grid to the 3 km spacing of the area maps. The stress is moved to the centres of
the cells and rotated to east and north before the average.

  python -u extract_surface.py            all four experiments
  python -u extract_surface.py FULL BACK  a subset

A cache already on disk is left alone.

Outputs (in OUTDIR): surface_BACK.nc, surface_TIDE.nc, surface_WIND.nc,
surface_FULL.nc
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset

ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
RUNS = {'BACK': ROOT + 'avg_GLORYS_ONLY_HC.nc',
        'TIDE': ROOT + 'avg_TIDES_ONLY_HC.nc',
        'WIND': ROOT + 'avg_1km_NIWS_HC.nc',
        'FULL': ROOT + 'avg_FULL_HC.nc'}
GRID = ('/home/arian/dd_waves/pyroms_tools/data/2004_paper_2/1km/'
        'paper_2_1km_closed_cropped_smooth_sponge.nc')
OUTDIR = './surface/'
os.makedirs(OUTDIR, exist_ok=True)

SUB = 3                    # block size in cells, the grid of the area maps
REPORT_EVERY = 100
FILL_LIMIT = 1.0e30        # anything above this is a fill value

TODO = [a.upper() for a in sys.argv[1:]] or list(RUNS)
for r in TODO:
    if r not in RUNS:
        raise SystemExit(f'{r} is not one of {list(RUNS)}')

# --------------------------------------------------------------------------
# the records the four experiments share, matched on stamps rounded to the
# second because the model writes a sub-second part that differs between runs
# --------------------------------------------------------------------------
TIMES = {r: xr.open_dataset(p).ocean_time.values for r, p in RUNS.items()}
KEY = {r: v.astype('datetime64[s]') for r, v in TIMES.items()}
t_key = KEY[list(RUNS)[0]]
for v in KEY.values():
    t_key = np.intersect1d(t_key, v)
KEEP = {}
for r in RUNS:
    k = np.searchsorted(KEY[r], t_key)
    assert np.array_equal(KEY[r][k], t_key), f'the record times of {r} do not match'
    KEEP[r] = k
print(f'{t_key.size} records shared by the four experiments, '
      f'{str(t_key[0])[:16]} to {str(t_key[-1])[:16]}')

# --------------------------------------------------------------------------
# the grid and its block average
# --------------------------------------------------------------------------
ref = xr.open_dataset(RUNS['FULL'])
h, lon, lat = ref.h.values, ref.lon_rho.values, ref.lat_rho.values
mask, f = ref.mask_rho.values, ref.f.values
pm, pn = ref.pm.values, ref.pn.values
angle = ref.angle.values if 'angle' in ref else xr.open_dataset(GRID).angle.values
neta, nxi = h.shape
ny, nx = neta//SUB, nxi//SUB
cut = (slice(0, ny*SUB), slice(0, nx*SUB))
w = mask[cut].astype(np.float64)
wsum = w.reshape(ny, SUB, nx, SUB).sum(axis=(1, 3))
ocean = wsum >= 0.5*SUB*SUB           # a block is ocean when most of it is


def block(a, weight=True):
    """Block mean of a field on rho points, weighted by the land mask."""
    a = a[cut].astype(np.float64)
    if not weight:
        return a.reshape(ny, SUB, nx, SUB).mean(axis=(1, 3))
    s = (a*w).reshape(ny, SUB, nx, SUB).sum(axis=(1, 3))
    return np.where(wsum > 0, s/np.maximum(wsum, 1.0), np.nan)


def clean(a):
    """Fill values to zero, they carry no weight in the block mean."""
    return np.where(np.abs(a) > FILL_LIMIT, 0.0, a)


eta_c = np.arange(ny)*SUB + SUB//2
xi_c = np.arange(nx)*SUB + SUB//2
grid = dict(lon=block(lon, False), lat=block(lat, False), h=block(h, False),
            f=block(f, False), dx=SUB*block(1.0/pm, False), dy=SUB*block(1.0/pn, False))
print(f'grid {neta} x {nxi} averaged in blocks of {SUB} to {ny} x {nx}, '
      f'mean spacing {np.nanmean(grid["dx"])/1e3:.2f} x {np.nanmean(grid["dy"])/1e3:.2f} km')

cosa, sina = np.cos(angle), np.sin(angle)

for run in TODO:
    fout = OUTDIR + f'surface_{run}.nc'
    if os.path.exists(fout):
        print(f'{fout} is already on disk, nothing to do for {run}')
        continue
    print(f'--- {run} ---', flush=True)
    kf = KEEP[run]
    t = TIMES[run][kf]
    nc = Dataset(RUNS[run])
    nc.set_auto_mask(False)
    vz = nc.variables['zeta']
    has_str = 'sustr' in nc.variables and 'svstr' in nc.variables
    if not has_str:
        print('   no surface stress in this experiment')
    Z = np.full((t.size, ny, nx), np.nan, np.float32)
    if has_str:
        TE = np.full((t.size, ny, nx), np.nan, np.float32)
        TN = np.full((t.size, ny, nx), np.nan, np.float32)
        tx, ty = np.zeros((neta, nxi)), np.zeros((neta, nxi))

    t0 = pd.Timestamp.now()
    for n, k in enumerate(kf):
        Z[n] = np.where(ocean, block(clean(vz[k])), np.nan)
        if has_str:
            su = clean(nc.variables['sustr'][k])       # on u points, (eta, xi - 1)
            sv = clean(nc.variables['svstr'][k])       # on v points, (eta - 1, xi)
            tx[:, 1:-1] = 0.5*(su[:, :-1] + su[:, 1:])
            tx[:, 0], tx[:, -1] = su[:, 0], su[:, -1]
            ty[1:-1, :] = 0.5*(sv[:-1, :] + sv[1:, :])
            ty[0, :], ty[-1, :] = sv[0, :], sv[-1, :]
            TE[n] = np.where(ocean, block(tx*cosa - ty*sina), np.nan)
            TN[n] = np.where(ocean, block(tx*sina + ty*cosa), np.nan)
        if n == 9 or (n + 1) % REPORT_EVERY == 0:
            el = (pd.Timestamp.now() - t0).total_seconds()
            print(f'{n+1:4d} of {t.size} records, {el/60:.1f} min, '
                  f'{(t.size - n - 1)*el/(n+1)/60:.0f} min left', flush=True)
    nc.close()

    out = {'zeta': (('ocean_time', 'y', 'x'), Z)}
    if has_str:
        out['taue'] = (('ocean_time', 'y', 'x'), TE)
        out['taun'] = (('ocean_time', 'y', 'x'), TN)
    res = xr.Dataset(out, coords=dict(ocean_time=t, y=np.arange(ny), x=np.arange(nx)))
    res = res.assign(**{k: (('y', 'x'), v.astype(np.float64)) for k, v in grid.items()},
                     ocean=(('y', 'x'), ocean), eta_c=('y', eta_c), xi_c=('x', xi_c))
    res.attrs.update(run=run, source=RUNS[run], sub=SUB,
                     note='block means weighted by the land mask, stress rotated to east and north',
                     t_start=str(t[0])[:19], t_end=str(t[-1])[:19])
    res.to_netcdf(fout, encoding={'ocean_time': {'units': 'nanoseconds since 2004-01-01',
                                                 'dtype': 'int64'}})
    print('wrote', fout, f'{os.path.getsize(fout)/1e6:.1f} MB in '
          f'{(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min', flush=True)

print('done')