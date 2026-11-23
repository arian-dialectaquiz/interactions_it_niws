"""
figS9_extract.py  (full_v5)  Supplementary Figure S9, step 1
------------------------------------------------------------
One footprint tile at the full 1 km resolution of the model, for the test of
the 3 km sampling.

Every energy term of the paper is formed on every third point of the 1 km
grid (fig_06_extract.py, SUB = 3). This script caches u, v and w of one
footprint tile at every point, over the same records and the same levels, so
that figS9_compute.py can form the works at 1 km and at coarser spacings on
identical columns.

The tile is the slab tile of fig_06_extract.py that holds the largest
footprint area (or the one named in TILE). Its rectangle runs from the first
to the last row and column of that tile, halo included, so every third point
of it is a point of the slab cache and the 3 km fields are reproduced exactly.
The reading follows fig_06_extract.py: one read of each record, u and v taken
at the index of the rho point, w averaged to the rho levels.

Revision 2. An experiment may be held in several history files (TIDE is now
avg_TIDES_ONLY_V2_part1 to part3). The records of the slab cache are looked up
across the parts by their time on the nearest hour, and a record that two parts
share is taken from the first. At the end every third point of the new cache is
compared with the slab cache of the same experiment. A difference of zero says
the history files are the output the caches of the paper were made from. For
TIDE this is the test that the V2 files reproduce the run of the paper, which
the wake (FULL minus TIDE) needs.

  cd ~/dd_waves/paper_3/full_v5
  nohup python -u figS9_extract.py FULL > log_figS9_extract_FULL.txt 2>&1 &
  nohup python -u figS9_extract.py TIDE > log_figS9_extract_TIDE.txt 2>&1 &
  tail -f log_figS9_extract_FULL.txt

Each run costs one pass over the records of its history file, about the time
fig_06_extract.py took for one experiment. A cache that is already complete is
left alone.

Outputs (in ./figS9_cache_<RUN>/)
  geometry.npz              the tile, its 1 km indices and the record times
  u.npy  v.npy  w.npy       (time, eta, xi, level), float32
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
PATTERNS = {'TIDE': ['avg_TIDES_ONLY_V2_part*.nc'],      # every file of an experiment, in any number of parts
            'FULL': ['avg_FULL_HC.nc']}
NEW_FULL = '../new_full/'
SLAB = [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/']      # the slab geometry
SLAB_RUN = {'TIDE': [NEW_FULL + 'fig6_cache_TIDE/'], 'FULL': SLAB}    # the slab cache of each experiment, for the check
FIG2 = './fig2_out/fig2_wake.nc'
TILE = None                    # None takes the tile with the largest footprint area, or give its number
MAX_CACHE_GB = 16.0            # refuse to allocate more than this for one run

RUN = (sys.argv[1] if len(sys.argv) > 1 else 'FULL').upper()
if RUN not in PATTERNS:
    raise SystemExit(f'the experiment must be one of {sorted(PATTERNS)}')
FILES = sorted(f for p in PATTERNS[RUN] for f in glob.glob(ROOT + p))
if not FILES:
    raise SystemExit(f'no history file of {RUN} matches {PATTERNS[RUN]} in {ROOT}')
CACHE = f'./figS9_cache_{RUN}/'
os.makedirs(CACHE, exist_ok=True)
print(f'experiment {RUN}\nsource     ' + '\n           '.join(FILES) + f'\ncache      {CACHE}', flush=True)


def to_hour(x):
    """Record stamps on the nearest hour, as in the compute scripts. The model writes each
    record a fraction of a second off the hour and the parts of a run may differ in it."""
    x = np.asarray(x).astype('datetime64[ns]')
    return (x + np.timedelta64(30, 'm')).astype('datetime64[h]').astype('datetime64[ns]')


# --------------------------------------------------------------------------
# the slab geometry and the tile
# --------------------------------------------------------------------------
slab = next((d for d in SLAB if os.path.exists(d + 'geometry.npz')), None)
if slab is None:
    raise SystemExit('no slab cache of FULL, the tiles are defined there')
geo = np.load(slab + 'geometry.npz', allow_pickle=True)
SUB = int(geo['SUB'])
ntiles = int(geo['ntiles'])
tw = np.asarray(geo['time']).astype('datetime64[s]')
NT = tw.size

f2 = xr.open_dataset(FIG2)
foot = f2.slope.values.astype(bool) & (f2.sector_code.values == 0)
area_s = f2.area.values
foot_area = np.zeros(ntiles)
for k in range(ntiles):
    b0, b1, c0 = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k])
    cols = slice(c0, c0 + geo[f'xs{k:03d}'].size)
    foot_area[k] = float(np.sum(area_s[b0:b1, cols][foot[b0:b1, cols]]))
k_tile = int(np.argmax(foot_area)) if TILE is None else int(TILE)
if foot_area[k_tile] == 0:
    raise SystemExit(f'tile {k_tile} holds no footprint point')
ys, xs = geo[f'ys{k_tile:03d}'], geo[f'xs{k_tile:03d}']
ys1 = np.arange(int(ys[0]), int(ys[-1]) + 1)           # every row of the 1 km grid
xs1 = np.arange(int(xs[0]), int(xs[-1]) + 1)
print(f'tile {k_tile} of {ntiles} (footprint area {foot_area[k_tile]/1e6:.0f} km2 of '
      f'{foot_area.sum()/1e6:.0f} km2), slab block {ys.size} x {xs.size} at {SUB} km, '
      f'1 km block {ys1.size} x {xs1.size}', flush=True)

# every record of the slab cache, located in the parts of the experiment
where = {}                                             # hour -> (part, record), the first part that holds it
for i, f in enumerate(FILES):
    tt = to_hour(xr.open_dataset(f).ocean_time.values)
    print(f'   part {i + 1}: {tt.size} records from {str(tt[0])[:13]} to {str(tt[-1])[:13]}', flush=True)
    for k, x in enumerate(tt):
        where.setdefault(x, (i, k))
twh = to_hour(tw)
missing = [str(x)[:13] for x in twh if x not in where]
if missing:
    raise SystemExit(f'{len(missing)} records of the slab cache are not in the history files of {RUN}, '
                     f'the first is {missing[0]} and the last {missing[-1]}')
kf = [where[x] for x in twh]
print('   records taken from each part: ' + ', '.join(f'part {i + 1} {sum(1 for a, _ in kf if a == i)}'
                                                     for i in range(len(FILES))), flush=True)

NCS = [Dataset(f) for f in FILES]
for nc_ in NCS:
    nc_.set_auto_mask(False)
vu, vv, vw = NCS[0].variables['u'], NCS[0].variables['v'], NCS[0].variables['w']
for nc_ in NCS[1:]:
    assert all(nc_.variables[v].shape[1:] == NCS[0].variables[v].shape[1:] for v in ('u', 'v', 'w')), \
        'the parts of the experiment are not on the same grid'
nlev = vu.shape[1]
gb = ys1.size*xs1.size*nlev*NT*4*3/1e9
print(f'{NT} records from {str(tw[0])[:16]} to {str(tw[-1])[:16]}, {nlev} levels, '
      f'cache {gb:.1f} GB for u, v and w', flush=True)
if gb > MAX_CACHE_GB:
    raise SystemExit(f'the cache exceeds {MAX_CACHE_GB} GB, choose a smaller TILE or raise MAX_CACHE_GB')

shp = (NT, ys1.size, xs1.size, nlev)


def cache_ready():
    g = CACHE + 'geometry.npz'
    if not os.path.exists(g):
        return False
    try:
        z = np.load(g, allow_pickle=True)
        if int(z['tile']) != k_tile or not np.array_equal(z['time'].astype('datetime64[s]'), tw):
            return False
        if not bool(z['complete']):
            return False
        for v in ('u', 'v', 'w'):
            if np.lib.format.open_memmap(CACHE + f'{v}.npy', mode='r').shape != shp:
                return False
    except Exception:
        return False
    return True


if cache_ready():
    raise SystemExit(f'{CACHE} already holds this tile over the same records, nothing to do')

meta = dict(tile=k_tile, SUB=SUB, ys=ys, xs=xs, ys1=ys1, xs1=xs1, time=tw, nlev=nlev, run=np.array(RUN),
            source=np.array(FILES))
np.savez(CACHE + 'geometry.npz', complete=False, **meta)
store = {v: np.lib.format.open_memmap(CACHE + f'{v}.npy', mode='w+', dtype=np.float32, shape=shp)
         for v in ('u', 'v', 'w')}


def window(var):
    """The contiguous hyperslab that covers the tile on the grid of a variable, and the
    indices inside it, clamped at the last row and column as in fig_06_extract.py."""
    jy = np.minimum(ys1, var.shape[-2] - 1)
    jx = np.minimum(xs1, var.shape[-1] - 1)
    return slice(int(jy[0]), int(jy[-1]) + 1), slice(int(jx[0]), int(jx[-1]) + 1), np.ix_(jy - jy[0], jx - jx[0])


WU, WV, WW = window(vu), window(vv), window(vw)

# --------------------------------------------------------------------------
# the single pass
# --------------------------------------------------------------------------
t0 = pd.Timestamp.now()
for n, (i, k) in enumerate(kf):
    var = NCS[i].variables
    su = var['u'][k, :, WU[0], WU[1]][:, WU[2][0], WU[2][1]]
    sv = var['v'][k, :, WV[0], WV[1]][:, WV[2][0], WV[2][1]]
    sw = var['w'][k, :, WW[0], WW[1]][:, WW[2][0], WW[2][1]]
    sw = 0.5*(sw[:-1] + sw[1:])
    store['u'][n] = np.moveaxis(su, 0, -1)
    store['v'][n] = np.moveaxis(sv, 0, -1)
    store['w'][n] = np.moveaxis(sw, 0, -1)
    if n == 9 or (n + 1) % 24 == 0 or n == NT - 1:
        el = (pd.Timestamp.now() - t0).total_seconds()
        left = (NT - n - 1)*el/(n + 1)
        print(f'{n + 1:4d} of {NT} records, {el/60:.1f} min, {el/(n + 1):.2f} s per record, '
              f'{left/60:.0f} min left, finish near '
              f'{(pd.Timestamp.now() + pd.Timedelta(seconds=left)).strftime("%d %b %H:%M")}', flush=True)
for nc_ in NCS:
    nc_.close()
for a in store.values():
    a.flush()

# --------------------------------------------------------------------------
# check. Every third point of the new cache against the slab cache of the
# same experiment, on eight records spread over the window
# --------------------------------------------------------------------------
chk = {v: np.nan for v in ('u', 'v', 'w')}
slab_run = next((d for d in SLAB_RUN[RUN] if os.path.exists(d + f'u_t{k_tile:03d}.npy')), None)
if slab_run is None:
    print(f'no slab cache of {RUN} to check against', flush=True)
else:
    rec = np.unique(np.linspace(0, NT - 1, 8).astype(int))
    for v in chk:
        c = np.load(slab_run + f'{v}_t{k_tile:03d}.npy', mmap_mode='r')
        chk[v] = float(np.nanmax(np.abs(np.asarray(store[v][rec][:, ::SUB, ::SUB]) - np.asarray(c[rec]))))
    worst = max(chk.values())
    print(f'check against {slab_run}: largest difference u {chk["u"]:.2e}, v {chk["v"]:.2e}, w {chk["w"]:.2e} m s-1',
          flush=True)
    if worst == 0:
        print('   the history files are the output the slab cache was made from', flush=True)
    else:
        print('   WARNING. The history files are NOT the output the slab cache of the paper was made from.\n'
              '   A difference at the level of rounding (under 1e-6 m s-1) is harmless. A larger one means the\n'
              '   run was repeated and its eddies may differ, and then FULL minus this run is no longer the\n'
              '   wake of the paper. Do not run figS9_compute.py before this is understood.', flush=True)
np.savez(CACHE + 'geometry.npz', complete=True, check_u=chk['u'], check_v=chk['v'], check_w=chk['w'], **meta)
print(f'cache for {RUN} written to {CACHE} in {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min')