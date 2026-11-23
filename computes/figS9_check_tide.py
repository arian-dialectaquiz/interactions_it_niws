"""
figS9_check_tide.py  (full_v5)
------------------------------
Why does the 1 km cache of TIDE (from avg_TIDES_ONLY_V2_part*.nc) differ from the
slab cache of the paper (from the earlier avg_TIDES_ONLY_HC.nc)?

Every third point of the 1 km cache is set against the slab cache of the same
tile at every record and every level. Four things are told apart.

  a shift in time     the fields agree at another lag, and the mean flow agrees
  bad records         most records agree and a few do not (zeros, a part boundary)
  a run that diverges the difference is small at the first records and grows
  another realisation the difference is large from the first record and the mean
                      flow of the window differs

Reads ./figS9_cache_TIDE/ and ../new_full/fig6_cache_TIDE/. About one minute.

  cd ~/dd_waves/paper_3/full_v5
  python -u figS9_check_tide.py > log_figS9_check_tide.txt 2>&1
  cat log_figS9_check_tide.txt
"""

import glob
import numpy as np
import xarray as xr

ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
PATTERN = 'avg_TIDES_ONLY_V2_part*.nc'
CACHE1 = './figS9_cache_TIDE/'
SLAB = '../new_full/fig6_cache_TIDE/'
LAGS = np.arange(-6, 7)                          # hours
STEP = 24                                        # records between the lines of the table


def to_hour(x):
    x = np.asarray(x).astype('datetime64[ns]')
    return (x + np.timedelta64(30, 'm')).astype('datetime64[h]').astype('datetime64[ns]')


G = np.load(CACHE1 + 'geometry.npz', allow_pickle=True)
k_tile, SUB = int(G['tile']), int(G['SUB'])
t = to_hour(G['time'])
nt = t.size
a = {v: np.load(CACHE1 + f'{v}.npy', mmap_mode='r') for v in ('u', 'v')}
c = {v: np.load(SLAB + f'{v}_t{k_tile:03d}.npy', mmap_mode='r') for v in ('u', 'v')}
gs = np.load(SLAB + 'geometry.npz', allow_pickle=True)
ts = to_hour(gs['time'])
print(f'tile {k_tile}, {nt} records, 1 km cache {a["u"].shape}, slab cache {c["u"].shape}')
print(f'record times of the two caches equal: {np.array_equal(t, ts)}')

# the part each record came from
part = np.full(nt, -1)
files = sorted(glob.glob(ROOT + PATTERN))
where = {}
for i, f in enumerate(files):
    tt = to_hour(xr.open_dataset(f).ocean_time.values)
    raw = xr.open_dataset(f).ocean_time.values
    print(f'part {i + 1}: {tt.size} records, first {str(raw[0])[:23]}, last {str(raw[-1])[:23]}, '
          f'step {np.median(np.diff(raw))/np.timedelta64(1, "s"):.0f} s')
    for x in tt:
        where.setdefault(x, i)
part = np.array([where.get(x, -1) for x in t])


def both(n, src):
    """u and v of one record as one vector, the 1 km cache on the slab points or the slab cache."""
    if src == 'new':
        return np.concatenate([np.asarray(a[v][n][::SUB, ::SUB], dtype=np.float64).ravel() for v in ('u', 'v')])
    return np.concatenate([np.asarray(c[v][n], dtype=np.float64).ravel() for v in ('u', 'v')])


# --------------------------------------------------------------------------
# every record: rms of the difference, rms of each, correlation, zeros
# --------------------------------------------------------------------------
rms_d, rms_new, rms_old, cor, zeros = (np.zeros(nt) for _ in range(5))
mean_new, mean_old = 0.0, 0.0
for n in range(nt):
    x, y = both(n, 'new'), both(n, 'old')
    g = np.isfinite(x) & np.isfinite(y)
    rms_d[n], rms_new[n], rms_old[n] = np.sqrt(np.mean((x[g] - y[g])**2)), np.sqrt(np.mean(x[g]**2)), np.sqrt(np.mean(y[g]**2))
    cor[n] = np.corrcoef(x[g], y[g])[0, 1]
    zeros[n] = np.mean(x == 0)
    mean_new, mean_old = mean_new + np.nan_to_num(x)/nt, mean_old + np.nan_to_num(y)/nt
print('\nEVERY RECORD (one line every day)')
print('   record  time            part   rms difference   rms V2    rms paper   correlation   share of zeros in V2')
for n in list(range(0, nt, STEP)) + [nt - 1]:
    print(f'   {n:5d}  {str(t[n])[:13]}   {part[n] + 1:3d}   {rms_d[n]:12.4f}   {rms_new[n]:8.4f}   {rms_old[n]:8.4f}   '
          f'{cor[n]:10.3f}   {zeros[n]:10.3f}')
print('\nBY PART (median over its records)')
for i in sorted(set(part)):
    m = part == i
    print(f'   part {i + 1}: {int(m.sum())} records, rms difference {np.median(rms_d[m]):.4f}, '
          f'correlation {np.median(cor[m]):.3f}, smallest correlation {cor[m].min():.3f}, '
          f'records with correlation above 0.999: {int(np.sum(cor[m] > 0.999))}')

# --------------------------------------------------------------------------
# the mean flow of the window, which holds the eddies and no tide
# --------------------------------------------------------------------------
gm = np.isfinite(mean_new) & np.isfinite(mean_old)
cor_mean = np.corrcoef(mean_new[gm], mean_old[gm])[0, 1]
rms_mean_d, rms_mean = np.sqrt(np.mean((mean_new[gm] - mean_old[gm])**2)), np.sqrt(np.mean(mean_old[gm]**2))
print(f'\nMEAN FLOW OF THE RECORD (the eddies). Correlation {cor_mean:.3f}, rms difference {rms_mean_d:.4f} '
      f'against an rms of {rms_mean:.4f} m s-1')

# --------------------------------------------------------------------------
# lags: the V2 cache at record n against the cache of the paper at n + lag
# --------------------------------------------------------------------------
recs = np.arange(12, nt - 12, STEP)
lag_cor = np.zeros(LAGS.size)
for j, L in enumerate(LAGS):
    lag_cor[j] = np.mean([np.corrcoef(both(n, 'new'), both(n + int(L), 'old'))[0, 1] for n in recs])
print('\nLAGS (mean correlation over one record a day, V2 at n against the paper at n + lag)')
print('   lag (h)      ' + ' '.join(f'{L:+6d}' for L in LAGS))
print('   correlation  ' + ' '.join(f'{x:6.3f}' for x in lag_cor))
best = int(LAGS[np.argmax(lag_cor)])

# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
print('\nREADING')
if np.nanmax(rms_d) < 1e-6:
    print('   The two caches are the same. Nothing to fix.')
elif best != 0 and lag_cor.max() > 0.999:
    print(f'   A SHIFT IN TIME of {best:+d} h. The V2 records carry another time stamp. The extraction needs that shift.')
elif np.mean(cor > 0.999) > 0.5:
    bad = np.where(cor <= 0.999)[0]
    print(f'   BAD RECORDS. {bad.size} of {nt} records differ, from record {bad[0]} ({str(t[bad[0]])[:13]}) to '
          f'record {bad[-1]} ({str(t[bad[-1]])[:13]}), parts {sorted(set(part[bad] + 1))}. The rest are the same run.')
elif cor[:24].mean() > 0.99 and cor[-24:].mean() < cor[:24].mean() - 0.02:
    print(f'   A RUN THAT DIVERGES. Correlation {cor[:24].mean():.3f} on the first day and {cor[-24:].mean():.3f} on the '
          f'last. V2 started from the same state and drifted from the run of the paper.')
else:
    print(f'   ANOTHER REALISATION. Correlation {np.median(cor):.3f} at the median and {cor_mean:.3f} for the mean flow. '
          f'V2 is not the run the caches of the paper were made from.')