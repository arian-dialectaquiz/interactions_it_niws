"""
fig_07_compute.py  (full_v5, revision 2)
----------------------------------------
Figure 7. Where the energy goes.

The near-inertial kinetic energy of the two wakes by depth and time, its
vertical propagation and its decay, by class of the flow under the wake
(below resonance where the flow detuning is negative, so a wave at omega/2
is free, above it where the detuning is positive, as in Figures 2 to 6). The residual of the budget of Figure 6 holds
most of the decay of both wakes, and this figure follows that energy down the
column. The question is whether the boundary current that decides where the
tide feeds the wake also decides where the wake energy goes, and whether the
tide changes that.

Two passes over the slab caches (velocities only, no density, so kinetic
energy only).

Pass 1, by chunk of levels. The kinetic energy of each wake is accumulated
into depth bins by block (volume-weighted), which gives the depth-time
structure by class, the centroid depth through time, the e-folding time by
depth after the peak and the ratio with over without the tide by depth.

Pass 2, on a subset of full columns (every COL_STRIDE point of the footprint
deeper than Z_SPEC + 50 m). The complex wake velocity u + iv is interpolated
to a uniform depth grid over the upper Z_SPEC m, the depth mean removed, a
Hann window applied, and Fourier transformed in depth. In the southern
hemisphere (f < 0) the inertial current turns anticlockwise in time, and a
wave carrying its energy downward has its phase propagating upward, so its
velocity vector turns anticlockwise with increasing depth (the mirror of
Leaman and Sanford 1975). With the transform written over depth d increasing
downward, w(d) = sum W(kappa) exp(i kappa d), the energy at kappa > 0 is the
downward-propagating part and at kappa < 0 the upward part. The downward
fraction is E(kappa > 0) over E(kappa != 0), amplitude-weighted over the
columns of a class. The shear variance over the same layer gives the ratio
with over without the tide, and its square is the finescale dissipation ratio
of the Gregg et al. (2003) scaling without the frequency correction (no
strain without density).

Inputs
  ../new_full/fig6_cache_<RUN>/     u and v on the 3 km slope tiles, four runs
  ./fig2_out/fig2_wake.nc           geometry, windows, detuning of FULL
  ./fig5_out/fig5_blocks.csv        wake detuning by block (the alternative class, report only)

Outputs (in OUTDIR)
  fig7_fate.nc  fig7_blocks.csv  fig7_fate_report.txt

Run under nohup. Pass 1 is the cost of Figure 6, pass 2 is minutes.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------------------------------
# paths and settings
# --------------------------------------------------------------------------
ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
NEW_FULL = '../new_full/'
CACHES = {'BACK': [NEW_FULL + 'fig6_cache_BACK/'],
          'TIDE': [NEW_FULL + 'fig6_cache_TIDE/'],
          'WIND': [NEW_FULL + 'fig6_cache_WIND/'],
          'FULL': [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/']}
FIG2 = './fig2_out/fig2_wake.nc'
FIG5_BLOCKS = './fig5_out/fig5_blocks.csv'
OUTDIR = './fig7_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
PAIRS = {'with': ('FULL', 'TIDE'), 'without': ('WIND', 'BACK')}
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2, as Figures 2 to 6
CHUNK_Z = 4
SMOOTH_H = 25
Z_EDGES = np.r_[np.arange(0, 501, 25), 600, 800, 1000, 1500, 2000.0]      # m, depth bins of pass 1
Z_SPEC = 400.0                                     # m, the layer of the rotary spectra
DZ_SPEC = 10.0                                     # m, its grid
COL_STRIDE = 3                                     # every third point of the footprint in both directions
SPEC_STEP_H = 6                                    # hours between spectra
DECAY_DAYS = 4.0                                   # the fit of the e-folding time after the peak
N_BOOT = 2000
rng = np.random.default_rng(29)

# --------------------------------------------------------------------------
# grid, slab geometry and the Figure 2 fields
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h = ref.h.values
hc, Vt = float(ref.hc.values), int(ref.Vtransform.values)
s_rho, s_w, Cs_r, Cs_w = ref.s_rho.values, ref.s_w.values, ref.Cs_r.values, ref.Cs_w.values
neta, nxi = h.shape


def z_of(s, C, hh):
    s, C = s[:, None, None], C[:, None, None]
    return hh*((hc*s + hh*C)/(hc + hh)) if Vt == 2 else hc*s + (hh - hc)*C


def find_cache(run):
    for d in CACHES[run]:
        if os.path.exists(d + 'geometry.npz'):
            return d
    raise SystemExit(f'no slab cache for {run}')


def to_hour(x):
    x = np.asarray(x).astype('datetime64[ns]')
    return (x + np.timedelta64(30, 'm')).astype('datetime64[h]').astype('datetime64[ns]')


cache = {r: find_cache(r) for r in RUNS}
geo = np.load(cache['FULL'] + 'geometry.npz', allow_pickle=True)
SUB = int(geo['SUB'])
eta_s, xi_s = geo['eta_s'], geo['xi_s']
t = to_hour(geo['time'])
nt = t.size
dt_h = float((t[1] - t[0])/np.timedelta64(1, 'h'))
nyq = 0.5/dt_h
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
print(f'{ntiles} tiles, {nt} records, sub grid {ny_s} x {nx_s}', flush=True)

f2 = xr.open_dataset(FIG2)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s and f2.sizes['time'] == nt, \
    'fig2_wake.nc is not on the slab geometry, rerun fig_02_compute.py'
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
foot = slope_s & (sector_s == 0)
dc_full = f2.dc_FULL.values*RAD2CPH
CLASS_MAP = {'free': foot & (dc_full < 0), 'trapped': foot & (dc_full > 0)}
CLASSES = ['free', 'trapped']
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
WAKE = (t >= np.datetime64(WINDOWS['wake'][0])) & (t < np.datetime64(WINDOWS['wake'][1]))
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
ids = np.unique(blk[foot])
nblk = ids.size
bidx_map = np.full((ny_s, nx_s), -1)
for i, b in enumerate(ids):
    bidx_map[blk == b] = i
foot_area = float(area_s[foot].sum())
nzb = Z_EDGES.size - 1
zmid = 0.5*(Z_EDGES[1:] + Z_EDGES[:-1])
print(f'footprint {foot_area/1e6:.0f} km2, {nblk} blocks, {nzb} depth bins', flush=True)


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


half = 0.5*OMEGA_M2*RAD2CPH
SOS_NI = band_sos(NI_BAND[0]*half, NI_BAND[1]*half)


def bp(x):
    sos, pad = SOS_NI
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


def load(run, var, k, z0, z1):
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


def smooth(x, n=SMOOTH_H):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


# --------------------------------------------------------------------------
# pass 1. The kinetic energy of each wake by depth bin and block
# --------------------------------------------------------------------------
Eblk = {p: np.zeros((nt, nzb, nblk)) for p in PAIRS}                          # J, by bin and block
vol_blk = np.zeros((nzb, nblk))                                                # m3
area_blk = np.zeros(nblk)
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = np.moveaxis(z_of(s_rho, Cs_r, h_t), 0, -1)                           # (ny, nx, nz), negative up
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    nz = dz.shape[-1]
    ny_t, nx_t = h_t.shape
    ar_t = np.zeros((ny_t, nx_t))
    ar_t[rows, :] = area_s[b0:b1, cols]
    bi_t = np.full((ny_t, nx_t), -1)
    bi_t[rows, :] = np.where(foot[b0:b1, cols], bidx_map[b0:b1, cols], -1)
    m_t = bi_t >= 0
    if not m_t.any():
        print(f'tile {k + 1} of {ntiles}, no footprint', flush=True)
        continue
    for i in np.unique(bi_t[m_t]):
        area_blk[i] += ar_t[bi_t == i].sum()
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        dzc = dz[..., z0:z1]
        zb = np.clip(np.digitize(-zr[..., z0:z1], Z_EDGES) - 1, 0, nzb - 1)   # (ny, nx, nzc)
        key = (zb*nblk + bi_t[..., None])[m_t]                                # (npts, nzc)
        wgt = (dzc*ar_t[..., None])[m_t]                                      # m3
        vol_blk += np.bincount(key.ravel(), weights=wgt.ravel(), minlength=nzb*nblk).reshape(nzb, nblk)
        u = {r: load(r, 'u', k, z0, z1) for r in RUNS}
        v = {r: load(r, 'v', k, z0, z1) for r in RUNS}
        for p, (ra, rb) in PAIRS.items():
            au, av = bp(u[ra] - u[rb]), bp(v[ra] - v[rb])
            ke = 0.5*RHO0*(au*au + av*av)                                     # J m-3 (nt, ny, nx, nzc)
            kem = ke[:, m_t]                                                  # (nt, npts, nzc)
            for it in range(nt):
                Eblk[p][it] += np.bincount(key.ravel(), weights=(kem[it]*wgt).ravel(),
                                           minlength=nzb*nblk).reshape(nzb, nblk)
            del au, av, ke, kem
        del u, v
    print(f'pass 1 tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min', flush=True)

npts_blk = np.array([int(np.sum(foot & (bidx_map == i))) for i in range(nblk)])
ok_blk = (area_blk > 0) & (npts_blk >= BLOCK*BLOCK//4)                       # the same blocks as Figures 3 to 6
# class of each block from the detuning map, as Figure 3
free_share = np.array([np.mean(dc_full[foot & (bidx_map == i)] < 0) if ok_blk[i] else np.nan for i in range(nblk)])
blk_class = np.where(free_share > 0.5, 'free', 'trapped')
CLS_BLK = {c: ok_blk & (blk_class == c) for c in CLASSES}
# energy per unit area of each block and bin (J m-2), and its density (J m-3)
E_area = {p: Eblk[p]/np.maximum(area_blk, 1e-6)[None, None, :] for p in PAIRS}
with np.errstate(divide='ignore', invalid='ignore'):
    E_dens = {p: np.where(vol_blk[None] > 0, Eblk[p]/vol_blk[None], np.nan) for p in PAIRS}
thick_blk = vol_blk/np.maximum(area_blk, 1e-6)[None, :]                       # m, water in the bin per unit area


def class_mean(x, c, w=None):
    """Area-weighted mean over the blocks of a class along the last axis."""
    m = CLS_BLK[c]
    w = area_blk if w is None else w
    return np.nansum(x[..., m]*w[m], axis=-1)/np.sum(w[m])


hov = {(p, c): class_mean(E_area[p], c) for p in PAIRS for c in CLASSES}       # (nt, nzb) J m-2 per bin
thick = {c: class_mean(thick_blk, c) for c in CLASSES}                         # (nzb,) m
dens = {(p, c): hov[(p, c)]/np.maximum(thick[c], 1e-6)[None, :] for p in PAIRS for c in CLASSES}  # J m-3
centroid = {}
below = {}
Z_BELOW = 100.0
for p in PAIRS:
    for c in CLASSES:
        H = np.apply_along_axis(smooth, 0, hov[(p, c)])
        tot = H.sum(axis=1)
        centroid[(p, c)] = (H*zmid[None, :]).sum(axis=1)/np.maximum(tot, 1e-9)
        below[(p, c)] = H[:, zmid > Z_BELOW].sum(axis=1)/np.maximum(tot, 1e-9)

# the decay rate by depth after the peak of the class (d-1, positive for decay, the e-folding time is its inverse),
# by class and pair, with block bootstrap
i_peak = {}
rate = {}
for p in PAIRS:
    for c in CLASSES:
        tot = smooth(hov[(p, c)].sum(axis=1))
        ip = int(np.argmax(np.where(WORK, tot, -np.inf)))
        i_peak[(p, c)] = ip
        sel = (np.arange(nt) >= ip) & (np.arange(nt) < ip + int(DECAY_DAYS*24/dt_h))
        tt = (np.arange(nt)[sel] - ip)*dt_h/24.0
        m = CLS_BLK[c]
        idx = np.where(m)[0]

        def rate_of(ii):
            x = np.sum(E_area[p][:, :, ii]*area_blk[ii][None, None, :], axis=-1)/np.sum(area_blk[ii])
            out = np.full(nzb, np.nan)
            for j in range(nzb):
                y = smooth(x[:, j])[sel]
                if np.all(y > 0):
                    out[j] = -np.polyfit(tt, np.log(y), 1)[0]
            return out

        if idx.size == 0:
            rate[(p, c)] = np.full((nzb, 3), np.nan)
            continue
        est = rate_of(idx)
        bs = np.array([rate_of(idx[rng.integers(0, idx.size, size=idx.size)]) for _ in range(N_BOOT//10)])
        rate[(p, c)] = np.column_stack([est, np.nanpercentile(bs, 5, axis=0), np.nanpercentile(bs, 95, axis=0)])

# the ratio with over without the tide by depth, wake window, by class, with block bootstrap
ratio_z = {}
for c in CLASSES:
    idx = np.where(CLS_BLK[c])[0]
    num = E_area['with'][WAKE].mean(axis=0)[:, idx]*area_blk[idx]
    den = E_area['without'][WAKE].mean(axis=0)[:, idx]*area_blk[idx]
    est = num.sum(axis=1)/np.maximum(den.sum(axis=1), 1e-9)
    ii = rng.integers(0, idx.size, size=(N_BOOT, idx.size))
    bs = num[:, ii].sum(axis=2)/np.maximum(den[:, ii].sum(axis=2), 1e-9)           # (nzb, N_BOOT)
    ratio_z[c] = np.column_stack([est, np.percentile(bs, 5, axis=1), np.percentile(bs, 95, axis=1)])

# --------------------------------------------------------------------------
# pass 2. Rotary vertical wavenumber spectra on a subset of columns
# --------------------------------------------------------------------------
d_grid = np.arange(0.5*DZ_SPEC, Z_SPEC, DZ_SPEC)                               # m, depth
nd = d_grid.size
kappa = 2*np.pi*np.fft.fftfreq(nd, DZ_SPEC)                                     # rad m-1, positive = anticlockwise with depth
hann = np.hanning(nd + 2)[1:-1]
win_norm = np.mean(hann**2)
sel_cols = foot & (h_s > Z_SPEC + 50.0) & (np.arange(ny_s)[:, None] % COL_STRIDE == 0) \
    & (np.arange(nx_s)[None, :] % COL_STRIDE == 0)
i_spec = np.where(WAKE)[0][::int(SPEC_STEP_H/dt_h)]
n_spec = i_spec.size
pos, neg = kappa > 0, kappa < 0
spec = {(p, c): np.zeros(nd) for p in PAIRS for c in CLASSES}                  # mean |W|^2 by kappa, wake window
spec_n = {c: 0 for c in CLASSES}
down_t = {(p, c): np.zeros((n_spec, 2)) for p in PAIRS for c in CLASSES}        # (sum of downward, sum of total) by time
shear = {(p, c): 0.0 for p in PAIRS for c in CLASSES}                            # shear variance, wake window
col_rows = []
n_cols_used = 0
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    sub = sel_cols[b0:b1, cols]
    if not sub.any():
        continue
    iy_s, ix_s = np.where(sub)                                                  # relative to the slab block of the tile
    iy_t, ix_t = iy_s + rows.start, ix_s                                        # tile-local (rows carry the halo)
    h_t = h[np.ix_(jy, jx)]
    zr = np.moveaxis(z_of(s_rho, Cs_r, h_t), 0, -1)                             # (ny, nx, nz)
    ncol = iy_t.size
    series = {}
    for r in RUNS:
        u = np.load(cache[r] + f'u_t{k:03d}.npy', mmap_mode='r')
        v = np.load(cache[r] + f'v_t{k:03d}.npy', mmap_mode='r')
        series[r] = (np.nan_to_num(np.asarray(u[:, iy_t, ix_t, :], dtype=np.float32)),
                     np.nan_to_num(np.asarray(v[:, iy_t, ix_t, :], dtype=np.float32)))   # (nt, ncol, nz)
    for p, (ra, rb) in PAIRS.items():
        wu = bp(series[ra][0] - series[rb][0])[i_spec]
        wv = bp(series[ra][1] - series[rb][1])[i_spec]
        w = (wu + 1j*wv).astype(np.complex64)                                   # (n_spec, ncol, nz)
        for j in range(ncol):
            zc = -zr[iy_t[j], ix_t[j], :]                                       # depth of the levels, decreasing with index
            order = np.argsort(zc)
            prof = np.empty((n_spec, nd), np.complex64)
            for it in range(n_spec):
                prof[it] = np.interp(d_grid, zc[order], w[it, j, order].real) \
                    + 1j*np.interp(d_grid, zc[order], w[it, j, order].imag)
            prof -= prof.mean(axis=1, keepdims=True)
            W = np.fft.fft(prof*hann[None, :], axis=1)/nd
            P = (np.abs(W)**2)/win_norm                                         # (n_spec, nd)
            ib = bidx_map[b0 + iy_s[j], cols.start + ix_s[j]]
            if not ok_blk[ib]:
                continue
            cls = blk_class[ib]
            spec[(p, cls)] += P.sum(axis=0)
            down_t[(p, cls)][:, 0] += P[:, pos].sum(axis=1)
            down_t[(p, cls)][:, 1] += P[:, pos].sum(axis=1) + P[:, neg].sum(axis=1)
            shear[(p, cls)] += float(np.mean(np.abs(np.diff(prof, axis=1)/DZ_SPEC)**2))
            if p == 'with':
                spec_n[cls] += 1
            col_rows.append(dict(pair=p, cls=cls, iy=b0 + iy_s[j], ix=cols.start + ix_s[j],
                                 down=float(P[:, pos].sum()/max(P[:, pos].sum() + P[:, neg].sum(), 1e-30)),
                                 energy=float(P.sum()), shear=float(np.mean(np.abs(np.diff(prof, axis=1)/DZ_SPEC)**2))))
        del w, wu, wv
    n_cols_used += ncol
    del series
    print(f'pass 2 tile {k + 1} of {ntiles}, {ncol} columns, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)
cols_df = pd.DataFrame(col_rows, columns=['pair', 'cls', 'iy', 'ix', 'down', 'energy', 'shear'])
n_cols_total = int(foot.sum())
for key in spec:
    spec[key] /= max(spec_n[key[1]]*n_spec, 1)
down_frac = {key: float(v[:, 0].sum()/max(v[:, 1].sum(), 1e-30)) for key, v in down_t.items()}
down_series = {key: v[:, 0]/np.maximum(v[:, 1], 1e-30) for key, v in down_t.items()}
shear_ratio = {c: shear[('with', c)]/max(shear[('without', c)], 1e-30) for c in CLASSES}
# block bootstrap of the downward fraction over the columns (amplitude-weighted)
down_boot = {}
for key in down_frac:
    d = cols_df[(cols_df.pair == key[0]) & (cols_df.cls == key[1])]
    if len(d) >= 3:
        wgt, dn = d.energy.values, d.down.values
        ii = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        bs = (dn[ii]*wgt[ii]).sum(axis=1)/np.maximum(wgt[ii].sum(axis=1), 1e-30)
        down_boot[key] = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
    else:
        down_boot[key] = (np.nan, np.nan)
# the alternative class (wake frequency above or below the effective frequency), report only
alt = {}
if os.path.exists(FIG5_BLOCKS):
    b5 = pd.read_csv(FIG5_BLOCKS)
    above = dict(zip(b5.block.values, b5.drift_cph.values > b5.dc.values))
    alt_cls = np.array([above.get(int(b), np.nan) for b in ids], dtype=float)
    for name, val in (('above f_eff', 1.0), ('below f_eff', 0.0)):
        m = ok_blk & (alt_cls == val)
        if m.sum() >= 3:
            for p in PAIRS:
                tot = np.sum(E_area[p][:, :, m]*area_blk[m][None, None, :], axis=-1).sum(axis=1)/np.sum(area_blk[m])
                H = np.sum(E_area[p][:, :, m]*area_blk[m][None, None, :], axis=-1)/np.sum(area_blk[m])
                alt[(name, p)] = dict(n=int(m.sum()),
                                      centroid_end=float((H[WAKE].mean(axis=0)*zmid).sum()/H[WAKE].mean(axis=0).sum()),
                                      below=float(H[WAKE].mean(axis=0)[zmid > Z_BELOW].sum()/H[WAKE].mean(axis=0).sum()))

# --------------------------------------------------------------------------
# block table and save
# --------------------------------------------------------------------------
rows_b = []
for i, b in enumerate(ids):
    if not ok_blk[i]:
        continue
    rec = dict(block=int(b), cls=blk_class[i], free_share=float(free_share[i]), area_km2=float(area_blk[i]/1e6),
               n=int(npts_blk[i]))
    for p in PAIRS:
        H = np.apply_along_axis(smooth, 0, E_area[p][:, :, i])
        Hw = H[WAKE].mean(axis=0)
        rec[f'E_{p}_wake'] = float(Hw.sum())
        rec[f'centroid_{p}_wake'] = float((Hw*zmid).sum()/max(Hw.sum(), 1e-9))
        rec[f'below_{p}_wake'] = float(Hw[zmid > Z_BELOW].sum()/max(Hw.sum(), 1e-9))
    rows_b.append(rec)
blocks = pd.DataFrame(rows_b)
blocks.to_csv(OUTDIR + 'fig7_blocks.csv', index=False)
cols_df.to_csv(OUTDIR + 'fig7_columns.csv', index=False)

t_spec = t[i_spec]
ds = xr.Dataset(
    data_vars={**{f'hov_{p}_{c}': (('time', 'zbin'), hov[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'dens_{p}_{c}': (('time', 'zbin'), dens[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'centroid_{p}_{c}': (('time',), centroid[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'below_{p}_{c}': (('time',), below[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'rate_{p}_{c}': (('zbin', 'stat'), rate[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'ratio_{c}': (('zbin', 'stat'), ratio_z[c]) for c in CLASSES},
               **{f'spec_{p}_{c}': (('kappa',), spec[(p, c)]) for p in PAIRS for c in CLASSES},
               **{f'down_{p}_{c}': (('tspec',), down_series[(p, c)]) for p in PAIRS for c in CLASSES},
               'thick': (('cls', 'zbin'), np.array([thick[c] for c in CLASSES])),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'foot': (('y', 'x'), foot), 'spec_cols': (('y', 'x'), sel_cols)},
    coords=dict(time=t, tspec=t_spec, zbin=zmid, kappa=kappa, stat=['est', 'p05', 'p95'], cls=CLASSES),
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF)[:16],
               ni_band=str(NI_BAND), smooth_h=SMOOTH_H, z_edges=str([float(x) for x in Z_EDGES]), z_spec=Z_SPEC, dz_spec=DZ_SPEC,
               z_below=Z_BELOW, decay_days=DECAY_DAYS, foot_area_m2=foot_area, block_points=BLOCK, n_boot=N_BOOT,
               omega_M2=OMEGA_M2, down_frac=str(down_frac), down_boot=str(down_boot), shear_ratio=str(shear_ratio),
               n_cols=str(spec_n), n_cols_total=n_cols_total, i_peak=str({f'{p}_{c}': int(v) for (p, c), v in i_peak.items()}),
               alt=str({f'{a}|{p}': v for (a, p), v in alt.items()}),
               hemisphere='southern, downward energy is anticlockwise rotation with increasing depth (kappa > 0)'))
ds.to_netcdf(OUTDIR + 'fig7_fate.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def at_days(x, d):
    i = int(np.argmin(np.abs(t - (T_OFF + np.timedelta64(int(d*24), 'h')))))
    return float(x[i])


L = ['FIGURE 7 REPORT (full_v5, revision 2)  --  where the energy goes',
     '=' * 80,
     f'slab caches, {nt} records, footprint {foot_area/1e6:.0f} km2, {int(ok_blk.sum())} blocks of {BLOCK} points '
     f'({int(CLS_BLK["free"].sum())} free, {int(CLS_BLK["trapped"].sum())} trapped)',
     f'wake with the tide u_FULL - u_TIDE, without it u_WIND - u_BACK, band {NI_BAND} of omega/2, kinetic energy only',
     f'depth bins {Z_EDGES[0]:.0f} to {Z_EDGES[-1]:.0f} m, rotary spectra on the upper {Z_SPEC:.0f} m at {DZ_SPEC:.0f} m '
     f'of {spec_n["free"]} free and {spec_n["trapped"]} trapped columns (every {COL_STRIDE} points, h > {Z_SPEC + 50:.0f} m) '
     f'every {SPEC_STEP_H} h of the wake window',
     'southern hemisphere: downward energy is anticlockwise rotation of the velocity with increasing depth',
     '',
     f'1. THE CENTROID DEPTH (m) AND THE SHARE BELOW {Z_BELOW:.0f} m AFTER THE WIND STOPS (+1 +3 +5 +7 d)']
for p in PAIRS:
    for c in CLASSES:
        ce, bl = centroid[(p, c)], below[(p, c)]
        L.append(f'   {p:8s} {c:8s} centroid ' + '  '.join(f'{at_days(ce, d):6.0f}' for d in (1, 3, 5, 7))
                 + '   below ' + '  '.join(f'{at_days(bl, d):5.2f}' for d in (1, 3, 5, 7)))
L += ['', f'2. THE DECAY RATE BY DEPTH (d-1, over {DECAY_DAYS:.0f} d after the peak of the class, negative means growth, 5 to 95%, '
      'the e-folding time is the inverse)',
      '   depth m   ' + '   '.join(f'{p} {c:8s}' for p in PAIRS for c in CLASSES)]
for j in range(nzb):
    if zmid[j] > 600:
        continue
    L.append(f'   {Z_EDGES[j]:4.0f}-{Z_EDGES[j + 1]:4.0f}   '
             + '   '.join(f'{rate[(p, c)][j, 0]:6.2f} [{rate[(p, c)][j, 1]:6.2f} {rate[(p, c)][j, 2]:6.2f}]'
                         for p in PAIRS for c in CLASSES))
L += ['', '3. THE RATIO WITH OVER WITHOUT THE TIDE BY DEPTH (wake window, 5 to 95% over the blocks)',
      '   depth m        free                  trapped']
for j in range(nzb):
    if zmid[j] > 600:
        continue
    L.append(f'   {Z_EDGES[j]:4.0f}-{Z_EDGES[j + 1]:4.0f}   '
             + '   '.join(f'{ratio_z[c][j, 0]:5.2f} [{ratio_z[c][j, 1]:5.2f} {ratio_z[c][j, 2]:5.2f}]' for c in CLASSES))
L += ['', '4. THE DOWNWARD FRACTION OF THE WAKE ENERGY (rotary vertical wavenumber, wake window, amplitude-weighted, 5 to 95% over columns)']
for p in PAIRS:
    for c in CLASSES:
        L.append(f'   {p:8s} {c:8s} {down_frac[(p, c)]:.2f} [{down_boot[(p, c)][0]:.2f}, {down_boot[(p, c)][1]:.2f}]')
L += [f'   shear variance with over without the tide: free {shear_ratio["free"]:.2f}, trapped {shear_ratio["trapped"]:.2f}, '
      f'finescale dissipation ratio (its square, Gregg et al. 2003, no frequency correction): free {shear_ratio["free"]**2:.2f}, '
      f'trapped {shear_ratio["trapped"]**2:.2f}',
      '']
if alt:
    L.append('5. THE ALTERNATIVE CLASS (wake frequency above or below the effective frequency, Figure 5, wake window)')
    for (name, p), v in alt.items():
        L.append(f'   {name:12s} {p:8s} {v["n"]:3d} blocks, centroid {v["centroid_end"]:.0f} m, share below {Z_BELOW:.0f} m {v["below"]:.2f}')
    L.append('')
L += ['READING THE RESULT',
      '   question 4 (second half). The boundary current decides where the wake energy goes if the',
      '   centroid deepens and the downward fraction exceeds one half where the waves are free while',
      '   the energy stays in the upper layer and decays in place where they are trapped, in both',
      '   pairs. The tide changes the fate only if the ratio with over without the tide differs',
      '   between the surface and the depth of a class or the downward fraction differs between the',
      '   pairs beyond their intervals. A fate set by the class and the same in both pairs says the',
      '   current decides it and the tide feeds the wake without changing where its energy goes.']
open(OUTDIR + 'fig7_fate_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('saved', OUTDIR + 'fig7_fate.nc')