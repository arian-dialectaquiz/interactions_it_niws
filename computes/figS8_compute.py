"""
figS8_compute.py  (full_v5)  Supplementary Figure S8
----------------------------------------------------
The strain of the balanced flow and the second-order correction it makes to
the effective inertial frequency.

The paper measures the distance of a place from the subharmonic resonance with
the first-order effective frequency |f + zeta/2| (Mooers 1975, Kunze 1985). For
a uniform nondivergent gradient the frequency of an inertial current is exactly

    omega_0^2 = (f + zeta/2)^2 - |sigma|^2 / 4 ,     |sigma|^2 = sigma_n^2 + sigma_s^2 ,

so the strain of the balanced flow lowers the frequency by about
|sigma|^2 / (8 |f|). The derivation notes assumed |sigma| of 0.1 to 0.3 |f|.
This script measures it.

The balanced flow is formed exactly as the vorticity of Figures 2 to 6
(fig_02_compute.py and fig_03_compute.py): the velocity of each run averaged
over the upper 150 m and over a window, gradients by centred differences on
the 3 km slope tiles, and each gradient coarse-grained with the Gaussian of
12.5 km standard deviation normalised by the smoothed mask. The vorticity it
gives for FULL in the wake window is checked against dc_FULL of fig2_wake.nc.

What is formed, for every run and window
  zeta / f, |sigma| / |f| and delta / |f| of the smoothed flow
  the first-order frequency |f + zeta/2| and the exact one omega_0
  the correction omega_0 - |f + zeta/2| and its estimate -|sigma|^2 / (8|f|)
  the detuning of the flow with each frequency

Tests (FULL in the wake window, the flow that defines the classes)
  the distribution of the three gradients over the footprint, area weighted
  the correction by bin of the first-order detuning, median with a 5 to 95%
    block bootstrap
  the share of the footprint area and the number of blocks whose class
    (the sign of the detuning) changes when the exact frequency is used
  the same without the Gaussian, as an upper bound

Inputs
  ../new_full/fig6_cache_<RUN>/     u and v on the 3 km slope tiles, four runs
  ./fig2_out/fig2_wake.nc           geometry, windows, classes (dc_FULL)
  the history file of FULL          grid metrics only

Outputs (in OUTDIR)
  figS8_strain.nc  figS8_blocks.csv  figS8_strain_report.txt

Run (a few minutes, it reads two variables of the slab caches once per window)

  cd ~/dd_waves/paper_3/full_v5
  nohup python -u figS8_compute.py > log_figS8.txt 2>&1 &
  tail -f log_figS8.txt
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

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
OUTDIR = './figS8_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
MAIN_RUN, MAIN_WIN = 'FULL', 'wake'                # the flow and the window that define the classes
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RAD2CPH = 3600.0/(2*np.pi)
Z_UPPER = -150.0                                   # the layer of the balanced flow
L_MESO = 25.0e3                                    # two standard deviations of the Gaussian
DC_EDGES = np.arange(-0.008, 0.0061, 0.001)        # cph, the detuning bins of Figure 2
MIN_BIN = 40                                       # footprint points a detuning bin must hold
PCT = [5, 25, 50, 75, 95]
N_BOOT = 2000
rng = np.random.default_rng(21)

# --------------------------------------------------------------------------
# grid, slab geometry and the Figure 2 fields
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h, pm, pn, f_rho = ref.h.values, ref.pm.values, ref.pn.values, ref.f.values
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
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
f_s = f_rho[np.ix_(np.minimum(eta_s, neta - 1), np.minimum(xi_s, nxi - 1))]

f2 = xr.open_dataset(FIG2)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s and f2.sizes['time'] == nt, \
    'fig2_wake.nc is not on the slab geometry, rerun fig_02_compute.py'
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
mask_s = f2.mask.values if 'mask' in f2 else np.ones_like(h_s)
foot = slope_s & (sector_s == 0)
dc_fig2 = f2.dc_FULL.values*RAD2CPH                 # cph, wake window, the classes of the paper
WINDOWS = eval(f2.attrs['windows'])
W = {k: (t >= np.datetime64(a0)) & (t < np.datetime64(a1)) for k, (a0, a1) in WINDOWS.items()}
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
print(f'{ntiles} tiles, {nt} records, sub grid {ny_s} x {nx_s}, windows {WINDOWS}', flush=True)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


# --------------------------------------------------------------------------
# the pass. Window mean of the velocity of the upper 150 m, then its four
# gradients by centred differences on the 3 km tiles
# --------------------------------------------------------------------------
PARTS = ['zeta', 'sn', 'ss', 'delta']
raw = {(r, w, p): np.full((ny_s, nx_s), np.nan, np.float32) for r in RUNS for w in WINDOWS for p in PARTS}
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = z_of(s_rho, Cs_r, h_t)
    w150 = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0)*(zr > Z_UPPER), 0, -1)
    w150 = (w150/np.maximum(w150.sum(axis=-1, keepdims=True), 1e-6)).astype(np.float32)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    for run in RUNS:
        u = np.load(cache[run] + f'u_t{k:03d}.npy', mmap_mode='r')
        v = np.load(cache[run] + f'v_t{k:03d}.npy', mmap_mode='r')
        for w in WINDOWS:
            kw = np.where(W[w])[0]
            Um = (np.nan_to_num(np.asarray(u[kw[0]:kw[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
            Vm = (np.nan_to_num(np.asarray(v[kw[0]:kw[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
            ux = np.full(Um.shape, np.nan, np.float32)
            uy, vx, vy = ux.copy(), ux.copy(), ux.copy()
            ux[1:-1, 1:-1] = (Um[1:-1, 2:] - Um[1:-1, :-2])/(2*dxi[1:-1, 1:-1])
            vx[1:-1, 1:-1] = (Vm[1:-1, 2:] - Vm[1:-1, :-2])/(2*dxi[1:-1, 1:-1])
            uy[1:-1, 1:-1] = (Um[2:, 1:-1] - Um[:-2, 1:-1])/(2*deta[1:-1, 1:-1])
            vy[1:-1, 1:-1] = (Vm[2:, 1:-1] - Vm[:-2, 1:-1])/(2*deta[1:-1, 1:-1])
            for p, a in (('zeta', vx - uy), ('sn', ux - vy), ('ss', vx + uy), ('delta', ux + vy)):
                raw[(run, w, p)][b0:b1, cols] = a[rows]
    el = (pd.Timestamp.now() - t0).total_seconds()
    left = el/(k + 1)*(ntiles - k - 1)
    print(f'tile {k + 1} of {ntiles}, {el/60:.1f} min, about {left/60:.1f} min left, '
          f'finish near {(pd.Timestamp.now() + pd.Timedelta(seconds=left)).strftime("%H:%M")}', flush=True)

# --------------------------------------------------------------------------
# coarse-graining and the two frequencies
# --------------------------------------------------------------------------
sig_px = 0.5*L_MESO/float(np.nanmean(SUB/pm))


def coarse(a):
    g = np.isfinite(a)
    num = gaussian_filter(np.where(g, a, 0.0), sig_px, mode='nearest')
    den = gaussian_filter(g.astype(float), sig_px, mode='nearest')
    return np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)


def frequencies(zeta, sn, ss, delta):
    """Ratios to |f|, the two frequencies and the two detunings (cph)."""
    af = np.abs(f_s)
    s2 = sn*sn + ss*ss
    feff = np.abs(f_s + 0.5*zeta)                           # first order
    arg = feff*feff - 0.25*s2
    om0 = np.sqrt(np.where(arg > 0, arg, np.nan))           # exact, undefined where the strain exceeds 2 |f_eff|
    return dict(zof=zeta/f_s, sof=np.sqrt(s2)/af, dof=delta/af,
                feff=feff*RAD2CPH, om0=om0*RAD2CPH,
                corr=(om0 - feff)*RAD2CPH, corr_est=-(s2/(8*af))*RAD2CPH,
                dc1=(feff - 0.5*OMEGA_M2)*RAD2CPH, dc2=(om0 - 0.5*OMEGA_M2)*RAD2CPH)


F, Fraw = {}, {}
for r in RUNS:
    for w in WINDOWS:
        F[(r, w)] = frequencies(*[coarse(raw[(r, w, p)]) for p in PARTS])
        Fraw[(r, w)] = frequencies(*[raw[(r, w, p)] for p in PARTS])
M = F[(MAIN_RUN, MAIN_WIN)]

# check against the detuning that defines the classes of the paper
ok = foot & np.isfinite(M['dc1']) & np.isfinite(dc_fig2)
check = dict(n=int(ok.sum()), max_abs=float(np.nanmax(np.abs(M['dc1'][ok] - dc_fig2[ok]))) if ok.any() else np.nan,
             corr=float(np.corrcoef(M['dc1'][ok], dc_fig2[ok])[0, 1]) if ok.sum() > 2 else np.nan)
print(f'check against fig2_wake.nc: {check}', flush=True)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def wpct(x, wgt, q):
    """Area-weighted percentiles."""
    g = np.isfinite(x) & np.isfinite(wgt)
    if g.sum() < 3:
        return np.full(len(q), np.nan)
    x, wgt = x[g], wgt[g]
    o = np.argsort(x)
    c = np.cumsum(wgt[o]) - 0.5*wgt[o]
    return np.interp(np.asarray(q)/100.0, c/np.sum(wgt), x[o])


def stats_of(D, mask):
    out = {}
    for name, x in (('abs zeta over f', np.abs(D['zof'])), ('strain over f', D['sof']),
                    ('abs divergence over f', np.abs(D['dof'])),
                    ('strain over abs zeta', D['sof']/np.maximum(np.abs(D['zof']), 1e-6)),
                    ('correction (1e-3 cph)', D['corr']*1e3),
                    ('correction (% of f)', 100*D['corr']/(np.abs(f_s)*RAD2CPH)),
                    ('estimate -s2/8f (1e-3 cph)', D['corr_est']*1e3)):
        out[name] = wpct(x[mask], area_s[mask], PCT)
    return out


def class_change(D, mask):
    """Share of the area of mask whose sign of the detuning changes with the exact frequency,
    and the share whose first-order detuning is smaller than the correction."""
    g = mask & np.isfinite(D['dc1']) & np.isfinite(D['dc2'])
    a = area_s[g]
    flip = np.sign(D['dc1'][g]) != np.sign(D['dc2'][g])
    near = np.abs(D['dc1'][g]) < np.abs(D['corr'][g])
    below1 = D['dc1'][g] < 0
    below2 = D['dc2'][g] < 0
    return dict(points=int(g.sum()), share_flip=float(np.sum(a[flip])/np.sum(a)),
                share_near=float(np.sum(a[near])/np.sum(a)),
                share_below_first=float(np.sum(a[below1])/np.sum(a)),
                share_below_exact=float(np.sum(a[below2])/np.sum(a)),
                undefined=int((mask & np.isfinite(D['dc1']) & ~np.isfinite(D['dc2'])).sum()))


st_foot, st_slope = stats_of(M, foot), stats_of(M, slope_s)
st_foot_raw = stats_of(Fraw[(MAIN_RUN, MAIN_WIN)], foot)
cc_foot, cc_slope = class_change(M, foot), class_change(M, slope_s)
cc_foot_raw = class_change(Fraw[(MAIN_RUN, MAIN_WIN)], foot)
st_all = {(r, w): stats_of(F[(r, w)], foot) for r in RUNS for w in WINDOWS}

# blocks of 24 km, the unit of the statistics of the paper
rows_b = []
for b in np.unique(blk[foot]):
    m = foot & (blk == b) & np.isfinite(M['dc1']) & np.isfinite(M['dc2'])
    if m.sum() < BLOCK*BLOCK//4:
        continue
    a = area_s[m]
    wm = lambda x: float(np.sum(x[m]*a)/np.sum(a))        # noqa: E731
    rows_b.append(dict(block=int(b), n=int(m.sum()), lat=wm(lat_s), lon=wm(lon_s),
                       zeta_over_f=wm(M['zof']), strain_over_f=wm(M['sof']), div_over_f=wm(M['dof']),
                       dc_first_cph=float(np.median(M['dc1'][m])), dc_exact_cph=float(np.median(M['dc2'][m])),
                       correction_cph=float(np.median(M['corr'][m]))))
blocks = pd.DataFrame(rows_b)
blocks['class_first'] = np.where(blocks.dc_first_cph < 0, 'below', 'above')
blocks['class_exact'] = np.where(blocks.dc_exact_cph < 0, 'below', 'above')
blocks.to_csv(OUTDIR + 'figS8_blocks.csv', index=False)
n_flip_blocks = int((blocks.class_first != blocks.class_exact).sum())

# the correction by bin of the first-order detuning, median with a block bootstrap
centres = 0.5*(DC_EDGES[1:] + DC_EDGES[:-1])
curve = np.full((centres.size, 3), np.nan)                 # median, p05, p95 of the bootstrap
curve_est = np.full(centres.size, np.nan)
curve_sof = np.full((centres.size, 3), np.nan)
npts = np.zeros(centres.size, int)
g_all = foot & np.isfinite(M['dc1']) & np.isfinite(M['corr'])
for i in range(centres.size):
    m = g_all & (M['dc1'] >= DC_EDGES[i]) & (M['dc1'] < DC_EDGES[i + 1])
    npts[i] = int(m.sum())
    if npts[i] < MIN_BIN:
        continue
    ids = np.unique(blk[m])
    per = {b: (M['corr'][m & (blk == b)], M['sof'][m & (blk == b)]) for b in ids}
    curve[i, 0] = np.median(M['corr'][m])
    curve_est[i] = np.median(M['corr_est'][m])
    curve_sof[i, 0] = np.median(M['sof'][m])
    if ids.size >= 3:
        bc, bs = np.empty(N_BOOT), np.empty(N_BOOT)
        for n in range(N_BOOT):
            pick = rng.choice(ids, ids.size)
            bc[n] = np.median(np.concatenate([per[b][0] for b in pick]))
            bs[n] = np.median(np.concatenate([per[b][1] for b in pick]))
        curve[i, 1:] = np.percentile(bc, [5, 95])
        curve_sof[i, 1:] = np.percentile(bs, [5, 95])
    print(f'detuning bin {i + 1} of {centres.size}, {npts[i]} points, {ids.size} blocks', flush=True)

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
QOUT = ['zof', 'sof', 'dof', 'corr', 'corr_est', 'dc1', 'dc2']
ds = xr.Dataset(
    data_vars={**{f'{q}_{r}_{w}': (('y', 'x'), F[(r, w)][q].astype(np.float32))
                  for q in QOUT for r in RUNS for w in WINDOWS},
               **{f'{q}_raw': (('y', 'x'), Fraw[(MAIN_RUN, MAIN_WIN)][q].astype(np.float32)) for q in QOUT},
               'curve_corr': (('dc_bin', 'stat'), curve), 'curve_corr_est': (('dc_bin',), curve_est),
               'curve_sof': (('dc_bin', 'stat'), curve_sof), 'npts': (('dc_bin',), npts),
               'dc_fig2': (('y', 'x'), dc_fig2),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'slope': (('y', 'x'), slope_s), 'foot': (('y', 'x'), foot), 'area': (('y', 'x'), area_s),
               'mask': (('y', 'x'), mask_s)},
    coords=dict(dc_bin=centres, stat=['median', 'p05', 'p95']),
    attrs=dict(main_run=MAIN_RUN, main_window=MAIN_WIN, windows=str(WINDOWS), z_upper=Z_UPPER, l_meso=L_MESO,
               block_points=BLOCK, n_boot=N_BOOT, omega_M2=OMEGA_M2,
               share_flip_foot=cc_foot['share_flip'], share_near_foot=cc_foot['share_near'],
               share_below_first=cc_foot['share_below_first'], share_below_exact=cc_foot['share_below_exact'],
               n_blocks=int(len(blocks)), n_flip_blocks=n_flip_blocks,
               median_sof_foot=float(st_foot['strain over f'][2]),
               median_zof_foot=float(st_foot['abs zeta over f'][2]),
               median_corr_foot=float(st_foot['correction (1e-3 cph)'][2]),
               p05_corr_foot=float(st_foot['correction (1e-3 cph)'][0]),
               check_max_abs=check['max_abs'], check_corr=check['corr']))
ds.to_netcdf(OUTDIR + 'figS8_strain.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['Supplementary Figure S8. The strain of the balanced flow and the correction to the effective frequency.',
     f'Balanced flow: velocity of the upper {abs(Z_UPPER):.0f} m, window mean, centred differences on the '
     f'{SUB} km tiles, Gaussian of {0.5*L_MESO/1e3:.1f} km standard deviation ({sig_px:.2f} points).',
     f'Windows {WINDOWS}', '',
     f'CHECK. First-order detuning of {MAIN_RUN} in the {MAIN_WIN} window against dc_FULL of fig2_wake.nc over '
     f'{check["n"]} footprint points: largest difference {check["max_abs"]*1e3:.4f} x 1e-3 cph, '
     f'correlation {check["corr"]:.4f}',
     '   (the two should agree to rounding, the classes of the paper come from that field)', '',
     f'1. THE GRADIENTS OF THE BALANCED FLOW ({MAIN_RUN}, {MAIN_WIN} window), area-weighted percentiles '
     + ' '.join(f'p{p:02d}' for p in PCT)]
for title, st in (('footprint, coarse-grained', st_foot), ('whole slope band, coarse-grained', st_slope),
                  ('footprint, without the Gaussian (upper bound)', st_foot_raw)):
    L.append(f'   {title}')
    for name, v in st.items():
        L.append(f'      {name:30s} ' + ' '.join(f'{x:8.3f}' for x in v))
L += ['',
      '2. THE CLASSES. Sign of the detuning with the first-order and with the exact frequency']
for title, cc in (('footprint, coarse-grained', cc_foot), ('whole slope band, coarse-grained', cc_slope),
                  ('footprint, without the Gaussian', cc_foot_raw)):
    L.append(f'   {title}: {cc["points"]} points, area share that changes class {100*cc["share_flip"]:.1f}%, '
             f'share with |detuning| under |correction| {100*cc["share_near"]:.1f}%, '
             f'share below resonance {100*cc["share_below_first"]:.1f}% first order and '
             f'{100*cc["share_below_exact"]:.1f}% exact, exact frequency undefined at {cc["undefined"]} points')
L += [f'   blocks of {BLOCK} x {BLOCK} points: {len(blocks)} on the footprint, {n_flip_blocks} change class '
      f'({100*n_flip_blocks/max(len(blocks), 1):.1f}%)', '',
      '3. THE CORRECTION BY BIN OF THE FIRST-ORDER DETUNING (footprint)',
      '   bin centre (1e-3 cph)   points   correction (1e-3 cph) median [p05 p95]   estimate   strain over f']
for i in range(centres.size):
    if np.isfinite(curve[i, 0]):
        L.append(f'   {centres[i]*1e3:+8.2f}   {npts[i]:7d}   {curve[i, 0]*1e3:+8.4f} [{curve[i, 1]*1e3:+8.4f} '
                 f'{curve[i, 2]*1e3:+8.4f}]   {curve_est[i]*1e3:+8.4f}   {curve_sof[i, 0]:6.3f}')
L += ['', '4. EVERY RUN AND WINDOW, footprint medians (abs zeta over f, strain over f, correction in 1e-3 cph)']
for (r, w), st in st_all.items():
    L.append(f'   {r:5s} {w:6s} {st["abs zeta over f"][2]:6.3f} {st["strain over f"][2]:6.3f} '
             f'{st["correction (1e-3 cph)"][2]:+8.4f}')
L += ['', 'READING. The derivation notes assumed a strain of 0.1 to 0.3 |f| and a correction of 0.05 to 0.45 x 1e-3 cph.',
      'Section 1 gives the simulated range. Section 2 says how much of the footprint and how many blocks change',
      'class. If the share is a few percent the first-order frequency is enough for the classes, which the paper',
      'already repeats in bins of the continuous detuning.']
open(OUTDIR + 'figS8_strain_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('saved', OUTDIR + 'figS8_strain.nc')