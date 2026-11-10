"""
check_vorticity.py
------------------
Does the tide change the balanced flow, or only the elevation the balanced flow
was inferred from?

The Figure 1 report shows the four experiments falling into two pairs. The
geostrophic vorticity of FULL and TIDE agrees over most of the slope, and so
does that of WIND and BACK, while any pair with the tide in one run and not in
the other has a pattern correlation near 0.6 before the storm and near zero
during it. Two explanations fit. Either the tide changes the slope current
itself, through the mean flow a tide over sloping topography drives and
through the mixing it adds, or the 72 h low pass of the elevation keeps a tidal
residual (a mean set-up, the fortnightly modulation) whose Laplacian reads as
vorticity without any current behind it. The regression slope of the
geostrophic estimate on the model vorticity, 1.31 in TIDE against 0.66 in FULL,
points to the second, and the test is to take the vorticity from the velocity.

This reads the slab caches of fig_06_extract.py, which hold u and v on the
3 km slope tiles from 18 March to 9 April for each experiment already extracted,
and forms the vorticity of the upper 150 m directly from the velocity, low
passed at 72 h, every 12 h, coarse-grained with the Gaussian of Figure 1. The
velocity carries no set-up, so if the split by tide survives here the tide
changes the current, and if it disappears the elevation was the cause.

ROMS velocities are aligned with the grid, so the vorticity is formed in grid
coordinates as dv/dxi pm - du/deta pn, which needs no rotation.

  python -u check_vorticity.py

Outputs (in OUTDIR)
  vort_check.nc          epoch means of zeta/f from velocity, per experiment
  vort_check_report.txt
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
NEW_FULL = '../new_full/'
CACHES = {'BACK': [NEW_FULL + 'fig6_cache_BACK/'],
          'TIDE': [NEW_FULL + 'fig6_cache_TIDE/'],
          'WIND': [NEW_FULL + 'fig6_cache_WIND/'],
          'FULL': [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/']}
FIG1 = './fig1_out/fig1_v5.nc'
OUTDIR = './fig1_out/'

EPOCHS = {'T1': ('2004-03-18', '2004-03-24'),     # the caches start on 18 March
          'T2': ('2004-03-24', '2004-03-31'),
          'T3': ('2004-03-31', '2004-04-07')}
Z_UPPER = -150.0
LP_HOURS = 72.0
STEP_H = 12
L_MESO = 25.0e3
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
PAIRS = [('FULL', 'TIDE'), ('WIND', 'BACK'), ('FULL', 'WIND'), ('TIDE', 'BACK'),
         ('FULL', 'BACK'), ('TIDE', 'WIND')]

ref = xr.open_dataset(HIS)
pm, pn, hh, ff = ref.pm.values, ref.pn.values, ref.h.values, ref.f.values
hc, Vt = float(ref.hc.values), int(ref.Vtransform.values)
s_rho, s_w = ref.s_rho.values, ref.s_w.values
Cs_r, Cs_w = ref.Cs_r.values, ref.Cs_w.values


def z_of(s, C, h):
    s, C = s[:, None, None], C[:, None, None]
    return h*((hc*s + h*C)/(hc + h)) if Vt == 2 else hc*s + (h - hc)*C


def find_cache(run):
    for d in CACHES[run]:
        if os.path.exists(d + 'geometry.npz'):
            return d
    return None


fig1 = xr.open_dataset(FIG1)
slope1 = fig1.slope.values.astype(bool)
ny1, nx1 = slope1.shape
sector1 = fig1.sector.values
foot1 = slope1 & (sector1 == 0)

zof_vel, geo0 = {}, None
for run in CACHES:
    cache = find_cache(run)
    if cache is None:
        print(f'no slab cache for {run}, run fig_06_extract.py {run} in ../new_full')
        continue
    geo = np.load(cache + 'geometry.npz', allow_pickle=True)
    t = geo['time'].astype('datetime64[ns]')
    eta_s, xi_s = geo['eta_s'], geo['xi_s']
    SUB = int(geo['SUB'])
    if geo0 is None:
        geo0 = dict(eta_s=eta_s, xi_s=xi_s, t=t)
    dt_h = float(np.mean(np.diff(t)/np.timedelta64(1, 'h')))
    sos = butter(4, (1.0/LP_HOURS)/(0.5/dt_h), btype='low', output='sos')
    kout = np.arange(0, t.size, int(round(STEP_H/dt_h)))
    tb = t[kout]
    edge = (tb < t[0] + np.timedelta64(int(LP_HOURS/2), 'h')) | \
           (tb > t[-1] - np.timedelta64(int(LP_HOURS/2), 'h'))
    Z = np.full((tb.size, eta_s.size, xi_s.size), np.nan, np.float32)
    print(f'{run}: {int(geo["ntiles"])} tiles from {cache}, {t.size} records', flush=True)
    for k in range(int(geo['ntiles'])):
        ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
        b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
        u = np.load(cache + f'u_t{k:03d}.npy', mmap_mode='r')
        v = np.load(cache + f'v_t{k:03d}.npy', mmap_mode='r')
        jy = np.minimum(ys, hh.shape[0] - 1)
        jx = np.minimum(xs, hh.shape[1] - 1)
        h_t = hh[np.ix_(jy, jx)]
        zr = z_of(s_rho, Cs_r, h_t)
        dz = np.diff(z_of(s_w, Cs_w, h_t), axis=0)
        w = dz*(zr > Z_UPPER)
        w = w/np.maximum(w.sum(axis=0, keepdims=True), 1e-6)
        w = np.moveaxis(w, 0, -1).astype(np.float32)               # (y, x, s)
        U = np.empty(u.shape[:3], np.float32)
        V = np.empty(v.shape[:3], np.float32)
        for n0 in range(0, u.shape[0], 48):                          # the time chunks
            U[n0:n0+48] = (np.nan_to_num(u[n0:n0+48])*w).sum(axis=-1)
            V[n0:n0+48] = (np.nan_to_num(v[n0:n0+48])*w).sum(axis=-1)
        U = sosfiltfilt(sos, U, axis=0)[kout]
        V = sosfiltfilt(sos, V, axis=0)[kout]
        dxi = SUB/pm[np.ix_(jy, jx)]
        deta = SUB/pn[np.ix_(jy, jx)]
        zt = np.full(U.shape, np.nan, np.float32)
        zt[:, 1:-1, 1:-1] = ((V[:, 1:-1, 2:] - V[:, 1:-1, :-2])/(2*dxi[None, 1:-1, 1:-1])
                             - (U[:, 2:, 1:-1] - U[:, :-2, 1:-1])/(2*deta[None, 1:-1, 1:-1]))
        rows = slice(hlo, hlo + (b1 - b0))                           # the halo rows stay out
        Z[:, b0:b1, c0:c0 + xs.size] = np.where(np.isnan(Z[:, b0:b1, c0:c0 + xs.size]),
                                                 zt[:, rows], Z[:, b0:b1, c0:c0 + xs.size])
    # coarse-grain as in Figure 1 and divide by f
    f_s = ff[np.ix_(np.minimum(eta_s, ff.shape[0] - 1), np.minimum(xi_s, ff.shape[1] - 1))]
    dx_m = float(np.nanmean(SUB/pm))
    sig = 0.5*L_MESO/dx_m
    out = np.full_like(Z, np.nan)
    for n in range(tb.size):
        if edge[n]:
            continue
        g = np.isfinite(Z[n])
        num = gaussian_filter(np.where(g, Z[n], 0.0), sig, mode='nearest')
        den = gaussian_filter(g.astype(float), sig, mode='nearest')
        out[n] = np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)/f_s
    ep = []
    for e, (a0, a1) in EPOCHS.items():
        sel = (tb >= np.datetime64(a0)) & (tb < np.datetime64(a1)) & ~edge
        ep.append(np.nanmean(out[sel], axis=0))
    zof_vel[run] = np.stack(ep)[:, :ny1, :nx1]       # the slab grid sits one cell west of the blocks
    print(f'{run}: done', flush=True)

if not zof_vel:
    raise SystemExit('no slab cache found, nothing to compare')


def detuning(zf, f):
    return np.abs(f*(1.0 + 0.5*zf)) - 0.5*OMEGA_M2


f1 = fig1.lat.values
f1 = 2*7.2921159e-5*np.sin(np.radians(f1))
eps = list(EPOCHS)
runs = list(zof_vel)
L = ['VORTICITY CHECK  --  velocity against elevation, and the agreement between experiments',
     '=' * 84,
     f'upper {abs(Z_UPPER):.0f} m mean of the velocity, low pass {LP_HOURS:.0f} h, every {STEP_H} h, '
     f'Gaussian sigma {0.5*L_MESO/1e3:.1f} km, epochs {EPOCHS}', '']

L.append('1. VELOCITY AGAINST ELEVATION, SAME EXPERIMENT (slope band)')
L.append(f'{"run":5s} {"epoch":5s} {"n":>6s} {"spearman":>9s} {"slope":>7s} {"rms vel":>8s} '
         f'{"rms ssh":>8s} {"same regime":>12s}')
rows = []
for run in runs:
    for j, e in enumerate(eps):
        zv = zof_vel[run][j]
        zg = fig1.zof.sel(run=run, epoch=e).values
        m = slope1 & np.isfinite(zv) & np.isfinite(zg)
        if m.sum() < 100:
            continue
        sr = spearmanr(zv[m], zg[m])[0]
        sl = np.polyfit(zv[m], zg[m], 1)[0]
        same = (np.sign(detuning(zv, f1)) == np.sign(detuning(zg, f1)))[m].mean()
        rows.append(dict(run=run, epoch=e, spearman=sr, slope=sl))
        L.append(f'{run:5s} {e:5s} {int(m.sum()):6d} {sr:9.3f} {sl:7.3f} '
                 f'{np.sqrt(np.mean(zv[m]**2)):8.3f} {np.sqrt(np.mean(zg[m]**2)):8.3f} {same:12.3f}')
L.append('   slope is the elevation estimate regressed on the velocity estimate. A slope far above')
L.append('   one in the runs with a tide and near one without it means the elevation carries a')
L.append('   tidal residual that is not a current.')
L.append('')

L.append('2. AGREEMENT BETWEEN EXPERIMENTS, FROM THE VELOCITY (slope band)')
L.append(f'{"pair":11s} ' + ' '.join(f'{"r " + e:>7s} {"same " + e:>8s} {"foot " + e:>8s}'
                                     for e in eps))
for a, b in PAIRS:
    if a not in zof_vel or b not in zof_vel:
        continue
    row = f'{a + "-" + b:11s} '
    for j, e in enumerate(eps):
        za, zb = zof_vel[a][j], zof_vel[b][j]
        m = slope1 & np.isfinite(za) & np.isfinite(zb)
        same = np.sign(detuning(za, f1)) == np.sign(detuning(zb, f1))
        row += (f'{np.corrcoef(za[m], zb[m])[0, 1]:7.3f} {same[m].mean():8.3f} '
                f'{same[m & foot1].mean():8.3f} ')
    L.append(row)
L.append('   the same table from the elevation is in fig1_report.txt. If FULL-WIND and TIDE-BACK')
L.append('   recover the level of FULL-TIDE and WIND-BACK here, the split was the elevation and')
L.append('   the experiments share one balanced flow. If the split survives, the tide changes the')
L.append('   slope current and every difference between experiments carries that change.')
L.append('')
L.append('3. WHAT FOLLOWS')
L.append('   whichever field passes, the detuning of every later figure is taken from it, on the')
L.append('   grid of the slab caches, so Figures 2 to 6 use one balanced flow for each experiment.')

ds = xr.Dataset({'zof_vel': (('run', 'epoch', 'y', 'x'), np.stack([zof_vel[r] for r in runs]))},
                coords=dict(run=runs, epoch=eps, y=np.arange(ny1), x=np.arange(nx1)),
                attrs=dict(epochs=str(EPOCHS), lp_hours=LP_HOURS, z_upper=Z_UPPER,
                           l_meso=L_MESO, note='grid of the slab caches, one cell west of fig1_v5'))
ds.to_netcdf(OUTDIR + 'vort_check.nc')
with open(OUTDIR + 'vort_check_report.txt', 'w') as fh:
    fh.write('\n'.join(L))
print('\n'.join(L))