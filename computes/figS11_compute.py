"""
figS11_compute.py  (full_v5)
----------------------------
Do the phase, the lock and the deficit of the tide belong to the internal tide?

Figure 4 forms the phase of the wake against the tide and the half-width of
the lock with the strain of the semidiurnal band of TIDE at every level
(fig_04_compute.py, bu, bv = bp(u['TIDE'], 'M2'), bp(v['TIDE'], 'M2')), and
Figure 3d forms the deficit of the semidiurnal band and its residence time
with the same velocity (fig_06_compute.py, tu, tv = bp(u['TIDE'], 'M2'), ...).
That velocity holds the barotropic tide (the depth mean) and the internal tide
(the deviation from the depth mean), as in figS10_compute.py,

    u_T = u_bt + u_it,   u_bt = (1/H) sum_z u_T dz,   u_it = u_T - u_bt.

Every quantity below is linear in u_T, so each one splits exactly.

  the rotary strain        S_r = S_r,bt + S_r,it          (demodulated at omega, 72 h cut-off)
  the interaction          Z = sum_z S_r A*^2 dz = S_r,bt sum_z A*^2 dz + sum_z S_r,it A*^2 dz
  the phase                Phi = pi + arg Z, formed with Z, with Z_bt and with Z_it
  the cross term           X = rho0 sum_z u_T . u_sm dz = X_bt + X_it   (u_sm the semidiurnal band of FULL - TIDE)
  the kinetic energy       sum_z u_T^2 dz = H u_bt^2 + sum_z u_it^2 dz  (the cross term has zero depth sum)

The half-width of the lock |S_r|/2 is not linear (it is an amplitude), so it
is formed for each part and for the whole tide from its own rotary strain.

What is written, for the whole tide, the barotropic tide and the internal tide
  1. the phase of the footprint in the wake window, the share of the area with
     a gaining phase and the phase efficiency by class of the flow
  2. the agreement of the phase of each part with the phase of the whole tide
     (blocks of the wake window, and blocks and hours of the wake window)
  3. the lock (resultant length of exp(i Phi) over the blocks) and the
     coherence of the phase of each block over the wake window
  4. the rotary strain amplitude and the half-width of the lock |S_r|/2
  5. the strain work rebuilt from Z and the efficiency (the work over the
     largest work the columns could do at any phase)
  6. the semidiurnal kinetic energy, the deficit of the semidiurnal band and
     the residence time (deficit over the rate of work, as fig_06_compute.py)
  7. checks against fig4_blocks.csv, fig6_budget.nc and figS10_tide_parts.nc

Inputs
  ../new_full/fig6_cache_<RUN>/        u and v of FULL and TIDE on the slope tiles
  ./fig2_out/fig2_wake.nc              geometry, classes and windows
  ./figS10_out/figS10_tide_parts.nc    the rate of work of each part (for the residence time)
  ./fig4_out/fig4_blocks.csv  ./fig3_out/fig3_blocks.csv  ./fig6_out/fig6_budget.nc   for the checks (used if there)

Outputs (in ./figS11_out/)
  figS11_phase_parts.nc   figS11_blocks.csv   figS11_phase_parts_report.txt

Nothing is extracted from the model. About the time of fig_04_compute.py.

  cd ~/dd_waves/paper_3/full_v5
  nohup python -u figS11_compute.py > log_figS11_compute.txt 2>&1 &
  tail -f log_figS11_compute.txt
  cat figS11_out/figS11_phase_parts_report.txt
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------------------------------
# paths and settings (those of fig_04_compute.py and fig_06_compute.py)
# --------------------------------------------------------------------------
ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
NEW_FULL = '../new_full/'
CACHES = {'TIDE': [NEW_FULL + 'fig6_cache_TIDE/'],
          'FULL': [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/']}
FIG2 = './fig2_out/fig2_wake.nc'
FIG3_BLOCKS = './fig3_out/fig3_blocks.csv'
FIG4_BLOCKS = './fig4_out/fig4_blocks.csv'
FIG6 = './fig6_out/fig6_budget.nc'
FIGS10 = './figS10_out/figS10_tide_parts.nc'
OUTDIR = './figS11_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['TIDE', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2
M2_BAND = (0.95, 1.05)                             # times omega_M2
LP_H = 72.0                                        # demodulation cut-off (h), as fig_04_compute.py
CHUNK_Z = 4
SMOOTH_H = 25                                      # as fig_06_compute.py
AMP_FLOOR = 0.10                                   # of the median |Z| for a phase to count, as fig_04_compute.py

PARTS = ['bt', 'bc']                               # barotropic tide and internal tide (the keys of figS10_compute.py)
SRC = ['all', 'bt', 'bc']
NAME = {'all': 'whole tide', 'bt': 'barotropic tide', 'bc': 'internal tide'}

# --------------------------------------------------------------------------
# grid, slab geometry and the Figure 2 fields
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h, pm, pn = ref.h.values, ref.pm.values, ref.pn.values
hc, Vt = float(ref.hc.values), int(ref.Vtransform.values)
s_w, Cs_w = ref.s_w.values, ref.Cs_w.values
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
dt_s = dt_h*3600.0
nyq = 0.5/dt_h
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
tsec = ((t - t[0])/np.timedelta64(1, 's')).astype(np.float64)
print(f'{ntiles} tiles, {nt} records, slab {ny_s} x {nx_s}', flush=True)

f2 = xr.open_dataset(FIG2)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s and f2.sizes['time'] == nt, \
    'fig2_wake.nc is not on the slab geometry'
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
foot = slope_s & (sector_s == 0)
dc_full = f2.dc_FULL.values*RAD2CPH                   # cph, wake window
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
WAKE = (t >= T_OFF) & (t < np.datetime64(T_WAKE1))                                 # the wake window of fig_04_compute.py
WAKE6 = (t >= np.datetime64(WINDOWS['wake'][0])) & (t < np.datetime64(WINDOWS['wake'][1]))   # that of fig_06_compute.py
PRE = (t >= np.datetime64(WINDOWS['pre'][0])) & (t < np.datetime64(WINDOWS['pre'][1]))
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
foot_area = float(area_s[foot].sum())

# blocks of the footprint, the same ids as fig_04_compute.py
ids = np.unique(blk[foot])
counts = {b: int(np.sum(foot & (blk == b))) for b in ids}
ids = np.array([b for b in ids if counts[b] >= BLOCK*BLOCK//4])
blk_index = {b: i for i, b in enumerate(ids)}
nblk = ids.size
print(f'footprint {foot_area/1e6:.0f} km2, {nblk} blocks, work window {T_STORM0} to {T_WAKE1}, '
      f'wind subsides {T_OFF}', flush=True)


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


half = 0.5*OMEGA_M2*RAD2CPH
SOS = {'ni': band_sos(NI_BAND[0]*half, NI_BAND[1]*half),
       'M2': band_sos(M2_BAND[0]*2*half, M2_BAND[1]*2*half)}
SOS_LP = (butter(4, (1.0/LP_H)/nyq, btype='low', output='sos'), int(min(3*LP_H/dt_h, nt - 1)))


def bp(x, name):
    sos, pad = SOS[name]
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def lp_complex(x):
    sos, pad = SOS_LP
    return (sosfiltfilt(sos, x.real, axis=0, padlen=pad)
            + 1j*sosfiltfilt(sos, x.imag, axis=0, padlen=pad)).astype(np.complex64)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


def d_xi(a, dxi):
    """Centred difference along xi of a field (time, y, x) or (time, y, x, z)."""
    out = np.full_like(a, np.nan)
    d = dxi[None, :, 1:-1] if a.ndim == 3 else dxi[None, :, 1:-1, None]
    out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2])/(2*d)
    return out


def d_eta(a, deta):
    out = np.full_like(a, np.nan)
    d = deta[None, 1:-1, :] if a.ndim == 3 else deta[None, 1:-1, :, None]
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2])/(2*d)
    return out


def rotary_strain(bu, bv, dxi, deta):
    """sigma = (u_x - v_y) + i (v_x + u_y) of a horizontal field."""
    return (d_xi(bu, dxi) - d_eta(bv, deta)) + 1j*(d_xi(bv, dxi) + d_eta(bu, deta))


def load(run, var, k, z0, z1):
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


def clock(t0, done, total):
    """Elapsed minutes, minutes left and the hour of the end."""
    el = (pd.Timestamp.now() - t0).total_seconds()/60.0
    left = el*(total - done)/max(done, 1)
    return (f'{el:.1f} min elapsed, about {left:.0f} min left, ends near '
            f'{(pd.Timestamp.now() + pd.Timedelta(minutes=left)).strftime("%H:%M")}')


# the demodulation carriers
car = {'half': np.exp(-0.5j*OMEGA_M2*tsec).astype(np.complex64),      # the wake, at omega/2
       'full': np.exp(-1.0j*OMEGA_M2*tsec).astype(np.complex64),      # the anticlockwise strain, at omega
       'back': np.exp(+1.0j*OMEGA_M2*tsec).astype(np.complex64)}      # the clockwise strain

# --------------------------------------------------------------------------
# the pass over the tiles
# --------------------------------------------------------------------------
Z = {p: np.zeros((nt, ny_s, nx_s), np.complex64) for p in PARTS}                   # sum_z S_r A*^2 dz of each part
S2 = {(p, s): np.zeros((ny_s, nx_s), np.float64) for p in SRC for s in 'rl'}       # work-window mean of sum |S|^2 dz
Hcol = np.zeros((ny_s, nx_s), np.float64)
valid = np.zeros((ny_s, nx_s), bool)
QE = ['M2_bt', 'M2_bc', 'M2_all', 'X_bt', 'X_bc', 'X_all']                         # J m-2 by record and point
series = {q: np.zeros(nt) for q in QE}                                             # footprint sum, J

nsteps = sum(int(np.ceil(np.load(cache['FULL'] + f'u_t{k:03d}.npy', mmap_mode='r').shape[-1]/CHUNK_Z))
             for k in range(ntiles))*2
step = 0
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    H = dz.sum(axis=-1)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    nz = dz.shape[-1]
    ny_t, nx_t = h_t.shape

    # ---------------------------------------------------------------- pass 1. The barotropic tide and its rotary strain
    Ub, Vb = np.zeros((nt, ny_t, nx_t), np.float64), np.zeros((nt, ny_t, nx_t), np.float64)
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        Ub += (load('TIDE', 'u', k, z0, z1)*dz[None, ..., z0:z1]).sum(axis=-1)
        Vb += (load('TIDE', 'v', k, z0, z1)*dz[None, ..., z0:z1]).sum(axis=-1)
        step += 1
    Ub, Vb = bp((Ub/H[None]).astype(np.float32), 'M2'), bp((Vb/H[None]).astype(np.float32), 'M2')
    sig = rotary_strain(Ub, Vb, dxi, deta)                                           # (time, y, x)
    ok_t = np.isfinite(sig[0])
    sig = np.nan_to_num(sig)
    Sr_bt = lp_complex(sig*car['full'][:, None, None])
    Sl_bt = lp_complex(sig*car['back'][:, None, None])
    del sig
    print(f'tile {k + 1} of {ntiles}, barotropic tide done, {clock(t0, step, nsteps)}', flush=True)

    # ---------------------------------------------------------------- pass 2. The internal tide, the wake and the cross terms
    Zt = np.zeros((nt, ny_t, nx_t), np.complex64)                                    # sum_z S_r,it A*^2 dz
    A2z = np.zeros((nt, ny_t, nx_t), np.complex64)                                   # sum_z A*^2 dz
    s2 = {(p, s): np.zeros((ny_t, nx_t), np.float64) for p in ('bc', 'all') for s in 'rl'}
    acc = {q: np.zeros((nt, ny_t, nx_t), np.float32) for q in ('M2_bc', 'M2_all', 'X_bc', 'X_all')}
    Suz, Svz = np.zeros((nt, ny_t, nx_t), np.float32), np.zeros((nt, ny_t, nx_t), np.float32)
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        dzc = dz[..., z0:z1]
        uT, vT = load('TIDE', 'u', k, z0, z1), load('TIDE', 'v', k, z0, z1)
        du, dv = load('FULL', 'u', k, z0, z1) - uT, load('FULL', 'v', k, z0, z1) - vT
        # the wake with the tide, demodulated at omega/2
        A = lp_complex((bp(du, 'ni') + 1j*bp(dv, 'ni'))*car['half'][:, None, None, None])
        Ac2 = np.conj(A)**2
        del A
        # the semidiurnal band of TIDE and of the storm-made motion
        tu, tv = bp(uT, 'M2'), bp(vT, 'M2')
        su, sv = bp(du, 'M2'), bp(dv, 'M2')
        del uT, vT, du, dv
        bu, bv = tu - Ub[..., None], tv - Vb[..., None]                              # the internal tide
        acc['M2_all'] += (0.5*RHO0*(tu*tu + tv*tv)*dzc).sum(-1)
        acc['M2_bc'] += (0.5*RHO0*(bu*bu + bv*bv)*dzc).sum(-1)
        acc['X_all'] += (RHO0*(tu*su + tv*sv)*dzc).sum(-1)
        acc['X_bc'] += (RHO0*(bu*su + bv*sv)*dzc).sum(-1)
        Suz += (su*dzc).sum(-1)
        Svz += (sv*dzc).sum(-1)
        del tu, tv, su, sv
        # the rotary strain of the internal tide in both senses
        sig = rotary_strain(bu, bv, dxi, deta)
        del bu, bv
        ok_t &= np.isfinite(sig[0]).all(axis=-1)
        sig = np.nan_to_num(sig)
        Sr = lp_complex(sig*car['full'][:, None, None, None])
        Sl = lp_complex(sig*car['back'][:, None, None, None])
        del sig
        Zt += ((Sr*Ac2)*dzc).sum(axis=-1)
        A2z += (Ac2*dzc).sum(axis=-1)
        s2[('bc', 'r')] += ((np.abs(Sr[WORK])**2)*dzc).sum(axis=-1).mean(axis=0)
        s2[('bc', 'l')] += ((np.abs(Sl[WORK])**2)*dzc).sum(axis=-1).mean(axis=0)
        s2[('all', 'r')] += ((np.abs(Sr[WORK] + Sr_bt[WORK][..., None])**2)*dzc).sum(axis=-1).mean(axis=0)
        s2[('all', 'l')] += ((np.abs(Sl[WORK] + Sl_bt[WORK][..., None])**2)*dzc).sum(axis=-1).mean(axis=0)
        del Sr, Sl, Ac2
        step += 1
        print(f'   tile {k + 1} of {ntiles}, levels {z0} to {z1 - 1} of {nz}, {clock(t0, step, nsteps)}', flush=True)
    # the barotropic tide acts on the depth sums
    Z['bc'][:, b0:b1, cols] = Zt[:, rows]
    Z['bt'][:, b0:b1, cols] = (Sr_bt*A2z)[:, rows]
    s2[('bt', 'r')] = (np.abs(Sr_bt[WORK])**2).mean(axis=0)*H
    s2[('bt', 'l')] = (np.abs(Sl_bt[WORK])**2).mean(axis=0)*H
    for key in s2:
        S2[key][b0:b1, cols] = s2[key][rows]
    Hcol[b0:b1, cols] = H[rows]
    valid[b0:b1, cols] = ok_t[rows]
    acc['M2_bt'] = (0.5*RHO0*(Ub*Ub + Vb*Vb)*H[None]).astype(np.float32)
    acc['X_bt'] = (RHO0*(Ub*Suz + Vb*Svz)).astype(np.float32)
    m = foot[b0:b1, cols]
    a_t = area_s[b0:b1, cols]
    for q in QE:
        x = acc[q][:, rows]
        series[q] += (x[:, m]*a_t[m]).sum(axis=1)
    del acc, Zt, A2z, Sr_bt, Sl_bt, Ub, Vb, Suz, Svz

good = foot & valid
for p in PARTS:
    Z[p][:, ~valid] = 0.0
Z['all'] = Z['bt'] + Z['bc']
strain_amp = {key: np.sqrt(S2[key]/np.maximum(Hcol, 1e-6)) for key in S2}          # s-1, rms over depth and the work window

# --------------------------------------------------------------------------
# blocks, the phase, the lock and the strain of each source
# --------------------------------------------------------------------------


def block_table(p):
    """Block means of Z of one source and the quantities fig_04_compute.py forms from them."""
    Zw = Z[p][WAKE].mean(axis=0)                                                    # the effective phase of the window
    Zb = np.zeros((nt, nblk), np.complex64)
    rec = []
    for b, i in blk_index.items():
        m = good & (blk == b)
        if not m.any():
            continue
        a = area_s[m]
        Zb[:, i] = (Z[p][:, m]*a).sum(axis=1)/a.sum()
        zw = (Zw[m]*a).sum()/a.sum()
        rec.append(dict(block=int(b), i=i, area=float(a.sum()), lon=float(np.mean(lon_s[m])),
                        lat=float(np.mean(lat_s[m])), dc=float(np.sum(dc_full[m]*a)/a.sum()),
                        free_share=float(np.mean(dc_full[m] < 0)),
                        phi=float(np.angle(-zw)), cosphi=float(np.cos(np.angle(-zw))), amp=float(np.abs(zw))*0.5*RHO0,
                        work_from_Z=float(-0.5*RHO0*Zb[WORK, i].real.sum()*dt_s),
                        max_work=float(0.5*RHO0*np.abs(Zb[WORK, i]).sum()*dt_s),
                        Sr=float(np.sum(strain_amp[(p, 'r')][m]*a)/a.sum()),
                        Sl=float(np.sum(strain_amp[(p, 'l')][m]*a)/a.sum())))
    df = pd.DataFrame(rec)
    df['class'] = np.where(df.free_share > 0.5, 'free', 'trapped')
    bi_ = df.i.values
    num, den = np.abs(Zb[WAKE][:, bi_].sum(axis=0)), np.abs(Zb[WAKE][:, bi_]).sum(axis=0)
    df['coh'] = np.where(den > 0, num/np.maximum(den, 1e-30), np.nan)
    return Zw, Zb, df


def resultant(Zs_, bi_, floor):
    R, n = np.full(nt, np.nan), np.zeros(nt, int)
    for j in range(nt):
        m = np.abs(Zs_[j, bi_]) > floor
        n[j] = int(m.sum())
        if m.sum() >= 5:
            R[j] = np.abs(np.exp(1j*np.angle(-Zs_[j, bi_][m])).mean())
    return R, n


CLASS_NAME = {'free': 'below resonance', 'trapped': 'above resonance'}
Zw, Zb, tabs, out = {}, {}, {}, {}
for p in SRC:
    Zw[p], Zb[p], tabs[p] = block_table(p)
    df, bi = tabs[p], tabs[p].i.values
    zf = Zw[p][good]
    o = dict(phi=float(np.angle(-(zf*area_s[good]).sum())),
             share_pos=float(np.sum(area_s[good]*(np.cos(np.angle(-zf)) > 0))/area_s[good].sum()))
    o['cos'] = float(np.cos(o['phi']))
    for c in CLASS_NAME:
        m = df['class'].values == c
        o[f'cos_{c}'] = float(np.cos(np.angle((df.amp.values[m]*np.exp(1j*df.phi.values[m])).sum()))) if m.sum() >= 3 else np.nan
        o[f'coh_{c}'] = float(np.nanmedian(df.coh.values[m])) if m.sum() >= 3 else np.nan
        o[f'n_{c}'] = int(m.sum())
    o['coh'] = float(np.nanmedian(df.coh.values))
    R, n = resultant(Zb[p], bi, AMP_FLOOR*np.median(np.abs(Zb[p][WAKE][:, bi])))
    ray = np.sqrt(-np.log(0.05)/np.maximum(n, 1))
    o.update(R_wake=float(np.nanmean(R[WAKE])), R_storm=float(np.nanmean(R[WORK & ~WAKE])),
             rayleigh=float(np.nanmean(ray[WAKE])), hours_above=int(np.nansum((R > ray)[WAKE])), hours=int(WAKE.sum()))
    o['R_t'] = R
    o['phi_foot_t'] = np.angle(-Zb[p][:, bi].sum(axis=1))
    # the strain work rebuilt from Z and the largest work at any phase, footprint mean (J m-2)
    w = df.area.values
    o['work'] = float(np.sum(df.work_from_Z.values*w)/w.sum())
    o['max_work'] = float(np.sum(df.max_work.values*w)/w.sum())
    o['eff'] = o['work']/o['max_work'] if o['max_work'] > 0 else np.nan
    out[p] = o

# the half-width of the lock |S_r|/2, median of the blocks (those of Figure 5 when the block tables are there)
use = np.ones(len(tabs['all']), bool)
which = 'all blocks of the footprint'
if os.path.exists(FIG4_BLOCKS) and os.path.exists(FIG3_BLOCKS):
    keep = set(pd.read_csv(FIG4_BLOCKS).block.values) & set(pd.read_csv(FIG3_BLOCKS).block.values)
    use = np.isin(tabs['all'].block.values, list(keep))
    which = f'the {int(use.sum())} blocks of Figure 5'
for p in SRC:
    df = tabs[p]
    out[p]['half_S'] = float(np.nanmedian(0.5*df.Sr.values[use]*RAD2CPH))           # cph
    out[p]['Sr_foot'] = float(np.sqrt(np.sum(S2[(p, 'r')][good]*area_s[good])/np.sum(Hcol[good]*area_s[good])))
    out[p]['Sl_foot'] = float(np.sqrt(np.sum(S2[(p, 'l')][good]*area_s[good])/np.sum(Hcol[good]*area_s[good])))

# the agreement of the phase of each part with the phase of the whole tide
agree = {}
ta, bi = tabs['all'], tabs['all'].i.values
for p in PARTS:
    tp = tabs[p]
    d = np.angle(np.exp(1j*(tp.phi.values - ta.phi.values)))                        # blocks, wake window
    wgt = ta.amp.values
    cross = (Zb[p][WAKE][:, bi]*np.conj(Zb['all'][WAKE][:, bi]))                    # blocks and hours
    den = (np.abs(Zb[p][WAKE][:, bi])*np.abs(Zb['all'][WAKE][:, bi])).sum()
    agree[p] = dict(mean_cos=float(np.sum(wgt*np.cos(d))/wgt.sum()), med_abs_deg=float(np.degrees(np.median(np.abs(d)))),
                    same_sign=float(np.mean(np.sign(tp.cosphi.values) == np.sign(ta.cosphi.values))),
                    resultant_t=float(np.abs(cross.sum())/den) if den > 0 else np.nan,
                    lag_deg=float(np.degrees(np.angle(cross.sum()))),
                    amp_ratio=float(np.sum(np.abs(Zb[p][WAKE][:, bi])*ta.area.values[None])
                                    / np.sum(np.abs(Zb['all'][WAKE][:, bi])*ta.area.values[None])))

# --------------------------------------------------------------------------
# the semidiurnal kinetic energy, the deficit and the residence time
# --------------------------------------------------------------------------


def smooth(x, n=SMOOTH_H):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


ser = {q: series[q]/foot_area for q in QE}                                         # J m-2
X_sm = {p: smooth(ser[f'X_{p}']) - np.mean(smooth(ser[f'X_{p}'])[PRE]) for p in SRC}   # cross term above its pre-storm level
ke = {p: float(np.mean(ser[f'M2_{p}'][WAKE6])) for p in SRC}
deficit = {p: float(-np.mean(X_sm[p][WAKE6])) for p in SRC}
rate, loss_rate = {}, {}
if os.path.exists(FIGS10):
    s10 = xr.open_dataset(FIGS10)
    assert s10.sizes['time'] == nt, 'figS10_tide_parts.nc is not on the slab time axis'
    for p in SRC:
        rate[p] = float(np.mean(s10[f'ser_P_{p}'].values[WAKE6]))                   # W m-2, the work on the wake
        loss_rate[p] = float(-np.mean(s10[f'ser_T_{p}'].values[WAKE6]))             # W m-2, the loss of the tide
L_foot = np.sqrt(foot_area)


def residence(dfc, r):
    """Days and the group speed over the footprint (m s-1), as fig_06_compute.py. Not defined for a rate below zero."""
    if not np.isfinite(r) or r <= 0 or dfc <= 0:
        return np.nan, np.nan
    return dfc/r/86400.0, L_foot/(dfc/r)


res = {p: residence(deficit[p], rate.get(p, np.nan)) for p in SRC}
res_T = {p: residence(deficit[p], loss_rate.get(p, np.nan)) for p in SRC}

# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
check = []
check.append(f'   Z of the two parts is summed into the whole by construction. Cross term: largest difference of the hourly '
             f'series, parts against direct, {np.max(np.abs(ser["X_bt"] + ser["X_bc"] - ser["X_all"])):.2e} of '
             f'{np.max(np.abs(ser["X_all"])):.2e} J m-2')
check.append(f'   kinetic energy: largest difference, parts against direct, '
             f'{np.max(np.abs(ser["M2_bt"] + ser["M2_bc"] - ser["M2_all"])):.2e} of {np.max(ser["M2_all"]):.2e} J m-2')
if os.path.exists(FIG4_BLOCKS):
    b4 = pd.read_csv(FIG4_BLOCKS).merge(tabs['all'][['block', 'phi', 'Sr', 'coh']], on='block', suffixes=('_f4', ''))
    dphi = np.abs(np.angle(np.exp(1j*(b4.phi.values - b4.phi_f4.values))))
    check.append(f'   fig4_blocks.csv, {len(b4)} blocks: largest phase difference {np.degrees(np.nanmax(dphi)):.3f} deg, '
                 f'largest relative difference of |S_r| {np.nanmax(np.abs(b4.Sr.values/b4.Sr_f4.values - 1)):.2e}'
                 + (f', of the coherence {np.nanmax(np.abs(b4.coh.values - b4.coh_with.values)):.2e}'
                    if 'coh_with' in b4 else ''))
else:
    check.append('   fig4_blocks.csv was not found, no check of the phase')
if os.path.exists(FIG6):
    ts6 = eval(xr.open_dataset(FIG6).attrs['tide_side'], {'nan': np.nan, 'inf': np.inf})
    check.append(f'   fig6_budget.nc: deficit of the whole tide {deficit["all"]:.2f} J m-2 here against '
                 f'{ts6["deficit_wake"]:.2f}, residence time {res["all"][0]:.2f} d here against {ts6["tau_days"]:.2f}')
else:
    check.append('   fig6_budget.nc was not found, no check of the deficit')
if os.path.exists(FIGS10):
    tb = eval(s10.attrs['table'])
    check.append('   strain work rebuilt from Z over the work window against figS10 (J m-2, the rebuilt work keeps only '
                 'the part held by the 72 h demodulation): '
                 + ', '.join(f'{NAME[p]} {out[p]["work"]:+.1f} against {tb[f"sig|{p}|work"]:+.1f}' for p in SRC))

# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
blocks = tabs['all'][['block', 'lon', 'lat', 'area', 'dc', 'free_share', 'class']].copy()
for p in SRC:
    for col in ('phi', 'cosphi', 'amp', 'work_from_Z', 'max_work', 'Sr', 'Sl', 'coh'):
        blocks[f'{col}_{p}'] = tabs[p][col].values
blocks.to_csv(OUTDIR + 'figS11_blocks.csv', index=False)

ds = xr.Dataset(
    data_vars={**{f'cosphi_{p}': (('y', 'x'), np.where(good, np.cos(np.angle(-Zw[p])), np.nan).astype(np.float32)) for p in SRC},
               **{f'Zamp_{p}': (('y', 'x'), np.where(good, 0.5*RHO0*np.abs(Zw[p]), np.nan).astype(np.float32)) for p in SRC},
               **{f'Sr_{p}': (('y', 'x'), np.where(valid, strain_amp[(p, "r")], np.nan).astype(np.float32)) for p in SRC},
               **{f'Sl_{p}': (('y', 'x'), np.where(valid, strain_amp[(p, "l")], np.nan).astype(np.float32)) for p in SRC},
               **{f'R_{p}': (('time',), out[p]['R_t']) for p in SRC},
               **{f'phi_foot_{p}': (('time',), out[p]['phi_foot_t']) for p in SRC},
               **{f'ser_{q}': (('time',), ser[q]) for q in QE},
               **{f'X_sm_{p}': (('time',), X_sm[p]) for p in SRC},
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'foot': (('y', 'x'), foot.astype(np.int8)), 'dc_FULL': (('y', 'x'), dc_full)},
    coords={'time': t},
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF), lp_h=LP_H, smooth_h=SMOOTH_H,
               phase=str({p: {k_: v for k_, v in out[p].items() if not k_.endswith('_t')} for p in SRC}),
               agree=str(agree), ke=str(ke), deficit=str(deficit), rate=str(rate), loss_rate=str(loss_rate),
               residence=str(res), residence_loss=str(res_T), half_S_blocks=which))
ds.to_netcdf(OUTDIR + 'figS11_phase_parts.nc')


def row(label, fn, w=44):
    return f'   {label:<{w}s}' + ''.join(f'{fn(p):>12s}' for p in SRC)


HEAD = f'   {"":<44s}' + ''.join(f'{x:>12s}' for x in ('tide', 'barotropic', 'internal'))
L = ['figS11_compute.py. The phase, the lock and the deficit by part of the tide', '',
     'The tide is the semidiurnal band of the velocity of TIDE. Barotropic tide: its depth mean. Internal tide: the',
     'deviation from the depth mean. Z, the cross term and the kinetic energy are linear or exactly additive, so the two',
     'parts add up to the quantities of Figures 3d and 4. The phase and the strain amplitude are formed for each part.',
     f'Footprint {foot_area/1e6:.0f} km2, {len(blocks)} blocks. Wake window {str(T_OFF)[:16]} to {T_WAKE1}. '
     f'Demodulation cut-off {LP_H:.0f} h.', '',
     '1. THE PHASE (wake window, Phi = pi + arg Z)', HEAD,
     row('phase of the footprint (deg)', lambda p: f'{np.degrees(out[p]["phi"]):+.0f}'),
     row('phase efficiency cos(Phi) of the footprint', lambda p: f'{out[p]["cos"]:+.2f}'),
     row('share of the area with a gaining phase', lambda p: f'{out[p]["share_pos"]:.2f}')]
for c, cn in CLASS_NAME.items():
    L.append(row(f'cos(Phi) {cn} ({out["all"][f"n_{c}"]} blocks)', lambda p: f'{out[p][f"cos_{c}"]:+.2f}'))
L += ['', '2. THE PHASE OF EACH PART AGAINST THE PHASE OF THE WHOLE TIDE',
      f'   {"":<58s}' + ''.join(f'{x:>12s}' for x in ('barotropic', 'internal'))]
for lab, key, fmt in (('blocks: amplitude-weighted mean cosine of the difference', 'mean_cos', '{:+.2f}'),
                      ('blocks: median absolute difference (deg)', 'med_abs_deg', '{:.1f}'),
                      ('blocks: share with the same sign of cos(Phi)', 'same_sign', '{:.2f}'),
                      ('blocks and hours: resultant of the difference', 'resultant_t', '{:.2f}'),
                      ('blocks and hours: mean difference (deg)', 'lag_deg', '{:+.1f}'),
                      ('|Z| of the part over |Z| of the whole tide', 'amp_ratio', '{:.2f}')):
    L.append(f'   {lab:<58s}' + ''.join(f'{fmt.format(agree[p][key]):>12s}' for p in PARTS))
L += ['', '3. THE LOCK (resultant length of exp(i Phi) over the blocks) AND THE COHERENCE IN TIME', HEAD,
      row('resultant length, wake window', lambda p: f'{out[p]["R_wake"]:.2f}'),
      row('resultant length, storm window', lambda p: f'{out[p]["R_storm"]:.2f}'),
      row(f'hours above the Rayleigh level (of {out["all"]["hours"]})', lambda p: f'{out[p]["hours_above"]}'),
      row('coherence of the block phases (median)', lambda p: f'{out[p]["coh"]:.2f}')]
for c, cn in CLASS_NAME.items():
    L.append(row(f'coherence {cn}', lambda p: f'{out[p][f"coh_{c}"]:.2f}'))
L += ['', f'4. THE ROTARY STRAIN AND THE HALF-WIDTH OF THE LOCK (work window, {which})', HEAD,
      row('half-width |S_r|/2, block median (1e-3 cph)', lambda p: f'{out[p]["half_S"]*1e3:.3f}'),
      row('rms anticlockwise strain |S_r| (1e-6 s-1)', lambda p: f'{out[p]["Sr_foot"]*1e6:.3f}'),
      row('rms clockwise strain |S_l| (1e-6 s-1)', lambda p: f'{out[p]["Sl_foot"]*1e6:.3f}'),
      f'   share of the squared anticlockwise strain: barotropic {(out["bt"]["Sr_foot"]/out["all"]["Sr_foot"])**2:.2f}, '
      f'internal {(out["bc"]["Sr_foot"]/out["all"]["Sr_foot"])**2:.2f}',
      f'   half-width of the internal tide over that of the whole tide {out["bc"]["half_S"]/out["all"]["half_S"]:.2f}']
L += ['', '5. THE STRAIN WORK REBUILT FROM Z AND THE EFFICIENCY (work window, footprint blocks)', HEAD,
      row('work from Z (J m-2)', lambda p: f'{out[p]["work"]:+.1f}'),
      row('largest work at any phase (J m-2)', lambda p: f'{out[p]["max_work"]:.1f}'),
      row('efficiency (work over the largest work)', lambda p: f'{out[p]["eff"]:+.2f}')]
L += ['', f'6. THE SEMIDIURNAL BAND OF THE TIDE OVER THE FOOTPRINT (wake window of Figure 3d, series smoothed over {SMOOTH_H} h)',
      HEAD,
      row('kinetic energy (J m-2)', lambda p: f'{ke[p]:.1f}'),
      row('deficit of the band (J m-2)', lambda p: f'{deficit[p]:+.2f}')]
if rate:
    L += [row('rate of work on the wake (mW m-2)', lambda p: f'{1e3*rate[p]:+.3f}'),
          row('residence time, deficit over that rate (d)', lambda p: f'{res[p][0]:.2f}'),
          row('group speed over the footprint (m s-1)', lambda p: f'{res[p][1]:.2f}'),
          row('rate of loss of the tide (mW m-2)', lambda p: f'{1e3*loss_rate[p]:+.3f}'),
          row('residence time, deficit over the loss rate (d)', lambda p: f'{res_T[p][0]:.2f}'),
          f'   scale of the footprint {L_foot/1e3:.0f} km. A residence time is written only for a positive deficit and rate.']
else:
    L.append('   figS10_tide_parts.nc was not found, so no rate of work and no residence time by part')
L += ['', '7. CHECKS'] + check
L += ['', 'READING',
      f'   The phase of the internal tide against the wake differs from that of the whole tide by a median of '
      f'{agree["bc"]["med_abs_deg"]:.0f} deg over the blocks, with the same sign of cos(Phi) in '
      f'{100*agree["bc"]["same_sign"]:.0f}% of them. The barotropic tide: {agree["bt"]["med_abs_deg"]:.0f} deg and '
      f'{100*agree["bt"]["same_sign"]:.0f}%.',
      f'   Half-width of the lock: whole tide {out["all"]["half_S"]*1e3:.2f}, internal tide {out["bc"]["half_S"]*1e3:.2f}, '
      f'barotropic tide {out["bt"]["half_S"]*1e3:.2f} (1e-3 cph).',
      f'   Deficit of the semidiurnal band: whole tide {deficit["all"]:+.1f}, internal tide {deficit["bc"]:+.1f}, '
      f'barotropic tide {deficit["bt"]:+.1f} J m-2'
      + (f'. Residence time: whole tide {res["all"][0]:.1f} d, internal tide {res["bc"][0]:.1f} d.' if rate else '.')]
with open(OUTDIR + 'figS11_phase_parts_report.txt', 'w') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nsaved {OUTDIR}figS11_phase_parts.nc, figS11_blocks.csv and the report, '
      f'{(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min')