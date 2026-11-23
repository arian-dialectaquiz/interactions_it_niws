"""
fig_06_compute.py  (full_v5, revision 2)
----------------------------------------
Figure 6. The budget from both sides.

The kinetic energy budget of each wake over the footprint, the tide side of
the exchange against the semidiurnal band, and the band at the sum of the
tidal and the wake frequencies as a second tracer of the interaction.

The wake with the tide is u_FULL - u_TIDE and the wake without it is
u_WIND - u_BACK, band-passed to the near-inertial band of Figure 2. The
kinetic energy of each wake, integrated over depth, obeys over the footprint

    dE/dt = Pi + P + P_sm + R,

where Pi is the wind work on the wake (Figure 2), P the work of the internal
tide (Figure 3, strain, divergence and vertical shear), P_sm the work of the
storm-made semidiurnal motion of the pair (Figure 3) and R the residual,
which holds the work against the balanced flow, the radiation out of the
footprint and the dissipation. For the wake without the tide the two works
are the kinematic pairing (the strain of the internal tide on a wake that
never met it) and the strain work of its own storm-made motion, both strain
only (Figure 3).

The tide side. In the semidiurnal band u_M2(FULL) = u_M2(TIDE) + u_M2(FULL -
TIDE) exactly, so the kinetic energy of the band in FULL is that of the
internal tide, that of the storm-made motion and the cross term
rho0 <u_M2(TIDE) . u_M2(FULL - TIDE)>. At first order in the interaction the
cross term is the change of the energy of the internal tide, which the tide
side of the triad T (Figure 3) accumulates. The same cross term formed with
the storm-made motion of the pair without the tide, u_M2(WIND - BACK), gives
the level two motions that never met reach.

The sum band. A wave at omega exchanging energy with a wave at omega/2 also
drives motion at 3 omega/2 through the same quadratic terms. The kinetic
energy in 1.40 to 1.60 omega (the sum of the tidal band and the wake band)
of FULL - TIDE against WIND - BACK is a tracer of the interaction that needs
no work term. Over the blocks its excess is set against the bilinear product
|S_r| E_w of the strain amplitude and the wake kinetic energy.

Inputs
  ../new_full/fig6_cache_<RUN>/     u and v on the 3 km slope tiles, four runs
  ./fig2_out/fig2_wake.nc           wake kinetic energy, wind work, semidiurnal kinetic energy
  ./fig3_out/fig3_strain.nc         the works, the tide side of the triad
  ./fig3_out/fig3_blocks.csv        works and wind work by block
  ./fig4_out/fig4_blocks.csv        strain amplitude by block

Outputs (in OUTDIR)
  fig6_budget.nc  fig6_blocks.csv  fig6_budget_report.txt

Run under nohup. One pass over the tiles, CHUNK_Z levels at a time, no
derivatives and no vertical velocity, so lighter than Figure 3.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

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
FIG3 = './fig3_out/fig3_strain.nc'
FIG3_BLOCKS = './fig3_out/fig3_blocks.csv'
FIG4_BLOCKS = './fig4_out/fig4_blocks.csv'
OUTDIR = './fig6_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2, as Figures 2 and 3
M2_BAND = (0.95, 1.05)                             # times omega_M2, as Figures 2 and 3
SUM_BAND = (1.40, 1.60)                            # times omega_M2, the tidal band plus the wake band
CHUNK_Z = 4
SMOOTH_H = 25
N_BOOT = 2000
N_DOSE = 6
MIN_BLOCKS = 6
rng = np.random.default_rng(23)

# --------------------------------------------------------------------------
# grid, slab geometry and the fields of Figures 2 and 3
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
f3 = xr.open_dataset(FIG3)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s and f2.sizes['time'] == nt, \
    'fig2_wake.nc is not on the slab geometry, rerun fig_02_compute.py'
assert f3.sizes['time'] == nt, 'fig3_strain.nc is not on the slab time axis, rerun fig_03_compute.py'
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
foot = slope_s & (sector_s == 0)
dc_full = f2.dc_FULL.values*RAD2CPH
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
AFTER = WORK & (t >= T_OFF)
WAKE = (t >= np.datetime64(WINDOWS['wake'][0])) & (t < np.datetime64(WINDOWS['wake'][1]))
PRE = (t >= np.datetime64(WINDOWS['pre'][0])) & (t < np.datetime64(WINDOWS['pre'][1]))
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
foot_area = float(area_s[foot].sum())
sec = list(f2.sector.values).index('South')
print(f'footprint {foot_area/1e6:.0f} km2, work window {T_STORM0} to {T_WAKE1}', flush=True)


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


om = OMEGA_M2*RAD2CPH
SOS = {'ni': band_sos(NI_BAND[0]*0.5*om, NI_BAND[1]*0.5*om),
       'M2': band_sos(M2_BAND[0]*om, M2_BAND[1]*om),
       'sum': band_sos(SUM_BAND[0]*om, SUM_BAND[1]*om)}


def bp(x, name):
    sos, pad = SOS[name]
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


def load(run, var, k, z0, z1):
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


# --------------------------------------------------------------------------
# the pass over the tiles
# --------------------------------------------------------------------------
# depth-integrated energies and cross terms, J m-2, per record and point
Q = ['E_with', 'E_without',                        # near-inertial kinetic energy of the two wakes
     'M2_tide', 'M2_sm', 'M2_ot',                  # semidiurnal kinetic energy of the three sources
     'X_with', 'X_null',                           # cross terms, tide with FULL - TIDE and with WIND - BACK
     'S_with', 'S_without', 'S_tide', 'S_back']    # sum band, pair differences and two runs alone
NAMES = {'E_with': 'near-inertial kinetic energy of the wake with the tide',
         'E_without': 'near-inertial kinetic energy of the wake without the tide',
         'M2_tide': 'semidiurnal kinetic energy of the internal tide (TIDE)',
         'M2_sm': 'semidiurnal kinetic energy of the storm-made motion (FULL - TIDE)',
         'M2_ot': 'semidiurnal kinetic energy of the storm-made motion of the other pair (WIND - BACK)',
         'X_with': 'cross term of the internal tide with the storm-made motion of its pair',
         'X_null': 'cross term of the internal tide with the storm-made motion of the other pair',
         'S_with': 'sum band kinetic energy of FULL - TIDE', 'S_without': 'sum band kinetic energy of WIND - BACK',
         'S_tide': 'sum band kinetic energy of TIDE', 'S_back': 'sum band kinetic energy of BACK'}
series = {q: np.zeros(nt) for q in Q}                                         # footprint sum, J
maps = {q: np.full((ny_s, nx_s), np.nan, np.float32) for q in Q}             # wake-window mean
ids = np.unique(blk[foot])
nblk = ids.size
bidx = {b: i for i, b in enumerate(ids)}
Eblk = {q: np.zeros((nt, nblk)) for q in ('E_with', 'E_without', 'S_with', 'S_without')}
Ablk = np.zeros(nblk)

t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    nz = dz.shape[-1]
    ny_t, nx_t = h_t.shape
    acc = {q: np.zeros((nt, ny_t, nx_t), np.float32) for q in Q}
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        dzc = dz[..., z0:z1]
        u = {r: load(r, 'u', k, z0, z1) for r in RUNS}
        v = {r: load(r, 'v', k, z0, z1) for r in RUNS}
        du, dv = u['FULL'] - u['TIDE'], v['FULL'] - v['TIDE']
        ou, ov = u['WIND'] - u['BACK'], v['WIND'] - v['BACK']
        # the two wakes
        au, av = bp(du, 'ni'), bp(dv, 'ni')
        acc['E_with'] += (0.5*RHO0*(au*au + av*av)*dzc).sum(-1)
        del au, av
        ku, kv = bp(ou, 'ni'), bp(ov, 'ni')
        acc['E_without'] += (0.5*RHO0*(ku*ku + kv*kv)*dzc).sum(-1)
        del ku, kv
        # the semidiurnal band, three sources and two cross terms
        tu, tv = bp(u['TIDE'], 'M2'), bp(v['TIDE'], 'M2')
        su, sv = bp(du, 'M2'), bp(dv, 'M2')
        acc['M2_tide'] += (0.5*RHO0*(tu*tu + tv*tv)*dzc).sum(-1)
        acc['M2_sm'] += (0.5*RHO0*(su*su + sv*sv)*dzc).sum(-1)
        acc['X_with'] += (RHO0*(tu*su + tv*sv)*dzc).sum(-1)
        del su, sv
        su, sv = bp(ou, 'M2'), bp(ov, 'M2')
        acc['M2_ot'] += (0.5*RHO0*(su*su + sv*sv)*dzc).sum(-1)
        acc['X_null'] += (RHO0*(tu*su + tv*sv)*dzc).sum(-1)
        del su, sv, tu, tv
        # the sum band
        for q, (xu, xv) in (('S_with', (du, dv)), ('S_without', (ou, ov)),
                            ('S_tide', (u['TIDE'], v['TIDE'])), ('S_back', (u['BACK'], v['BACK']))):
            fu, fv = bp(xu, 'sum'), bp(xv, 'sum')
            acc[q] += (0.5*RHO0*(fu*fu + fv*fv)*dzc).sum(-1)
            del fu, fv
        del u, v, du, dv, ou, ov
    m = foot[b0:b1, cols]
    a_t = area_s[b0:b1, cols]
    ids_t = blk[b0:b1, cols]
    for q in Q:
        x = acc[q][:, rows]
        maps[q][b0:b1, cols] = x[WAKE].mean(axis=0)
        series[q] += (x[:, m]*a_t[m]).sum(axis=1)
        if q in Eblk:
            for b in np.unique(ids_t[m]):
                mb = m & (ids_t == b)
                Eblk[q][:, bidx[b]] += (x[:, mb]*a_t[mb]).sum(axis=1)
    for b in np.unique(ids_t[m]):
        Ablk[bidx[b]] += a_t[m & (ids_t == b)].sum()
    del acc
    print(f'tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min', flush=True)

# --------------------------------------------------------------------------
# footprint series, the budget of each wake and the tide side
# --------------------------------------------------------------------------


def smooth(x, n=SMOOTH_H):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


def J(x, sel=WORK):
    return float(np.sum(x[sel])*dt_h*3600.0)


def cum(x):
    """Accumulated over the work window, J m-2, zero before it and held after it."""
    c = np.zeros(nt)
    kw = np.where(WORK)[0]
    c[kw] = np.cumsum(x[kw])*dt_h*3600.0
    c[kw[-1] + 1:] = c[kw[-1]]
    return c


ser = {q: series[q]/foot_area for q in Q}                                        # J m-2
for q in Eblk:
    Eblk[q] = Eblk[q]/np.maximum(Ablk, 1e-6)[None, :]                             # J m-2 by block

# consistency with Figure 2 (same band, same pairs)
E2 = {'with': f2.ser_Ewni_tide.values[:, sec], 'without': f2.ser_Ewni_notide.values[:, sec]}
consist = {w: dict(ratio=float(np.mean(ser[f'E_{w}'][WORK])/np.mean(E2[w][WORK])),
                   corr=float(np.corrcoef(ser[f'E_{w}'][WORK], E2[w][WORK])[0, 1])) for w in E2}
M2_full = f2.ser_M2_FULL.values[:, sec]
closure = dict(FULL=float(np.mean(M2_full[WAKE])),
               rebuilt=float(np.mean((ser['M2_tide'] + ser['M2_sm'] + ser['X_with'])[WAKE])),
               TIDE=float(np.mean(ser['M2_tide'][WAKE])), sm=float(np.mean(ser['M2_sm'][WAKE])),
               cross=float(np.mean(ser['X_with'][WAKE])), other=float(np.mean(ser['M2_ot'][WAKE])),
               null=float(np.mean(ser['X_null'][WAKE])),
               TIDE_fig2=float(np.mean(f2.ser_M2_TIDE.values[:, sec][WAKE])))

# the budget terms, W m-2
terms = {}
terms['with'] = {'tendency': np.gradient(smooth(ser['E_with']), dt_h*3600.0),
                 'wind work': f2.ser_Pi_tide.values[:, sec],
                 'internal tide': f3.ser_it.values,
                 'storm-made motion': f3.ser_sm.values}
terms['without'] = {'tendency': np.gradient(smooth(ser['E_without']), dt_h*3600.0),
                    'wind work': f2.ser_Pi_notide.values[:, sec],
                    'internal tide': f3.ser_sig_kin.values,
                    'storm-made motion': f3.ser_sig_ot.values}
for w in terms:
    terms[w]['residual'] = (terms[w]['tendency'] - terms[w]['wind work'] - terms[w]['internal tide']
                            - terms[w]['storm-made motion'])
TERMS = ['tendency', 'wind work', 'internal tide', 'storm-made motion', 'residual']
budget = {w: {k_: J(v_) for k_, v_ in terms[w].items()} for w in terms}
budget_after = {w: {k_: J(v_, AFTER) for k_, v_ in terms[w].items()} for w in terms}
budget_storm = {w: {k_: J(v_, WORK & ~AFTER) for k_, v_ in terms[w].items()} for w in terms}
for w in budget:
    budget[w]['change over the window'] = float(smooth(ser[f'E_{w}'])[np.where(WORK)[0][-1]]
                                                - smooth(ser[f'E_{w}'])[np.where(WORK)[0][0]])
    budget[w]['peak'] = float(smooth(ser[f'E_{w}'])[WORK].max())
    budget_after[w]['decay'] = float(smooth(ser[f'E_{w}'])[np.where(AFTER)[0][-1]]
                                     - smooth(ser[f'E_{w}'])[np.where(AFTER)[0][0]])
prop = dict(tide_over_wind=budget['with']['internal tide']/budget['with']['wind work'],
            tide_over_decay_after=-budget_after['with']['internal tide']/budget_after['with']['decay']
            if budget_after['with']['decay'] != 0 else np.nan,
            tide_over_residual=budget['with']['internal tide']/abs(budget['with']['residual'])
            if budget['with']['residual'] != 0 else np.nan,
            kin_over_tide=budget['without']['internal tide']/budget['with']['internal tide'])

# the tide side
T_cum = -cum(smooth(f3.ser_T_it.values))                                         # loss of the tide, J m-2, rates smoothed over 25 h
P_cum = cum(smooth(f3.ser_it.values))                                            # work on the wake, J m-2
X_sm = smooth(ser['X_with']) - np.mean(smooth(ser['X_with'])[PRE])               # cross term above its pre-storm level
X_nu = smooth(ser['X_null']) - np.mean(smooth(ser['X_null'])[PRE])
kw_end = np.where(WORK)[0][-1]
tide_side = dict(loss_T=float(T_cum[kw_end]), work_P=float(P_cum[kw_end]),
                 minus_T_over_P=float(T_cum[kw_end]/P_cum[kw_end]) if P_cum[kw_end] != 0 else np.nan,
                 cross_end=float(X_sm[kw_end]), cross_wake_mean=float(np.mean(X_sm[WAKE])),
                 null_wake_mean=float(np.mean(X_nu[WAKE])), null_abs_max=float(np.max(np.abs(X_nu[WORK]))),
                 cross_over_loss=float(-X_sm[kw_end]/T_cum[kw_end]) if T_cum[kw_end] != 0 else np.nan,
                 corr_cross_minus_loss=float(np.corrcoef(X_sm[AFTER], -T_cum[AFTER])[0, 1]),
                 tide_energy_wake=closure['TIDE'],
                 loss_share_of_tide=float(T_cum[kw_end]/closure['TIDE']) if closure['TIDE'] > 0 else np.nan)
# the tide is replenished by its incoming flux, so its standing energy carries a deficit of the order of the
# loss rate times the transit time of the tide through the footprint, not the accumulated loss
P_rate_wake = float(np.mean(f3.ser_it.values[WAKE]))                              # W m-2, wake window
deficit_wake = -tide_side['cross_wake_mean']                                      # J m-2
tau_s = deficit_wake/P_rate_wake if P_rate_wake > 0 else np.nan
L_foot = np.sqrt(foot_area)                                                       # m, the scale of the footprint
tide_side.update(P_rate_wake=P_rate_wake, deficit_wake=deficit_wake, tau_days=float(tau_s/86400.0),
                 L_foot_km=float(L_foot/1e3), cg_implied=float(L_foot/tau_s) if tau_s > 0 else np.nan)

# the sum band
sum_band = {}
for wname, sel in (('pre', PRE), ('storm', WORK & ~AFTER), ('wake', WAKE)):
    sum_band[wname] = {q: float(np.mean(ser[q][sel])) for q in ('S_with', 'S_without', 'S_tide', 'S_back')}
    sum_band[wname]['ratio'] = sum_band[wname]['S_with']/sum_band[wname]['S_without'] \
        if sum_band[wname]['S_without'] > 0 else np.nan
S_ratio_t = smooth(ser['S_with'])/np.maximum(smooth(ser['S_without']), 1e-9)

# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
b3 = pd.read_csv(FIG3_BLOCKS)
b4 = pd.read_csv(FIG4_BLOCKS)[['block', 'Sr', 'amp', 'cosphi']]
kw0, kw1 = np.where(WORK)[0][[0, -1]]
rows_b = []
for b, i in bidx.items():
    m = foot & (blk == b)
    if m.sum() < BLOCK*BLOCK//4 or Ablk[i] <= 0:
        continue
    rec = dict(block=int(b), n=int(m.sum()))
    for w in ('with', 'without'):
        e = smooth(Eblk[f'E_{w}'][:, i])
        rec[f'dE_{w}'] = float(e[kw1] - e[kw0])
        rec[f'E_{w}_wake'] = float(np.mean(Eblk[f'E_{w}'][WAKE, i]))
    rec['S_with_wake'] = float(np.mean(Eblk['S_with'][WAKE, i]))
    rec['S_without_wake'] = float(np.mean(Eblk['S_without'][WAKE, i]))
    rec['S_excess'] = rec['S_with_wake'] - rec['S_without_wake']
    rec['X_with_wake'] = float(np.nanmean(maps['X_with'][m]))
    rec['X_null_wake'] = float(np.nanmean(maps['X_null'][m]))
    rows_b.append(rec)
blocks = pd.DataFrame(rows_b).merge(
    b3[['block', 'lon', 'lat', 'class', 'free_share', 'work_it', 'work_sm', 'work_sig_kin', 'work_sig_ot',
        'work_T_it', 'W_tide', 'W_notide', 'Ew_tide', 'gain']], on='block', how='inner').merge(b4, on='block', how='left')
blocks['residual_with'] = blocks.dE_with - blocks.W_tide - blocks.work_it - blocks.work_sm
blocks['residual_without'] = blocks.dE_without - blocks.W_notide - blocks.work_sig_kin - blocks.work_sig_ot
blocks['product'] = blocks.Sr.values*blocks.Ew_tide.values                       # s-1 J m-2, the bilinear product
blocks.to_csv(OUTDIR + 'fig6_blocks.csv', index=False)
nb = len(blocks)
print(f'{nb} blocks', flush=True)

BLOCK_TERMS = {'with': {'tendency': 'dE_with', 'wind work': 'W_tide', 'internal tide': 'work_it',
                        'storm-made motion': 'work_sm', 'residual': 'residual_with'},
               'without': {'tendency': 'dE_without', 'wind work': 'W_notide', 'internal tide': 'work_sig_kin',
                           'storm-made motion': 'work_sig_ot', 'residual': 'residual_without'}}
CLASSES = ['free', 'trapped']
cls_terms = np.full((2, len(CLASSES), len(TERMS), 4), np.nan)                   # wake, class, term, (mean, p05, p95, n)
for iw, w in enumerate(('with', 'without')):
    for ic, c in enumerate(CLASSES):
        m = blocks['class'].values == c
        if m.sum() < 3:
            continue
        for it, term in enumerate(TERMS):
            x = blocks[BLOCK_TERMS[w][term]].values[m]
            x = x[np.isfinite(x)]
            if x.size < 3:
                continue
            bs = np.mean(x[rng.integers(0, x.size, size=(N_BOOT, x.size))], axis=1)
            cls_terms[iw, ic, it] = [x.mean(), np.percentile(bs, 5), np.percentile(bs, 95), x.size]


def shift_level(x, y, n=N_BOOT, q=95):
    r = np.array([abs(spearmanr(np.roll(x, s), y).correlation) for s in rng.integers(1, x.size - 1, size=n)])
    return float(np.percentile(r, q))


stats = {}
dose = np.full((N_DOSE, 5), np.nan)
if nb >= MIN_BLOCKS:
    ok = np.isfinite(blocks['product'].values) & np.isfinite(blocks.S_excess.values)
    x, y = blocks['product'].values[ok], blocks.S_excess.values[ok]
    stats['rho_excess_product'] = float(spearmanr(x, y).correlation)
    stats['rho_excess_product_level'] = shift_level(x, y)
    stats['rho_excess_work'] = float(spearmanr(blocks.work_it.values[ok], y).correlation)
    stats['rho_excess_work_level'] = shift_level(blocks.work_it.values[ok], y)
    stats['rho_excess_cos'] = float(spearmanr(blocks.cosphi.values[ok], y).correlation) \
        if np.isfinite(blocks.cosphi.values[ok]).all() else np.nan
    stats['excess_share_positive'] = float(np.mean(y > 0))
    stats['n'] = int(ok.sum())
    edges = np.quantile(x, np.linspace(0, 1, N_DOSE + 1))
    for i in range(N_DOSE):
        m = (x >= edges[i]) & ((x <= edges[i + 1]) if i == N_DOSE - 1 else (x < edges[i + 1]))
        if m.sum() < 3:
            continue
        bs = np.median(y[m][rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))], axis=1)
        dose[i] = [np.median(x[m]), np.median(y[m]), np.percentile(bs, 5), np.percentile(bs, 95), m.sum()]
    for c in CLASSES:
        m = blocks['class'].values == c
        if m.sum() >= 3:
            stats[f'S_ratio_{c}'] = float(blocks.S_with_wake.values[m].sum()/blocks.S_without_wake.values[m].sum())
            stats[f'cross_{c}'] = float(np.nanmean(blocks.X_with_wake.values[m]))
            stats[f'null_{c}'] = float(np.nanmean(blocks.X_null_wake.values[m]))
    stats['rho_cross_T'] = float(spearmanr(blocks.X_with_wake.values, blocks.work_T_it.values).correlation)
    stats['rho_cross_T_level'] = shift_level(blocks.X_with_wake.values, blocks.work_T_it.values)

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars={**{f'map_{q}': (('y', 'x'), maps[q]) for q in Q},
               **{f'ser_{q}': (('time',), ser[q]) for q in Q},
               **{f'term_{w}_{k_.replace(" ", "_").replace("-", "_")}': (('time',), v_)
                  for w in terms for k_, v_ in terms[w].items()},
               'T_cum': (('time',), T_cum), 'P_cum': (('time',), P_cum),
               'X_sm': (('time',), X_sm), 'X_null_s': (('time',), X_nu), 'S_ratio_t': (('time',), S_ratio_t),
               'cls_terms': (('wake', 'cls', 'term', 'stat'), cls_terms),
               'dose': (('dose_bin', 'dose_stat'), dose),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'slope': (('y', 'x'), slope_s), 'foot': (('y', 'x'), foot), 'area': (('y', 'x'), area_s),
               'dc_FULL': (('y', 'x'), dc_full), 'mask': (('y', 'x'), f2.mask.values)},
    coords=dict(time=t, wake=['with', 'without'], cls=CLASSES, term=TERMS, stat=['mean', 'p05', 'p95', 'n'],
                dose_bin=np.arange(N_DOSE), dose_stat=['product_med', 'excess_med', 'excess_p05', 'excess_p95', 'n']),
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF)[:16],
               ni_band=str(NI_BAND), m2_band=str(M2_BAND), sum_band=str(SUM_BAND), smooth_h=SMOOTH_H,
               foot_area_m2=foot_area, block_points=BLOCK, n_boot=N_BOOT, omega_M2=OMEGA_M2,
               budget=str(budget), budget_after=str(budget_after), budget_storm=str(budget_storm),
               prop=str(prop), tide_side=str(tide_side), closure=str(closure), sum_band_stats=str(sum_band),
               consist=str(consist), stats=str(stats), names=str(NAMES)))
ds.to_netcdf(OUTDIR + 'fig6_budget.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['FIGURE 6 REPORT (full_v5, revision 2)  --  the budget from both sides',
     '=' * 86,
     f'slab caches, {nt} records, footprint {foot_area/1e6:.0f} km2, {nb} blocks of {BLOCK} points',
     f'wake with the tide u_FULL - u_TIDE, without it u_WIND - u_BACK, near-inertial band {NI_BAND} of omega/2',
     f'semidiurnal band {M2_BAND} of omega, sum band {SUM_BAND} of omega, series smoothed over {SMOOTH_H} h',
     f'work window {T_STORM0} to {T_WAKE1}, wind stops {str(T_OFF)[:16]}',
     f'consistency with Figure 2, wake kinetic energy: with ratio {consist["with"]["ratio"]:.3f} corr '
     f'{consist["with"]["corr"]:.3f}, without ratio {consist["without"]["ratio"]:.3f} corr {consist["without"]["corr"]:.3f}',
     '',
     '1. THE BUDGET OF EACH WAKE OVER THE FOOTPRINT (J m-2 over the work window, then storm and after the wind stops)']
for w in ('with', 'without'):
    L.append(f'   wake {w} the tide (peak {budget[w]["peak"]:.0f}, change over the window {budget[w]["change over the window"]:.0f})')
    for term in TERMS:
        L.append(f'     {term:20s} {budget[w][term]:9.1f}   storm {budget_storm[w][term]:9.1f}   after {budget_after[w][term]:9.1f}')
    L.append(f'     decay after the wind stops {budget_after[w]["decay"]:.0f}')
L += ['   for the wake without the tide the internal tide term is the kinematic pairing and the storm-made',
      '   term is the strain work of WIND - BACK, both strain only (Figure 3)',
      f'   work of the internal tide over the wind work on the wake with the tide {prop["tide_over_wind"]:.3f}, '
      f'over the decay after the wind stops {prop["tide_over_decay_after"]:.3f}, over |residual| {prop["tide_over_residual"]:.3f}',
      f'   kinematic pairing over the work of the internal tide {prop["kin_over_tide"]:.2f}',
      '',
      '2. THE TIDE SIDE (footprint, J m-2)',
      f'   loss of the tide -int T dt over the window {tide_side["loss_T"]:.1f}, work on the wake int P dt {tide_side["work_P"]:.1f}, '
      f'ratio {tide_side["minus_T_over_P"]:.2f}',
      f'   semidiurnal kinetic energy in the wake window: FULL {closure["FULL"]:.0f} = TIDE {closure["TIDE"]:.0f} + storm-made '
      f'{closure["sm"]:.0f} + cross {closure["cross"]:.0f} (rebuilt {closure["rebuilt"]:.0f}), TIDE from Figure 2 {closure["TIDE_fig2"]:.0f}',
      f'   storm-made motion of the other pair {closure["other"]:.0f}, its cross term with the tide {closure["null"]:.0f}',
      f'   cross term above its pre-storm level at the end of the window {tide_side["cross_end"]:.1f}, wake-window mean '
      f'{tide_side["cross_wake_mean"]:.1f}, over the loss of the tide {tide_side["cross_over_loss"]:.2f}, '
      f'correlation with the accumulated loss after the wind stops {tide_side["corr_cross_minus_loss"]:.2f}',
      f'   the null cross term: wake-window mean {tide_side["null_wake_mean"]:.1f}, largest magnitude {tide_side["null_abs_max"]:.1f}',
      f'   loss of the tide as a share of its wake-window semidiurnal kinetic energy {tide_side["loss_share_of_tide"]:.3f}, '
      f'so the loss is drawn from the flux that replenishes the tide and not from its standing energy',
      f'   standing deficit (wake-window mean) {tide_side["deficit_wake"]:.1f} J m-2 against a loss rate of '
      f'{1e3*tide_side["P_rate_wake"]:.2f} mW m-2 gives a residence time of {tide_side["tau_days"]:.1f} d, which over the '
      f'{tide_side["L_foot_km"]:.0f} km of the footprint is a group speed of {tide_side["cg_implied"]:.1f} m s-1 (mode one on this slope)',
      f'   blocks, Spearman of the cross term against T {stats.get("rho_cross_T", np.nan):.2f} '
      f'(level {stats.get("rho_cross_T_level", np.nan):.2f})',
      '',
      '3. THE SUM BAND (1.40 to 1.60 omega, footprint means, J m-2)']
for wname in ('pre', 'storm', 'wake'):
    sb = sum_band[wname]
    L.append(f'   {wname:6s} FULL - TIDE {sb["S_with"]:8.2f}  WIND - BACK {sb["S_without"]:8.2f}  ratio {sb["ratio"]:.2f}'
             f'   TIDE {sb["S_tide"]:8.2f}  BACK {sb["S_back"]:8.2f}')
L += [f'   by class, with over without: free {stats.get("S_ratio_free", np.nan):.2f}, trapped {stats.get("S_ratio_trapped", np.nan):.2f}',
      f'   excess of the blocks against |S_r| E_w: Spearman {stats.get("rho_excess_product", np.nan):.2f} '
      f'(level {stats.get("rho_excess_product_level", np.nan):.2f}), against the work of the internal tide '
      f'{stats.get("rho_excess_work", np.nan):.2f} (level {stats.get("rho_excess_work_level", np.nan):.2f}), '
      f'against cos(Phi) {stats.get("rho_excess_cos", np.nan):.2f}, share of blocks with excess '
      f'{stats.get("excess_share_positive", np.nan):.2f}',
      '  product_med  excess_med        p05        p95    n']
for row in dose:
    if np.isfinite(row[0]):
        L.append(f'  {row[0]:11.3g} {row[1]:11.1f} {row[2]:10.1f} {row[3]:10.1f} {int(row[4]):4d}')
L += ['', '4. THE TERMS BY CLASS (block means, J m-2, 5 to 95% block bootstrap)']
for iw, w in enumerate(('with', 'without')):
    for ic, c in enumerate(CLASSES):
        n_ = cls_terms[iw, ic, 0, 3]
        if not np.isfinite(n_):
            continue
        L.append(f'   wake {w} the tide, {c} ({int(n_)} blocks)')
        for it, term in enumerate(TERMS):
            v_ = cls_terms[iw, ic, it]
            L.append(f'     {term:20s} {v_[0]:9.1f} [{v_[1]:9.1f}, {v_[2]:9.1f}]')
L += ['',
      'READING THE RESULT',
      '   question 4 (first half). The tide pays for the work if the loss of the tide matches the work',
      '   on the wake (ratio near one, Figure 3) and the semidiurnal band of FULL carries a standing',
      '   deficit of the order of the loss rate times the transit time of the tide through the',
      '   footprint, with the null cross term near zero. The sum band would be a tracer of the same',
      '   interaction if FULL - TIDE exceeded WIND - BACK in the wake window and the excess rose with',
      '   the bilinear product (on Rio it does neither, the band is the harmonic of the wake itself,',
      '   so it stays in the report). The budget sets the proportion',
      '   of the work of the internal tide against the wind work and against the decay of the wake,',
      '   and the residual, which holds the work against the balanced flow, the radiation and the',
      '   dissipation, is what the eddies of the two pairs do and what Figure 7 follows.']
open(OUTDIR + 'fig6_budget_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('saved', OUTDIR + 'fig6_budget.nc')