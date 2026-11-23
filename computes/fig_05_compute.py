"""
fig_05_compute.py  (full_v5, revision 2)
----------------------------------------
Figure 5. The frequency of the wake and what the boundary current does to it.

Figure 4 showed that the phase of the wake against the tide drifts and that
the sign of the work follows the detuning of the balanced flow. The wake
detuning measured from the drift of its own phase (Figure 4) is set against
two predictions for the frequency of the wake, that of a free inertial wave in
the balanced flow, f_eff = f + zeta/2 (Kunze 1985), so that the wake detuning
equals the flow detuning, and that of a wake whose frequency the storm set
above the planetary f and the vorticity leaves alone (the blue shift of a
storm wake, D'Asaro 1985, Alford 2016). A third possibility, the wake held at
omega/2 by the tide (the locked state of the amplitude equation of Young et
al. 2008, which holds for |Delta| < |S_r|/2), is settled by one number, the
strain of the internal tide against the wake detuning, and goes to the report.

Tests
  the frequency of the wake. The wake detuning against the flow detuning by
    bin with block-bootstrap bands, against the slope-one line of a free wave
    and the horizontal line of a fixed blue shift (the median wake detuning).
  the Kunze test. The wake frequency minus the planetary |f| against zeta/2
    over the blocks, slope one for a free wave, with the Theil and Sen slope.
  the lock. The share of blocks with |Delta| < |S_r|/2 and the share of them
    whose wake detuning is under a threshold (report only).
  where the wake gains. The gain of each block (Figure 2) and the tidal strain
    work on it (Figure 3) by bin of the flow detuning with bands.
  what controls them. Spearman and partial rank correlations (the wake kinetic
    energy held fixed) of the tidal strain work and of the gain with the
    candidates, flow detuning, its magnitude, vorticity, |S_r|, cos(Phi),
    wake kinetic energy, wind work difference, latitude and depth, each
    against a circular-shift level.

Inputs
  ./fig2_out/fig2_wake.nc        geometry, detuning and vorticity of FULL
  ./fig3_out/fig3_blocks.csv     gain, tidal strain work, wind work difference
  ./fig4_out/fig4_phase.nc       drift map, |S_r| map (revision 4)
  ./fig4_out/fig4_blocks.csv     wake detuning, |S_r|, cos(Phi), coherence

Outputs (in OUTDIR)
  fig5_resonance.nc  fig5_blocks.csv  fig5_resonance_report.txt

Runs in seconds (no pass over the slab caches).
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, theilslopes, rankdata, fisher_exact

FIG2 = './fig2_out/fig2_wake.nc'
FIG3_BLOCKS = './fig3_out/fig3_blocks.csv'
FIG4 = './fig4_out/fig4_phase.nc'
FIG4_BLOCKS = './fig4_out/fig4_blocks.csv'
OUTDIR = './fig5_out/'
os.makedirs(OUTDIR, exist_ok=True)

RAD2CPH = 3600.0/(2*np.pi)
OMEGA_EARTH = 7.2921159e-5                         # rad s-1, the planetary f of a block is 2 Omega sin(lat)
DC_EDGES = np.arange(-0.008, 0.0061, 0.001)        # cph
MIN_BIN = 5
N_BOOT = 2000
HELD_CPH = 0.0005                                  # |wake detuning| under this counts as held
rng = np.random.default_rng(17)

# --------------------------------------------------------------------------
# fields and tables
# --------------------------------------------------------------------------
f2 = xr.open_dataset(FIG2)
f4 = xr.open_dataset(FIG4)
assert 'drift_cph' in f4, 'fig4_phase.nc lacks the drift map, rerun fig_04_compute.py (revision 4)'
lon, lat, h = f2.lon.values, f2.lat.values, f2.h.values
area, slope, sector = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
foot = slope & (sector == 0)
dc = f2.dc_FULL.values*RAD2CPH                                       # cph, wake window
zof = f2.zof_FULL.values                                             # zeta / f
drift = f4.drift_cph.values                                          # cph, wake window
Sr = f4.Sr_amp.values*RAD2CPH                                        # cph
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(h.shape[0])[:, None]//BLOCK)*1000 + (np.arange(h.shape[1])[None, :]//BLOCK)

b3 = pd.read_csv(FIG3_BLOCKS)
b4 = pd.read_csv(FIG4_BLOCKS)
blocks = b4[['block', 'lon', 'lat', 'n', 'dc', 'free_share', 'class', 'phi', 'cosphi', 'amp',
             'Sr', 'drift_cph', 'Ew', 'coh_with', 'coh_without']].merge(
    b3[['block', 'gain', 'work_sig_it', 'work_it', 'wind_diff', 'Ew_tide', 'W_tide', 'W_notide']],
    on='block', how='inner')
# depth and vorticity of each block from the maps
dep, vort = [], []
for b in blocks.block.values:
    m = foot & (blk == b)
    a = area[m]
    dep.append(float(np.sum(h[m]*a)/a.sum()))
    vort.append(float(np.nansum(zof[m]*a)/a[np.isfinite(zof[m])].sum()))
blocks['depth'] = dep
blocks['zof'] = vort
blocks['Sr_cph'] = blocks.Sr.values*RAD2CPH
blocks['half_S'] = 0.5*blocks.Sr_cph.values
# the planetary |f| of each block from its latitude (the ROMS f is the same function) and the Kunze quantities (cph)
blocks['f_cph'] = 2.0*OMEGA_EARTH*np.sin(np.radians(np.abs(blocks.lat.values)))*RAD2CPH
OMEGA_HALF_CPH = 0.5*float(f2.attrs['omega_M2'])*RAD2CPH
blocks['wake_minus_f'] = OMEGA_HALF_CPH + blocks.drift_cph.values - blocks.f_cph.values   # omega_w - |f|
blocks['half_zeta'] = 0.5*blocks.zof.values*blocks.f_cph.values                           # zeta/2 in cph (signed as f_eff - f)
blocks['held'] = np.abs(blocks.drift_cph.values) < HELD_CPH
blocks['inside'] = np.abs(blocks.dc.values) < blocks.half_S.values
blocks['adler'] = np.sign(blocks.dc.values)*np.sqrt(np.maximum(blocks.dc.values**2 - blocks.half_S.values**2, 0.0))
nb = len(blocks)
print(f'{nb} blocks', flush=True)

# --------------------------------------------------------------------------
# the lock, wake detuning against flow detuning by bin
# --------------------------------------------------------------------------
centres = 0.5*(DC_EDGES[1:] + DC_EDGES[:-1])


def bins_of(y, x, weights=None):
    out = np.full((centres.size, 4), np.nan)
    for i in range(centres.size):
        m = (x >= DC_EDGES[i]) & (x < DC_EDGES[i + 1]) & np.isfinite(y)
        if m.sum() < MIN_BIN:
            continue
        v = y[m]
        draws = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        bs = np.median(v[draws], axis=1)
        out[i] = [np.median(v), np.percentile(bs, 5), np.percentile(bs, 95), m.sum()]
    return out


lock_bins = bins_of(blocks.drift_cph.values, blocks.dc.values)
half_S_med = float(np.nanmedian(blocks.half_S.values))
adler_curve = np.sign(centres)*np.sqrt(np.maximum(centres**2 - half_S_med**2, 0.0))
ok = np.isfinite(blocks.drift_cph.values)
ts_all = theilslopes(blocks.drift_cph.values[ok], blocks.dc.values[ok])
ins = ok & blocks.inside.values
outs = ok & ~blocks.inside.values
lock = dict(slope_all=float(ts_all[0]), slope_lo=float(ts_all[2]), slope_hi=float(ts_all[3]),
            rho_all=float(spearmanr(blocks.dc.values[ok], blocks.drift_cph.values[ok]).correlation),
            half_S_med=half_S_med, n_inside=int(ins.sum()), n_outside=int(outs.sum()),
            held_inside=float(np.mean(blocks.held.values[ins])) if ins.sum() else np.nan,
            held_outside=float(np.mean(blocks.held.values[outs])) if outs.sum() else np.nan,
            drift_abs_inside=float(np.median(np.abs(blocks.drift_cph.values[ins]))) if ins.sum() else np.nan,
            drift_abs_outside=float(np.median(np.abs(blocks.drift_cph.values[outs]))) if outs.sum() else np.nan)
if ins.sum() >= 3 and outs.sum() >= 3:
    table = [[int(blocks.held.values[ins].sum()), int((~blocks.held.values[ins]).sum())],
             [int(blocks.held.values[outs].sum()), int((~blocks.held.values[outs]).sum())]]
    lock['fisher_p'] = float(fisher_exact(table, alternative='greater')[1])
    ts_in = theilslopes(blocks.drift_cph.values[ins], blocks.dc.values[ins])
    ts_out = theilslopes(blocks.drift_cph.values[outs], blocks.dc.values[outs])
    lock.update(slope_inside=float(ts_in[0]), slope_outside=float(ts_out[0]))
    # the Adler prediction against the measured drift, outside the tongue
    lock['rho_adler_outside'] = float(spearmanr(blocks.adler.values[outs], blocks.drift_cph.values[outs]).correlation)
    err_adler = np.median(np.abs(blocks.drift_cph.values[ok] - blocks.adler.values[ok]))
    err_free = np.median(np.abs(blocks.drift_cph.values[ok] - blocks.dc.values[ok]))
    err_zero = np.median(np.abs(blocks.drift_cph.values[ok]))
    lock.update(err_adler=float(err_adler), err_free=float(err_free), err_held=float(err_zero))

# the Kunze test and the blue shift
okk = np.isfinite(blocks.wake_minus_f.values) & np.isfinite(blocks.half_zeta.values)
shift_feff = blocks.wake_minus_f.values - blocks.half_zeta.values    # wake frequency above f_eff = |f| + zeta/2 (the blue shift)
blocks['blue_shift'] = shift_feff
kunze = {}
if okk.sum() >= 6:
    ts = theilslopes(blocks.wake_minus_f.values[okk], blocks.half_zeta.values[okk])
    kunze = dict(slope=float(ts[0]), lo=float(ts[2]), hi=float(ts[3]), intercept=float(ts[1]),
                 rho=float(spearmanr(blocks.half_zeta.values[okk], blocks.wake_minus_f.values[okk]).correlation),
                 above_f_med=float(np.median(blocks.wake_minus_f.values[okk])),
                 blue_shift_med=float(np.median(shift_feff[okk])),
                 blue_shift_rel=float(np.median(shift_feff[okk]/blocks.f_cph.values[okk])),
                 wake_detuning_med=float(np.median(blocks.drift_cph.values[okk])))
    kunze['ratio_S_over_wake_detuning'] = float(lock['half_S_med']/max(abs(kunze['wake_detuning_med']), 1e-9))

# --------------------------------------------------------------------------
# where the wake gains, by bin of the flow detuning
# --------------------------------------------------------------------------
gain_bins = bins_of(blocks.gain.values, blocks.dc.values)
work_bins = bins_of(blocks.work_sig_it.values, blocks.dc.values)
wind_bins = bins_of(blocks.wind_diff.values, blocks.dc.values)
by_class = {}
for c in ('free', 'trapped'):
    m = blocks['class'].values == c
    if m.sum() >= 3:
        by_class[c] = dict(n=int(m.sum()), gain=float(np.median(blocks.gain.values[m])),
                           work=float(np.median(blocks.work_sig_it.values[m])),
                           wind=float(np.median(blocks.wind_diff.values[m])),
                           held=float(np.mean(blocks.held.values[m])))

# --------------------------------------------------------------------------
# what controls the work and the gain
# --------------------------------------------------------------------------
CANDS = {'flow detuning': blocks.dc.values, 'distance from resonance': np.abs(blocks.dc.values),
         'vorticity': blocks.zof.values, 'tidal strain': blocks.Sr_cph.values,
         'phase efficiency': blocks.cosphi.values, 'wake kinetic energy': blocks.Ew_tide.values,
         'wind work difference': blocks.wind_diff.values, 'latitude': blocks.lat.values,
         'depth': blocks.depth.values}
TARGETS = {'strain work': blocks.work_sig_it.values, 'gain': blocks.gain.values}


def shift_level(x, y, n=N_BOOT, q=95):
    r = np.array([abs(spearmanr(np.roll(x, s), y).correlation) for s in rng.integers(1, x.size - 1, size=n)])
    return float(np.percentile(r, q))


def partial_spearman(x, y, zs):
    Zm = np.column_stack([np.ones(x.size)] + [rankdata(z) for z in zs])
    res = [r - Zm @ np.linalg.lstsq(Zm, r, rcond=None)[0] for r in (rankdata(x), rankdata(y))]
    return float(np.corrcoef(res[0], res[1])[0, 1])


rows_c = []
for tname, y in TARGETS.items():
    for cname, x in CANDS.items():
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            continue
        rec = dict(target=tname, candidate=cname, n=int(m.sum()),
                   rho=float(spearmanr(x[m], y[m]).correlation), level=shift_level(x[m], y[m]))
        rec['partial'] = (np.nan if cname == 'wake kinetic energy'
                          else partial_spearman(x[m], y[m], [blocks.Ew_tide.values[m]]))
        rows_c.append(rec)
controls = pd.DataFrame(rows_c)
blocks.to_csv(OUTDIR + 'fig5_blocks.csv', index=False)
controls.to_csv(OUTDIR + 'fig5_controls.csv', index=False)

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
cand_names = list(CANDS)
ctl = np.full((len(TARGETS), len(cand_names), 3), np.nan)
for i, tname in enumerate(TARGETS):
    for j, cname in enumerate(cand_names):
        r = controls[(controls.target == tname) & (controls.candidate == cname)]
        if len(r):
            ctl[i, j] = [r.rho.iloc[0], r.partial.iloc[0], r.level.iloc[0]]
ds = xr.Dataset(
    data_vars={'drift': (('y', 'x'), drift.astype(np.float32)), 'dc': (('y', 'x'), dc.astype(np.float32)),
               'Sr_cph': (('y', 'x'), Sr.astype(np.float32)), 'zof': (('y', 'x'), zof.astype(np.float32)),
               'lock_bins': (('dc_bin', 'bstat'), lock_bins), 'adler_curve': (('dc_bin',), adler_curve),
               'gain_bins': (('dc_bin', 'bstat'), gain_bins), 'work_bins': (('dc_bin', 'bstat'), work_bins),
               'wind_bins': (('dc_bin', 'bstat'), wind_bins),
               'controls': (('target', 'candidate', 'cstat'), ctl),
               'lon': (('y', 'x'), lon), 'lat': (('y', 'x'), lat), 'h': (('y', 'x'), h),
               'slope': (('y', 'x'), slope), 'foot': (('y', 'x'), foot), 'mask': (('y', 'x'), f2.mask.values)},
    coords=dict(dc_bin=centres, bstat=['median', 'p05', 'p95', 'n'], target=list(TARGETS),
                candidate=cand_names, cstat=['rho', 'partial', 'level']),
    attrs=dict(held_cph=HELD_CPH, lock=str(lock), kunze=str(kunze), by_class=str(by_class), n_blocks=nb,
               work_window=f2.attrs['work_window'], windows=f2.attrs['windows'], t_off=f2.attrs['t_off']))
ds.to_netcdf(OUTDIR + 'fig5_resonance.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['FIGURE 5 REPORT (full_v5, revision 3)  --  the frequency of the wake and the boundary current', '=' * 92,
     f'{nb} footprint blocks of {BLOCK} points, flow detuning f_eff - omega/2 from the wake-window vorticity',
     'of FULL (Figure 2), wake detuning from the phase drift (Figure 4), |S_r| of the internal tide (Figure 4)',
     f'held means |wake detuning| under {HELD_CPH*1e3:.1f} x 1e-3 cph', '',
     '1. THE LOCK (wake detuning against flow detuning, cph)']
L.append(f'   all blocks: Theil and Sen slope {lock["slope_all"]:.2f} [{lock["slope_lo"]:.2f}, {lock["slope_hi"]:.2f}] '
         f'(one for a free wave), Spearman {lock["rho_all"]:.2f}')
L.append(f'   median |S_r|/2 of the blocks {lock["half_S_med"]*1e3:.2f} x 1e-3 cph, the half width of the tongue')
L.append(f'   inside the tongue {lock["n_inside"]} blocks, held {lock["held_inside"]:.2f}, median |drift| '
         f'{lock["drift_abs_inside"]*1e3:.2f} x 1e-3 cph')
L.append(f'   outside the tongue {lock["n_outside"]} blocks, held {lock["held_outside"]:.2f}, median |drift| '
         f'{lock["drift_abs_outside"]*1e3:.2f} x 1e-3 cph')
if 'fisher_p' in lock:
    L.append(f'   Fisher exact p (held more often inside) {lock["fisher_p"]:.3f}, slopes inside {lock["slope_inside"]:.2f} '
             f'and outside {lock["slope_outside"]:.2f}, Spearman of the Adler rate against the drift outside '
             f'{lock["rho_adler_outside"]:.2f}')
    L.append(f'   median absolute error of the wake detuning: free wave {lock["err_free"]*1e3:.2f}, held everywhere '
             f'{lock["err_held"]*1e3:.2f}, Adler {lock["err_adler"]*1e3:.2f} (x 1e-3 cph)')
L.append(f'{"dc":>9s} {"drift":>9s} {"p05":>9s} {"p95":>9s} {"n":>4s} {"adler":>9s}')
for i, c in enumerate(centres):
    if np.isfinite(lock_bins[i, 0]):
        L.append(f'{c*1e3:+9.2f} {lock_bins[i, 0]*1e3:9.2f} {lock_bins[i, 1]*1e3:9.2f} {lock_bins[i, 2]*1e3:9.2f} '
                 f'{int(lock_bins[i, 3]):4d} {adler_curve[i]*1e3:9.2f}')
if kunze:
    L.append(f'   the lock would need |S_r|/2 above the wake detuning, and |S_r|/2 is '
             f'{kunze["ratio_S_over_wake_detuning"]:.2f} of it')
L.append('')
L.append('1b. THE KUNZE TEST (wake frequency minus planetary |f| against zeta/2, cph, blocks)')
if kunze:
    L.append(f'   Theil and Sen slope {kunze["slope"]:.2f} [{kunze["lo"]:.2f}, {kunze["hi"]:.2f}] (one for a free wave), '
             f'intercept {kunze["intercept"]*1e3:.2f} x 1e-3 cph, Spearman {kunze["rho"]:.2f}')
    L.append(f'   blue shift (wake frequency above f_eff = |f| + zeta/2) median {kunze["blue_shift_med"]*1e3:.2f} x 1e-3 cph, '
             f'{100*kunze["blue_shift_rel"]:.1f}% of |f|, wake frequency above planetary |f| median {kunze["above_f_med"]*1e3:.2f} x 1e-3 cph, '
             f'median wake detuning from omega/2 {kunze["wake_detuning_med"]*1e3:.2f} x 1e-3 cph')
L.append('')
L.append('2. WHERE THE WAKE GAINS (medians by bin of flow detuning, J m-2)')
L.append(f'{"dc":>9s} {"gain":>9s} {"p05":>8s} {"p95":>8s} {"work":>9s} {"p05":>8s} {"p95":>8s} {"wind":>9s} {"n":>4s}')
for i, c in enumerate(centres):
    if np.isfinite(gain_bins[i, 0]):
        L.append(f'{c*1e3:+9.2f} {gain_bins[i, 0]:9.0f} {gain_bins[i, 1]:8.0f} {gain_bins[i, 2]:8.0f} '
                 f'{work_bins[i, 0]:9.0f} {work_bins[i, 1]:8.0f} {work_bins[i, 2]:8.0f} {wind_bins[i, 0]:9.0f} '
                 f'{int(gain_bins[i, 3]):4d}')
for c, v in by_class.items():
    L.append(f'   {c:8s} {v["n"]:3d} blocks, median gain {v["gain"]:.0f}, strain work {v["work"]:.0f}, '
             f'wind work difference {v["wind"]:.0f}, held {v["held"]:.2f}')
L.append('')
L.append('3. WHAT CONTROLS THE STRAIN WORK AND THE GAIN (Spearman, partial on the wake kinetic energy, shift level)')
for tname in TARGETS:
    L.append(f'   {tname}')
    sub = controls[controls.target == tname].copy()
    sub['a'] = sub.rho.abs()
    for r in sub.sort_values('a', ascending=False).itertuples():
        flag = '*' if abs(r.rho) > r.level else ' '
        L.append(f'   {flag} {r.candidate:26s} rho {r.rho:+.2f}  partial {r.partial:+.2f}  level {r.level:.2f}')
L.append('')
L.append('READING THE RESULT')
L.append('   question 3 (second half). The vorticity decides where the wake gains if the wake detuning')
L.append('   follows the flow detuning (Kunze slope near one) or if the gain and the strain work separate')
L.append('   by detuning class and the vorticity ranks above the shift level among the controls. A Kunze')
L.append('   slope well under one with a blue shift of a few percent of f says the storm set the frequency')
L.append('   and the vorticity only modulates it, so the flow detuning acts through the phase and not')
L.append('   through the frequency.')
open(OUTDIR + 'fig5_resonance_report.txt', 'w').write('\n'.join(L))
print('\n'.join(L))
print('saved', OUTDIR + 'fig5_resonance.nc')