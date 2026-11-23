"""
fig_08_compute.py  (full_v5, revision 2)
----------------------------------------
Figure 8. The wake as a detuned parametric oscillator.

The reduced model that closes the mechanism. With U = A exp(i omega t/2) the
wake and S = S_r exp(i omega t) the anticlockwise strain of the internal tide
(Figure 4), the strain work of a column is P = (rho0/2)|Z| cos(Phi) with
Z = sum_z S_r A*^2 dz and Phi = pi + arg S_r - 2 arg A. The parametric
subharmonic equation for the amplitude (Young et al. 2008),

    dA/dt = i Delta_w A + (|S_r|/2) exp(i(arg S_r + pi)) A*,

with Delta_w = omega_w - omega/2 the wake detuning, gives for the phase the
Adler equation

    dPhi/dt = -2 Delta_w - |S_r| sin(Phi),

which locks at a gaining phase (cos Phi > 0) only if |S_r| > 2 |Delta_w| and
otherwise drifts at twice the detuning from the phase the wake holds when the
wind stops. Figures 4 and 5 showed the phase drifting at the free rate
everywhere (equal slopes inside and outside the tongue, coherence on the sinc
of a free drift), so the effective coupling of the column-integrated phase is
weaker than the depth-mean strain amplitude. The model of this figure is
therefore the free drift,

    Phi_b(t) = Phi_b(t_off) - 2 Delta_w,b (t - t_off),

for each block from its measured phase at the wind stop with its measured
wake detuning (Figure 5), and the predicted work is formed with the measured
amplitude |Z_b(t)|, so that the test is on the phase alone (the mechanism)
and not on the amplitude. Two variants are run. One detuning for the whole
wake (the footprint median, the frequency the storm set) shows how much of
the skill is the storm's frequency alone. The Adler equation with the full
depth-mean |S_r| is the upper bound on the coupling and is reported.

The efficiency of a block is its work over the largest work it could have
done, W / int (rho0/2)|Z_b| dt, the amplitude-weighted cos Phi over the
window, between -1 and 1. For a uniform drift its envelope is
|int |Z| exp(-2 i Delta_w t) dt| / int |Z| dt, the transform of the
amplitude history at twice the detuning, which is the Lorentzian
r / sqrt(r^2 + Delta_w^2) for an amplitude decaying at 2r. The timing test
shifts the tide by delta, which shifts Phi_0 by omega delta (the phase null
of Figure 3), and since the dependence is through Phi_0 + omega delta only,
the shifts fold onto one tidal period.

Inputs (all from the outputs of Figures 3 to 5, runs in seconds)
  ./fig4_out/fig4_phase.nc     block series of Z, footprint phase
  ./fig4_out/fig4_blocks.csv   drift, strain amplitude, class, coherence
  ./fig3_out/fig3_blocks.csv   measured strain work by block
  ./fig3_out/fig3_strain.nc    footprint strain work series, the phase null through time

Outputs (in OUTDIR)
  fig8_model.nc  fig8_blocks.csv  fig8_model_report.txt
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, theilslopes

FIG3 = './fig3_out/fig3_strain.nc'
FIG3_BLOCKS = './fig3_out/fig3_blocks.csv'
FIG4 = './fig4_out/fig4_phase.nc'
FIG4_BLOCKS = './fig4_out/fig4_blocks.csv'
OUTDIR = './fig8_out/'
os.makedirs(OUTDIR, exist_ok=True)

RAD2CPH = 3600.0/(2*np.pi)
DECAY_DAYS = 4.0                                   # the fit of the decay rate of |Z_b| after its peak (report)
N_BOOT = 2000
SMOOTH_H = 25
N_SHIFT_MODEL = 121                                # shifts of the model timing curve over one tidal period
FOLD_BIN_H = 1.0                                   # bin of the folded measured null
HORIZON = 0.5                                      # the phase agreement at which the prediction has lost half its skill
rng = np.random.default_rng(31)

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
f3 = xr.open_dataset(FIG3)
f4 = xr.open_dataset(FIG4)
b3 = pd.read_csv(FIG3_BLOCKS)
b4 = pd.read_csv(FIG4_BLOCKS)
t = f4.time.values
nt = t.size
dt_s = float((t[1] - t[0])/np.timedelta64(1, 's'))
T_OFF = np.datetime64(f4.attrs['t_off'])
T_STORM0, T_WAKE1 = (np.datetime64(x) for x in eval(f4.attrs['work_window']))
OMEGA = float(f4.attrs['omega_M2'])
RHO0 = float(f4.attrs['rho0'])
T_TIDE_H = 2*np.pi/OMEGA/3600.0
AFTER = (t >= T_OFF) & (t < T_WAKE1)
ka = np.where(AFTER)[0]
tau_s = (t[ka] - T_OFF)/np.timedelta64(1, 's')                                    # s after the wind stop
ids4 = f4.block.values
Zb = f4.Zb_re.values + 1j*f4.Zb_im.values                                          # (time, block), m3 s-3

blocks = b4[['block', 'lon', 'lat', 'n', 'dc', 'free_share', 'class', 'drift_cph', 'Sr', 'coh_with']].merge(
    b3[['block', 'work_sig_it', 'work_it', 'gain']], on='block', how='inner')
blocks = blocks[np.isfinite(blocks.drift_cph) & np.isfinite(blocks.Sr)].reset_index(drop=True)
nb = len(blocks)
col = np.array([int(np.where(ids4 == b)[0][0]) for b in blocks.block.values])
print(f'{nb} blocks, model from {str(T_OFF)[:16]} to {str(T_WAKE1)[:10]}', flush=True)

A = 0.5*RHO0*np.abs(Zb[ka][:, col]).T                                              # (block, tafter), W m-2, the largest work
phi_meas = np.angle(-Zb[ka][:, col]).T                                             # (block, tafter), Phi = arg(-Z)
phi0 = phi_meas[:, 0]
delta = blocks.drift_cph.values/RAD2CPH                                            # rad s-1
delta_med = float(np.median(delta))
S = blocks.Sr.values                                                               # s-1
area_w = blocks.n.values.astype(float)
area_w /= area_w.sum()


def smooth(x, n=SMOOTH_H):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


def accumulate(P, axis=-1):
    return np.sum(P, axis=axis)*dt_s


def adler(p0, d, s, tt):
    """Phi(t) from dPhi/dt = -2 Delta_w - |S_r| sin Phi (fourth-order step on the hourly axis)."""
    phi = np.empty(tt.size)
    phi[0] = p0
    for i in range(1, tt.size):
        h = tt[i] - tt[i - 1]
        p = phi[i - 1]

        def f(q):
            return -2.0*d - s*np.sin(q)

        k1 = f(p)
        k2 = f(p + 0.5*h*k1)
        k3 = f(p + 0.5*h*k2)
        k4 = f(p + h*k3)
        phi[i] = p + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return phi


# --------------------------------------------------------------------------
# the models block by block
# --------------------------------------------------------------------------
phi_mod = {'drift': phi0[:, None] - 2.0*delta[:, None]*tau_s[None, :],
           'one': phi0[:, None] - 2.0*delta_med*tau_s[None, :],
           'frozen': np.repeat(phi0[:, None], tau_s.size, axis=1),
           'adler': np.array([adler(phi0[j], delta[j], S[j], tau_s) for j in range(nb)])}
P_meas = A*np.cos(phi_meas)
P_mod = {k: A*np.cos(v) for k, v in phi_mod.items()}
W_meas = accumulate(P_meas)
W_mod = {k: accumulate(v) for k, v in P_mod.items()}
A_int = accumulate(A)                                                              # J m-2, the largest work after the stop
eff_meas = W_meas/np.maximum(A_int, 1e-30)
eff_mod = {k: v/np.maximum(A_int, 1e-30) for k, v in W_mod.items()}
# the envelope of the efficiency for a uniform drift, per block from its own amplitude history
env_blk = np.abs(accumulate(A*np.exp(-2j*delta[:, None]*tau_s[None, :])))/np.maximum(A_int, 1e-30)
# the decay rate of |Z_b| over DECAY_DAYS after its peak (report, and the Lorentzian of the caption)
r_b = np.full(nb, np.nan)
for j in range(nb):
    a = smooth(A[j])
    ip = int(np.argmax(a))
    sel = (np.arange(a.size) >= ip) & (tau_s <= tau_s[ip] + DECAY_DAYS*86400.0)
    if sel.sum() >= 24 and np.all(a[sel] > 0):
        r_b[j] = -np.polyfit(tau_s[sel], np.log(a[sel]), 1)[0]/2.0
blocks['phi0'] = phi0
blocks['phi0_deg'] = np.degrees(phi0)
blocks['delta_rad_s'] = delta
blocks['S_rad_s'] = S
blocks['adler'] = S/np.maximum(2*np.abs(delta), 1e-30)
blocks['r_per_day'] = r_b*86400.0
blocks['A_int'] = A_int
blocks['W_meas'] = W_meas
for k in W_mod:
    blocks[f'W_{k}'] = W_mod[k]
    blocks[f'eff_{k}'] = eff_mod[k]
    blocks[f'rms_phase_err_{k}'] = np.sqrt(np.mean(np.angle(np.exp(1j*(phi_mod[k] - phi_meas)))**2, axis=1))
blocks['eff_meas'] = eff_meas
blocks['env'] = env_blk
blocks['inside'] = np.abs(eff_meas) <= env_blk
blocks.to_csv(OUTDIR + 'fig8_blocks.csv', index=False)

# --------------------------------------------------------------------------
# skill over the blocks and over the footprint
# --------------------------------------------------------------------------


def shift_level(x, y, n=N_BOOT, q=95):
    r = np.array([abs(spearmanr(np.roll(x, s), y).correlation) for s in rng.integers(1, x.size - 1, size=n)])
    return float(np.percentile(r, q))


stats = {}
for k in ('drift', 'one', 'adler', 'frozen'):
    x, y = W_mod[k], W_meas
    ts = theilslopes(y, x)
    sa = np.mean(np.sign(x) == np.sign(y))
    stats[k] = dict(rho=float(spearmanr(x, y).correlation), level=shift_level(x, y), slope=float(ts[0]),
                    lo=float(ts[2]), hi=float(ts[3]), sign_agreement=float(sa),
                    sign_level=float(np.percentile([np.mean(np.sign(np.roll(x, s)) == np.sign(y))
                                                    for s in rng.integers(1, x.size - 1, size=N_BOOT)], 95)),
                    sum_pred=float(np.sum(x)), sum_meas=float(np.sum(y)),
                    phase_err_rad=float(np.median(blocks[f'rms_phase_err_{k}'])),
                    rho_eff=float(spearmanr(eff_mod[k], eff_meas).correlation))
stats['phase_err_no_skill_rad'] = float(np.pi/np.sqrt(3))
locked = blocks.adler.values >= 1
stats['n_locked'] = int(locked.sum())
stats['adler_max'] = float(np.max(blocks.adler))
stats['coh_locked'] = float(np.nanmedian(blocks.coh_with.values[locked])) if locked.any() else np.nan
stats['coh_unlocked'] = float(np.nanmedian(blocks.coh_with.values[~locked]))
stats['delta_locked_cph'] = float(np.max(np.abs(blocks.drift_cph.values[locked]))) if locked.any() else np.nan
stats['r_median_per_day'] = float(np.nanmedian(blocks.r_per_day))
stats['delta_median_cph'] = float(np.median(blocks.drift_cph))
stats['inside_envelope'] = float(np.mean(blocks.inside))
for c in ('free', 'trapped'):
    m = blocks['class'].values == c
    if m.sum() >= 3:
        stats[f'r_{c}_per_day'] = float(np.nanmedian(blocks.r_per_day.values[m]))
        stats[f'delta_{c}_cph'] = float(np.median(blocks.drift_cph.values[m]))
        stats[f'W_meas_{c}'] = float(np.sum(W_meas[m]))
        stats[f'W_drift_{c}'] = float(np.sum(W_mod['drift'][m]))
        stats[f'inside_{c}'] = float(np.mean(blocks.inside.values[m]))
        stats[f'eff_meas_{c}'] = float(np.sum(W_meas[m])/np.sum(A_int[m]))

# footprint series (area-weighted over the blocks)
foot_meas = (P_meas*area_w[:, None]).sum(axis=0)
foot_mod = {k: (v*area_w[:, None]).sum(axis=0) for k, v in P_mod.items()}
ser3 = f3.ser_sig_it.values[ka]
stats['footprint'] = dict(W_meas=float(accumulate(foot_meas)), W_fig3=float(accumulate(ser3)),
                          corr_blocks_vs_fig3=float(np.corrcoef(foot_meas, ser3)[0, 1]),
                          **{f'W_{k}': float(accumulate(v)) for k, v in foot_mod.items()},
                          **{f'corr_{k}': float(np.corrcoef(smooth(foot_meas), smooth(v))[0, 1]) for k, v in foot_mod.items()})

# the phase agreement through time, amplitude-weighted resultant of exp(i(Phi_meas - Phi_model))
agree = {}
for k, v in phi_mod.items():
    agree[k] = np.abs((A*np.exp(1j*(phi_meas - v))).sum(axis=0))/np.maximum(A.sum(axis=0), 1e-30)
agree_level = np.sqrt((A**2).sum(axis=0))/np.maximum(A.sum(axis=0), 1e-30)       # random phases with these weights
horizon = {}
for k, v in agree.items():
    below = np.where(v < HORIZON)[0]
    horizon[k] = float(tau_s[below[0]]/86400.0) if below.size else np.nan

# --------------------------------------------------------------------------
# the timing test, folded onto one tidal period
# --------------------------------------------------------------------------
shifts_h = f3['shift'].values
null_meas = np.nansum(f3.null_with_t.values[:, ka], axis=1)*dt_s                    # J m-2 after the wind stop, Figure 3
fold_h = ((shifts_h + 0.5*T_TIDE_H) % T_TIDE_H) - 0.5*T_TIDE_H
shift_model_h = np.linspace(-0.5*T_TIDE_H, 0.5*T_TIDE_H, N_SHIFT_MODEL)
null_mod = np.array([(accumulate(A*np.cos(phi_mod['drift'] + OMEGA*sh*3600.0))*area_w).sum() for sh in shift_model_h])
null_mod_at = np.interp(fold_h, shift_model_h, null_mod)
fold_edges = np.arange(-0.5*T_TIDE_H, 0.5*T_TIDE_H + 1e-9, FOLD_BIN_H)
fold_mid = 0.5*(fold_edges[1:] + fold_edges[:-1])
fold_mean = np.array([np.mean(null_meas[(fold_h >= a) & (fold_h < b)]) if np.any((fold_h >= a) & (fold_h < b)) else np.nan
                      for a, b in zip(fold_edges[:-1], fold_edges[1:])])
# the scatter of the fold, the measured null against its own value one and two periods away
i0 = int(np.argmin(np.abs(shifts_h)))
timing = dict(corr=float(np.corrcoef(null_meas, null_mod_at)[0, 1]),
              amp_meas=float(0.5*(np.nanmax(fold_mean) - np.nanmin(fold_mean))),
              amp_mod=float(0.5*(null_mod.max() - null_mod.min())),
              actual_meas=float(null_meas[i0]), actual_mod=float(null_mod_at[i0]),
              rank_actual_meas=float(np.mean(null_meas <= null_meas[i0])),
              best_shift_mod_h=float(shift_model_h[np.argmax(null_mod)]),
              loss_shift_mod_h=float(shift_model_h[np.argmin(null_mod)]),
              zero_shift_mod_h=float(shift_model_h[np.argmin(np.abs(null_mod[:N_SHIFT_MODEL//2 + 1]))]),
              fold_rms=float(np.sqrt(np.nanmean((null_meas - np.interp(fold_h, fold_mid, fold_mean, period=T_TIDE_H))**2))),
              period_h=float(T_TIDE_H))

# --------------------------------------------------------------------------
# the analytic map and the envelopes
# --------------------------------------------------------------------------
phi_grid = np.radians(np.linspace(-180, 180, 181))
delta_grid = np.linspace(-6e-3, 6e-3, 121)/RAD2CPH
r_ref = stats['r_median_per_day']/86400.0
W_map = (r_ref*np.cos(phi_grid)[None, :] + delta_grid[:, None]*np.sin(phi_grid)[None, :])/(2.0*(r_ref**2 + delta_grid[:, None]**2))
W_map_norm = W_map*2.0*r_ref                                                        # one at the best phase with no detuning
env_delta = np.linspace(-6e-3, 6e-3, 241)/RAD2CPH
envelope = {}
eff_curve = {}                                                                      # efficiency against Phi_0 at the class detuning
transform = {}
for c in ('free', 'trapped'):
    m = blocks['class'].values == c
    Ac = (A[m]*area_w[m][:, None]).sum(axis=0)/area_w[m].sum() if m.any() else np.full(tau_s.size, np.nan)
    envelope[c] = np.array([np.abs(np.sum(Ac*np.exp(-2j*d*tau_s)))/max(np.sum(Ac), 1e-30) for d in env_delta])
    d_c = float(np.median(delta[m])) if m.any() else np.nan
    T_c = np.sum(Ac*np.exp(-2j*d_c*tau_s))/max(np.sum(Ac), 1e-30) if m.any() else np.nan + 0j
    transform[c] = dict(delta_cph=d_c*RAD2CPH if np.isfinite(d_c) else np.nan, env=float(np.abs(T_c)),
                        psi_deg=float(np.degrees(np.angle(T_c))))
    eff_curve[c] = np.real(T_c*np.exp(1j*phi_grid))                              # |T| cos(Phi_0 + psi)
r_grid = np.linspace(0.0, 1.2, 121)/86400.0                                         # s-1, decay rate of the amplitude over 2
env_r = {c: r_grid/np.sqrt(r_grid**2 + (transform[c]['delta_cph']/RAD2CPH)**2) if np.isfinite(transform[c]['delta_cph'])
         else np.full(r_grid.size, np.nan) for c in transform}                     # the envelope of an exponential decay
okr = np.isfinite(blocks.r_per_day.values)
stats['rho_abs_eff_rate'] = float(spearmanr(blocks.r_per_day.values[okr], np.abs(eff_meas[okr])).correlation)
stats['rho_abs_eff_rate_level'] = shift_level(blocks.r_per_day.values[okr], np.abs(eff_meas[okr]))
stats['inside_lorentz'] = float(np.mean(np.abs(eff_meas[okr]) <= blocks.r_per_day.values[okr]/86400.0
                                        / np.sqrt((blocks.r_per_day.values[okr]/86400.0)**2 + delta[okr]**2)))

# --------------------------------------------------------------------------
# save and report
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars={'P_meas': (('block', 'tafter'), P_meas), 'A': (('block', 'tafter'), A),
               **{f'P_{k}': (('block', 'tafter'), v) for k, v in P_mod.items()},
               **{f'phi_{k}': (('block', 'tafter'), v) for k, v in phi_mod.items()},
               'phi_meas': (('block', 'tafter'), phi_meas),
               'foot_meas': (('tafter',), foot_meas), 'foot_fig3': (('tafter',), ser3),
               **{f'foot_{k}': (('tafter',), v) for k, v in foot_mod.items()},
               **{f'agree_{k}': (('tafter',), v) for k, v in agree.items()},
               'agree_level': (('tafter',), agree_level),
               'null_meas': (('shift',), null_meas), 'fold_h': (('shift',), fold_h),
               'fold_mean': (('fold_bin',), fold_mean), 'null_mod': (('shift_model',), null_mod),
               'W_map': (('delta_grid', 'phi_grid'), W_map_norm),
               **{f'envelope_{c}': (('env_delta',), envelope[c]) for c in envelope},
               **{f'eff_curve_{c}': (('phi_grid',), eff_curve[c]) for c in eff_curve},
               **{f'env_r_{c}': (('r_grid',), env_r[c]) for c in env_r}},
    coords=dict(block=blocks.block.values, tafter=t[ka], shift=shifts_h, fold_bin=fold_mid, shift_model=shift_model_h,
                delta_grid=delta_grid*RAD2CPH*1e3, phi_grid=np.degrees(phi_grid), env_delta=env_delta*RAD2CPH*1e3,
                r_grid=r_grid*86400.0),
    attrs=dict(t_off=str(T_OFF)[:16], work_window=str((str(T_STORM0)[:10], str(T_WAKE1)[:10])), omega_M2=OMEGA, rho0=RHO0,
               period_h=T_TIDE_H, decay_days=DECAY_DAYS, smooth_h=SMOOTH_H, horizon_level=HORIZON,
               r_ref_per_day=stats['r_median_per_day'], delta_median_cph=stats['delta_median_cph'],
               stats=str(stats), timing=str(timing), horizon=str(horizon), transform=str(transform), n_blocks=nb))
ds.to_netcdf(OUTDIR + 'fig8_model.nc')

fp = stats['footprint']
NAMES = {'drift': 'free drift (the model)', 'one': 'one frequency for the wake', 'adler': 'Adler with the full coupling',
         'frozen': 'frozen phase (no drift)'}
L = ['FIGURE 8 REPORT (full_v5, revision 2)  --  the wake as a detuned parametric oscillator',
     '=' * 88,
     f'{nb} blocks, the model runs from the wind stop {str(T_OFF)[:16]} to {str(T_WAKE1)[:10]}',
     'the phase of each block drifts from its measured value at the wind stop at twice the measured wake detuning',
     '(Figure 5), the work is formed with the measured |Z_b(t)| (Figure 4), so the test is on the phase alone',
     f'blocks checked against the Figure 3 series over the same hours: correlation {fp["corr_blocks_vs_fig3"]:.3f}, '
     f'accumulated {fp["W_meas"]:.1f} against {fp["W_fig3"]:.1f} J m-2',
     '',
     '1. THE COUPLING',
     f'   Adler parameter |S_r| / (2 |Delta_w|) above one in {stats["n_locked"]} of {nb} blocks (largest {stats["adler_max"]:.1f}, '
     f'all with |Delta_w| under {stats["delta_locked_cph"]*1e3:.2f} x 1e-3 cph), whose phase coherence (Figure 4) is '
     f'{stats["coh_locked"]:.2f} against {stats["coh_unlocked"]:.2f} for the others, so they do not lock',
     f'   median wake detuning {stats["delta_median_cph"]*1e3:.2f} x 1e-3 cph, median decay rate of |Z| after its peak '
     f'{stats["r_median_per_day"]:.2f} d-1 (free {stats.get("r_free_per_day", np.nan):.2f}, trapped {stats.get("r_trapped_per_day", np.nan):.2f})',
     '',
     '2. THE SKILL OVER THE BLOCKS (work after the wind stop, J m-2)']
for k in ('drift', 'one', 'adler', 'frozen'):
    st = stats[k]
    L.append(f'   {NAMES[k]:30s} Spearman {st["rho"]:.2f} (level {st["level"]:.2f}), slope of measured on predicted '
             f'{st["slope"]:.2f} [{st["lo"]:.2f}, {st["hi"]:.2f}], sign agreement {st["sign_agreement"]:.2f} (level {st["sign_level"]:.2f}), '
             f'sum {st["sum_pred"]:.0f} against {st["sum_meas"]:.0f}, rms phase error {st["phase_err_rad"]:.2f} rad, '
             f'efficiency rho {st["rho_eff"]:.2f}')
L += [f'   no skill in the phase {stats["phase_err_no_skill_rad"]:.2f} rad',
      f'   by class, measured against the model: free {stats.get("W_meas_free", np.nan):.0f} against {stats.get("W_drift_free", np.nan):.0f}, '
      f'trapped {stats.get("W_meas_trapped", np.nan):.0f} against {stats.get("W_drift_trapped", np.nan):.0f}',
      f'   efficiency of the class (work over the largest work): free {stats.get("eff_meas_free", np.nan):.2f}, '
      f'trapped {stats.get("eff_meas_trapped", np.nan):.2f}',
      f'   blocks inside the envelope of a uniform drift: all {stats["inside_envelope"]:.2f}, free {stats.get("inside_free", np.nan):.2f}, '
      f'trapped {stats.get("inside_trapped", np.nan):.2f}',
      '   the detuning shifts the gaining phase: at the class detuning the efficiency is |T| cos(Phi_0 + psi), largest at Phi_0 = -psi, with '
      + ', '.join(f'{c} |T| {transform[c]["env"]:.2f} psi {transform[c]["psi_deg"]:+.0f} deg at {transform[c]["delta_cph"]*1e3:.2f} x 1e-3 cph'
                  for c in transform),
      f'   the size against the decay rate: |efficiency| against the decay rate of the block, Spearman {stats["rho_abs_eff_rate"]:.2f} '
      f'(level {stats["rho_abs_eff_rate_level"]:.2f}), blocks inside the Lorentzian r / sqrt(r^2 + Delta_w^2) {stats["inside_lorentz"]:.2f}',
      '',
      '3. THE FOOTPRINT AFTER THE WIND STOP (J m-2, correlation of the 25 h smoothed series)',
      f'   measured {fp["W_meas"]:.1f}' + ''.join(f', {NAMES[k]} {fp[f"W_{k}"]:.1f} ({fp[f"corr_{k}"]:.2f})'
                                               for k in ('drift', 'one', 'adler', 'frozen')),
      '   phase agreement (amplitude-weighted resultant of the phase error over the blocks) falls to '
      f'{HORIZON:.1f} after: ' + ', '.join(f'{NAMES[k]} {horizon[k]:.1f} d' for k in horizon),
      '',
      '4. THE TIMING (the tide shifted by delta, work after the wind stop, J m-2, folded onto one tidal period)',
      f'   correlation of the measured null with the model over the {shifts_h.size} shifts {timing["corr"]:.2f}, amplitude measured '
      f'{timing["amp_meas"]:.0f} against model {timing["amp_mod"]:.0f}, rms scatter of the fold {timing["fold_rms"]:.0f}',
      f'   the actual timing: measured {timing["actual_meas"]:.0f} (rank {timing["rank_actual_meas"]:.2f} among the shifts), '
      f'model {timing["actual_mod"]:.0f}. The model puts the largest gain at {timing["best_shift_mod_h"]:+.1f} h, the sign change at '
      f'{timing["zero_shift_mod_h"]:+.1f} h and the largest loss at {timing["loss_shift_mod_h"]:+.1f} h (period {timing["period_h"]:.2f} h)',
      '',
      'READING THE RESULT',
      '   The mechanism is closed if the phase at the wind stop drifting at the measured wake detuning',
      '   predicts the work of the blocks (slope near one, sign agreement above its level) and the',
      '   footprint series, if one frequency for the whole wake keeps most of that skill (the storm set',
      '   the frequency), if the measured phase null folds onto the model over one tidal period (the',
      '   sign is the timing of the storm against the tide), and if the blocks fall inside the envelope',
      '   of a uniform drift (the size is the detuning against the amplitude history). The Adler',
      '   version with the full depth-mean strain overpredicts where it locks, which is the data',
      '   rejecting a coupling that strong.']
open(OUTDIR + 'fig8_model_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('saved', OUTDIR + 'fig8_model.nc')