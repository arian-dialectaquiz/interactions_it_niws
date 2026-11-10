"""
fig_01_compute.py  (full_v5)
----------------------------
Figure 1. The margin, the storm, the tide and the distance from resonance, in
space and in time, for the four experiments.

What changed from the version of new_full
-----------------------------------------
1. The balanced vorticity is now geostrophic and time resolved. It is formed
   from the sea surface elevation, low passed at LP_HOURS and coarse-grained
   with the same Gaussian as before, at every STEP_H hours of all four
   experiments. The earlier version held three eight-day means of the model
   vorticity for the joint run and two for the tide run, so the regions where
   the near-inertial band reaches half the tidal frequency could be counted in
   three windows only, and the experiments could not be compared hour by hour.
   The geostrophic estimate is checked against the earlier fields before it is
   trusted, and the check is written to the report.
2. The balanced flow experiment enters, so the detuning of every experiment is
   known and the agreement between experiments is measured through the record.
   That agreement is the floor under every difference between experiments.
3. The band geometry of the old Figure 2 moves here as one panel, the rotary
   spectrum of the joint experiment against latitude and frequency, from the
   several hundred slope columns of the profile caches.
4. The wind stress is carried along the whole slope against time, and the test
   that the stress is the same field in the two wind forced experiments is made
   explicitly, since every later statement about the source of the energy rests
   on it.

Inputs
  ./surface/surface_<RUN>.nc                 from extract_surface.py
  ../new_full/profiles/profiles_<RUN>.nc     from extract_profiles.py
  ../new_full/fig1_out/fig1_data.nc          optional, for the check of item 1
  ../new_full/fig1_out/fig1_track.csv        optional, the best track
  ../new_full/fig1_out/station_labels.csv    optional, the virtual moorings

Outputs (in OUTDIR)
  fig1_v5.nc        the new fields fig_01_plot.py draws in panels (c) and (d), the
                    outlines of the two model grids and the series of the report
  fig1_track.csv    the best track when one is found
  fig1_stations.csv the virtual moorings
  fig1_report.txt   the numbers for the text and the caption
"""

import os
import glob
import warnings
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, detrend, get_window
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', message='Mean of empty slice')

# --------------------------------------------------------------------------
# paths and settings
# --------------------------------------------------------------------------
RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
SURF = './surface/surface_{run}.nc'
OLD = '../new_full/'
PROF = OLD + 'profiles/profiles_{run}.nc'
OLD_FIG1 = OLD + 'fig1_out/fig1_data.nc'
OLD_TRACK = OLD + 'fig1_out/fig1_track.csv'
OLD_LABELS = OLD + 'fig1_out/station_labels.csv'
OUTDIR = './fig1_out/'
os.makedirs(OUTDIR, exist_ok=True)
# the two model grids, for the outlines of panel (a). The parent path is searched
# for when the name below is not found, and the outline is left out with a
# message when nothing turns up, so set PARENT_GRID to the 3 km grid file.
GRID_DIR = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/'
CHILD_GRID = GRID_DIR + '1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc'
PARENT_GRID = GRID_DIR + '3km/inputs/deproas_spongebob_grd_cropped_3km.nc'

# One set of windows serves every figure. The storm window ends and the wake
# window opens at the hour the wind stops over the southern slope, which this
# script measures further down (28 March 05:00 in the first run) and checks
# against T_OFF in the report. Change T_OFF here if the report says they differ.
T_OFF = '2004-03-28T05:00'
EPOCHS = {'T1': ('2004-03-16', '2004-03-24'),      # before the storm
          'T2': ('2004-03-24', T_OFF),             # the storm
          'T3': (T_OFF, '2004-04-07')}             # the wake
LANDFALL = np.datetime64('2004-03-28T00:00')
BASE_MOORINGS = {'A': [380, 1233], 'B': [276, 896], 'C': [333, 557], 'D': [453, 323]}
REGION_ETA = {'South': (0, 420), 'Central': (421, 900), 'North': (901, 10**6)}

G = 9.81
RHO0 = 1025.0
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
OMEGA_E = 7.2921159e-5
RAD2CPH = 3600.0/(2*np.pi)
LAT_CRIT = -np.degrees(np.arcsin(0.5*OMEGA_M2/(2*OMEGA_E)))    # close to -28.8

LP_HOURS = 72.0            # low pass on the elevation, removes tides and the wake
STEP_H = 12                # hours between the balanced fields that are kept
L_MESO = 25.0e3            # Gaussian coarse-graining, sigma = L_MESO/2 as in new_full
ROSSBY_LIMIT = 0.25        # |zeta/2f| above which f_eff = f + zeta/2 is strained
SLOPE_BAND = (200.0, 2000.0)
EDGE_BUFFER = 60           # cells of the 1 km grid kept clear of every edge
LAT_BIN_W = 0.25
LAT_BINS = np.arange(-32.0, -21.9, LAT_BIN_W)
MIN_BIN = 15               # slope points a latitude bin must hold
WIND_OFF_FRAC = 0.15       # the wind is taken to have stopped below this share of its peak
SD_BAND_H = (11.0, 13.0)   # semidiurnal band for the elevation envelope
ENV_SMOOTH_H = 72.0
ENV_STRIDE = 5             # every fifth slope point enters the envelope

SPINUP_DAYS = 10.0         # trimmed before the spectra, as in new_full
SPEC_LAT_W = 0.5
SPEC_BINS = np.arange(-32.0, -21.9, SPEC_LAT_W)
SPEC_MIN = 3               # columns a latitude bin of the spectrum must hold

PAIRS = [('FULL', 'WIND'), ('FULL', 'TIDE'), ('FULL', 'BACK'),
         ('TIDE', 'BACK'), ('WIND', 'BACK')]

# --------------------------------------------------------------------------
# the surface caches on the record they share
# --------------------------------------------------------------------------
sets = {}
for run in RUNS:
    p = SURF.format(run=run)
    if not os.path.exists(p):
        print(f'{p} is missing, run extract_surface.py {run}')
        continue
    d = xr.open_dataset(p)
    sets[run] = d.assign_coords(ocean_time=d.ocean_time.dt.round('s'))
RUNS = [r for r in RUNS if r in sets]
if 'FULL' not in sets:
    raise SystemExit('the joint experiment is needed, run extract_surface.py FULL')
t = sets['FULL'].ocean_time.values
for d in sets.values():
    t = np.intersect1d(t, d.ocean_time.values)
sets = {k: d.sel(ocean_time=t) for k, d in sets.items()}
nt = t.size
dt_h = float(np.mean(np.diff(t)/np.timedelta64(1, 'h')))
nyq = 0.5/dt_h
print(f'{len(RUNS)} experiments, {nt} records, {str(t[0])[:16]} to {str(t[-1])[:16]}')

g0 = sets['FULL']
lon, lat, h, f = g0.lon.values, g0.lat.values, g0.h.values, g0.f.values
dx, dy = g0.dx.values, g0.dy.values
ocean = g0.ocean.values.astype(bool)
eta_c, xi_c = g0.eta_c.values, g0.xi_c.values
sub = int(g0.attrs['sub'])
ny, nx = h.shape

buf = int(np.ceil(EDGE_BUFFER/sub))
inside = np.zeros((ny, nx), bool)
inside[buf:-buf, buf:-buf] = True
slope = (h > SLOPE_BAND[0]) & (h < SLOPE_BAND[1]) & ocean & inside
sector = np.full((ny, nx), -1)
for code, (name, (lo, hi)) in enumerate(REGION_ETA.items()):
    sector[(eta_c >= lo) & (eta_c <= hi), :] = code
SECTORS = list(REGION_ETA)
foot = slope & (sector == SECTORS.index('South'))
print(f'slope band {slope.sum()} blocks, '
      + ', '.join(f'{s} {(slope & (sector == k)).sum()}' for k, s in enumerate(SECTORS)))

lat_mid = 0.5*(LAT_BINS[1:] + LAT_BINS[:-1])
nb = lat_mid.size
ibin = np.digitize(lat, LAT_BINS) - 1
ibin_s = ibin[slope]
ok_bin = (ibin_s >= 0) & (ibin_s < nb)
count = np.bincount(ibin_s[ok_bin], minlength=nb).astype(float)


def bin_mean(values):
    """Mean over the slope points of each latitude bin, values on slope points."""
    good = ok_bin & np.isfinite(values)
    n = np.bincount(ibin_s[good], minlength=nb).astype(float)
    s = np.bincount(ibin_s[good], weights=values[good], minlength=nb)
    return np.where(n >= MIN_BIN, s/np.maximum(n, 1.0), np.nan)


# --------------------------------------------------------------------------
# the balanced vorticity from the elevation
#
# The elevation is low passed in time, which removes the barotropic and the
# internal tide and the near-inertial wake, and the geostrophic vorticity
# (g/f) del^2 eta is coarse-grained with a Gaussian that ignores land. The ends
# of the record are blanked over half the filter period, where the forward and
# backward filter has no data to lean on.
# --------------------------------------------------------------------------
sos_lp = butter(4, (1.0/LP_HOURS)/nyq, btype='low', output='sos')
kout = np.arange(0, nt, int(round(STEP_H/dt_h)))
tb = t[kout]
edge = (tb < t[0] + np.timedelta64(int(LP_HOURS/2), 'h')) | \
       (tb > t[-1] - np.timedelta64(int(LP_HOURS/2), 'h'))
sigma = 0.5*L_MESO/np.array([np.nanmean(dy), np.nanmean(dx)])
print(f'low pass {LP_HOURS:.0f} h, kept every {STEP_H} h ({tb.size} fields), '
      f'Gaussian sigma {sigma[0]:.2f} x {sigma[1]:.2f} blocks')


def laplacian(a):
    out = np.full_like(a, np.nan)
    out[1:-1, 1:-1] = ((a[1:-1, 2:] - 2*a[1:-1, 1:-1] + a[1:-1, :-2])/dx[1:-1, 1:-1]**2
                       + (a[2:, 1:-1] - 2*a[1:-1, 1:-1] + a[:-2, 1:-1])/dy[1:-1, 1:-1]**2)
    return out


def smooth_masked(a):
    good = np.isfinite(a)
    num = gaussian_filter(np.where(good, a, 0.0), sigma=sigma, mode='nearest')
    den = gaussian_filter(good.astype(float), sigma=sigma, mode='nearest')
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.3)
    return np.where(ocean, out, np.nan)


zof = {r: np.full((tb.size, ny, nx), np.nan, np.float32) for r in RUNS}
zof_raw_absmax = {}
for run in RUNS:
    z = sets[run].zeta.values.astype(np.float64)
    z = np.where(np.isfinite(z), z, 0.0)
    lp = np.empty((tb.size, ny, nx))
    for j0 in range(0, ny, 64):
        lp[:, j0:j0+64] = sosfiltfilt(sos_lp, z[:, j0:j0+64], axis=0)[kout]
    del z
    lp[:, ~ocean] = np.nan
    amax = 0.0
    for n in range(tb.size):
        raw = G/f*laplacian(lp[n])
        amax = max(amax, float(np.nanmax(np.abs((raw/f)[slope]))))
        zof[run][n] = smooth_masked(raw)/f
    zof[run][edge] = np.nan
    zof_raw_absmax[run] = amax
    print(f'{run}: balanced vorticity done, slope |zeta/f| before coarse-graining '
          f'up to {amax:.2f}, after {np.nanmax(np.abs(zof[run][:, slope])):.2f}')


def detuning(zf):
    """Delta_c = |f + zeta/2| - omega_M2/2 from zeta/f."""
    return np.abs(f*(1.0 + 0.5*zf)) - 0.5*OMEGA_M2


dc_plan = np.abs(f) - 0.5*OMEGA_M2
free_plan = bin_mean((dc_plan[slope] < 0).astype(float))

# --------------------------------------------------------------------------
# the detuning through the record, experiment by experiment
# --------------------------------------------------------------------------
pole, equa = slope & (lat < LAT_CRIT), slope & (lat > LAT_CRIT)
free_hov = np.full((len(RUNS), tb.size, nb), np.nan)
free_pole = np.full((len(RUNS), tb.size), np.nan)
forb_equa = np.full((len(RUNS), tb.size), np.nan)
strain_frac = np.full((len(RUNS), tb.size), np.nan)
for i, run in enumerate(RUNS):
    for n in range(tb.size):
        if edge[n]:
            continue
        dcn = detuning(zof[run][n])
        free_hov[i, n] = bin_mean(np.where(np.isfinite(dcn[slope]),
                                           (dcn[slope] < 0).astype(float), np.nan))
        free_pole[i, n] = np.nanmean(dcn[pole] < 0)
        forb_equa[i, n] = np.nanmean(dcn[equa] > 0)
        strain_frac[i, n] = np.nanmean(np.abs(0.5*zof[run][n][slope]) > ROSSBY_LIMIT)

eps = list(EPOCHS)
zof_ep = np.full((len(RUNS), len(eps), ny, nx), np.nan, np.float32)
dc_ep = np.full((len(RUNS), len(eps), ny, nx), np.nan, np.float32)
free_prof = np.full((len(RUNS), len(eps), nb), np.nan)
ep_stats = {}
for i, run in enumerate(RUNS):
    for j, (ep, (t0, t1)) in enumerate(EPOCHS.items()):
        sel = (tb >= np.datetime64(t0)) & (tb < np.datetime64(t1)) & ~edge
        zm = np.nanmean(zof[run][sel], axis=0)       # the mean vorticity, then the detuning
        zof_ep[i, j] = zm
        dc_ep[i, j] = detuning(zm)
        d = dc_ep[i, j]
        free_prof[i, j] = bin_mean(np.where(np.isfinite(d[slope]),
                                            (d[slope] < 0).astype(float), np.nan))
        ep_stats[(run, ep)] = dict(
            free_pole=float(np.nanmean(d[pole] < 0)), forb_equa=float(np.nanmean(d[equa] > 0)),
            strained=float(np.nanmean(np.abs(0.5*zm[slope]) > ROSSBY_LIMIT)),
            p05=float(np.nanpercentile(zm[slope], 5)), p95=float(np.nanpercentile(zm[slope], 95)),
            nrec=int(sel.sum()))

# --------------------------------------------------------------------------
# the agreement between experiments, which is the floor under every difference
# formed between them. Two numbers at every kept hour over the slope band, the
# pattern correlation of the balanced vorticity and the share of the slope on
# which two experiments put the same sign on the detuning.
# --------------------------------------------------------------------------
pairs = [p for p in PAIRS if p[0] in RUNS and p[1] in RUNS]
agree_r = np.full((len(pairs), tb.size), np.nan)
agree_s = np.full((len(pairs), tb.size), np.nan)
agree_s_foot = np.full((len(pairs), tb.size), np.nan)
for k, (a, b) in enumerate(pairs):
    for n in range(tb.size):
        if edge[n]:
            continue
        za, zb = zof[a][n], zof[b][n]
        m = slope & np.isfinite(za) & np.isfinite(zb)
        agree_r[k, n] = np.corrcoef(za[m], zb[m])[0, 1]
        same = np.sign(detuning(za)) == np.sign(detuning(zb))
        agree_s[k, n] = same[m].mean()
        agree_s_foot[k, n] = same[m & foot].mean()

# --------------------------------------------------------------------------
# check of the geostrophic vorticity against the model vorticity of new_full
# --------------------------------------------------------------------------
check = []
if os.path.exists(OLD_FIG1):
    old = xr.open_dataset(OLD_FIG1)
    je = np.clip(eta_c - 1, 0, old.sizes['eta_psi'] - 1)
    jx = np.clip(xi_c - 1, 0, old.sizes['xi_psi'] - 1)
    todo = [('FULL', ep, 'zof_meso', 'dc', dict(epoch=ep)) for ep in eps]
    if 'zof_tide' in old:
        todo += [('TIDE', str(ep), 'zof_tide', 'dc_tide', dict(epoch_ctrl=ep))
                 for ep in old.epoch_ctrl.values]
    for run, ep, vz, vd, selk in todo:
        if run not in RUNS or ep not in eps:
            continue
        zo = old[vz].sel(**selk).values[np.ix_(je, jx)]
        do = old[vd].sel(**selk).values[np.ix_(je, jx)]
        zn = zof_ep[RUNS.index(run), eps.index(ep)]
        dn = dc_ep[RUNS.index(run), eps.index(ep)]
        m = slope & np.isfinite(zo) & np.isfinite(zn)
        check.append(dict(run=run, epoch=ep, n=int(m.sum()),
                          spearman=float(spearmanr(zo[m], zn[m])[0]),
                          slope=float(np.polyfit(zo[m], zn[m], 1)[0]),
                          rms_model=float(np.sqrt(np.mean(zo[m]**2))),
                          rms_geostrophic=float(np.sqrt(np.mean(zn[m]**2))),
                          same_regime=float((np.sign(do[m]) == np.sign(dn[m])).mean())))
else:
    print(f'{OLD_FIG1} not found, the geostrophic vorticity is not checked')

# --------------------------------------------------------------------------
# the wind stress along the slope and over the footprint
# --------------------------------------------------------------------------
tau = {r: np.hypot(sets[r].taue.values, sets[r].taun.values)
       for r in RUNS if 'taue' in sets[r]}
tau_hov = np.full((nt, nb), np.nan)
tau_foot = {}
wind = {}
if 'FULL' in tau:
    for n in range(nt):
        tau_hov[n] = bin_mean(tau['FULL'][n][slope])
    for r in tau:
        tau_foot[r] = np.nanmean(tau[r][:, foot], axis=1)
    s = tau_foot['FULL']
    ipk = int(np.nanargmax(s))
    below = np.where(s[ipk:] < WIND_OFF_FRAC*s[ipk])[0]
    ioff = ipk + int(below[0]) if below.size else nt - 1
    wind = dict(t_peak=t[ipk], peak=float(s[ipk]), t_off=t[ioff])
    wind['max_point'] = float(np.nanmax(tau['FULL'][:, slope]))
    if 'WIND' in tau:
        # the same wind field forces both experiments, and whether the stress
        # the ocean received is the same is measured on the vectors themselves
        st = (t >= np.datetime64(EPOCHS['T2'][0])) & (t < np.datetime64(EPOCHS['T2'][1]))
        de = sets['FULL'].taue.values[st][:, foot] - sets['WIND'].taue.values[st][:, foot]
        dn = sets['FULL'].taun.values[st][:, foot] - sets['WIND'].taun.values[st][:, foot]
        wind['stress_rel_diff'] = float(np.sqrt(np.nanmean(de**2 + dn**2))
                                        / np.sqrt(np.nanmean(tau['FULL'][st][:, foot]**2)))

# --------------------------------------------------------------------------
# the semidiurnal elevation, its envelope and the spring and neap phases
# --------------------------------------------------------------------------
sos_sd = butter(3, [(1.0/SD_BAND_H[1])/nyq, (1.0/SD_BAND_H[0])/nyq], btype='band', output='sos')
sos_env = butter(3, (1.0/ENV_SMOOTH_H)/nyq, btype='low', output='sos')
pad = int(24/dt_h)
env, env_s, tide = {}, {}, {}
for run in ('FULL', 'TIDE'):
    if run not in RUNS:
        continue
    zs = sets[run].zeta.values[:, slope][:, ::ENV_STRIDE].astype(np.float64)
    zs = zs - np.nanmean(zs, axis=0)
    amp = np.abs(hilbert(sosfiltfilt(sos_sd, np.nan_to_num(zs), axis=0), axis=0))
    env[run] = np.median(amp, axis=1)
    env_s[run] = sosfiltfilt(sos_env, env[run])
e = env_s['FULL']
core = e[pad:-pad]
prom = 0.15*(core.max() - core.min())
i_spring = find_peaks(core, distance=int(6*24/dt_h), prominence=prom)[0] + pad
i_neap = find_peaks(-core, distance=int(6*24/dt_h), prominence=prom)[0] + pad
i_land = int(np.argmin(np.abs(t - LANDFALL)))
tide = dict(phase_landfall=float((e[i_land] - core.min())/(core.max() - core.min())),
            env_landfall=float(e[i_land]), env_min=float(core.min()), env_max=float(core.max()))
if 'TIDE' in env_s:
    tide['r_full_tide'] = float(np.corrcoef(env_s['FULL'][pad:-pad], env_s['TIDE'][pad:-pad])[0, 1])

# --------------------------------------------------------------------------
# rotary spectra of the slope columns against latitude
#
# One Hann windowed transform of the trimmed record at every level, integrated
# over depth and averaged over the columns of each latitude bin. The inertial
# sense is anticlockwise in time south of the equator, which is the positive
# half of the transform of u + iv.
# --------------------------------------------------------------------------
spec_mid = 0.5*(SPEC_BINS[1:] + SPEC_BINS[:-1])
spec_in, spec_out, spec_n, freq = {}, {}, None, None


def z_levels(s, C, hh, hc, Vt):
    s, C = np.asarray(s), np.asarray(C)
    return hh*((hc*s + hh*C)/(hc + hh)) if Vt == 2 else hc*s + (hh - hc)*C


# The profile caches were written as each integration finished, so they do not
# hold the same hours (649 for BACK and WIND, 657 for TIDE, 713 for FULL). The
# spectra are formed on the hours all four share, so every experiment has the
# same record, the same frequencies and the same resolution.
prof = {}
for run in RUNS:
    p = PROF.format(run=run)
    if not os.path.exists(p):
        print(f'{p} not found, no spectrum for {run}')
        continue
    prof[run] = xr.open_dataset(p)
tp_common = None
for d in prof.values():
    v = d.ocean_time.dt.round('s').values
    tp_common = v if tp_common is None else np.intersect1d(tp_common, v)
if prof:
    print(f'profile caches share {tp_common.size} records, '
          + ', '.join(f'{r} holds {d.sizes["ocean_time"]}' for r, d in prof.items()))

for run, d in prof.items():
    tp_all = d.ocean_time.dt.round('s').values
    kk = np.searchsorted(tp_all, tp_common)
    tp = tp_all[kk]
    keep_t = tp >= tp[0] + np.timedelta64(int(SPINUP_DAYS*24), 'h')
    keep = kk[keep_t]
    tp = tp[keep_t]
    n_t = keep.size
    dth = float(np.mean(np.diff(tp)/np.timedelta64(1, 'h')))
    win = get_window('hann', n_t)
    norm = (win**2).sum()*dth
    nh = n_t//2
    freq = np.fft.fftfreq(n_t, d=dth)[1:nh+1]
    hp, latp = d.h.values, d.lat.values
    u = d.u.values[keep]                                  # (t, s, point)
    v = d.v.values[keep]
    u = detrend(np.nan_to_num(u), axis=0)
    v = detrend(np.nan_to_num(v), axis=0)
    F = np.fft.fft((u + 1j*v)*win[:, None, None], axis=0)/np.sqrt(norm)
    P = np.abs(F)**2
    del F, u, v
    dz = np.stack([np.diff(z_levels(d.s_w.values, d.Cs_w.values, hh,
                                    d.attrs['hc'], d.attrs['Vtransform'])) for hh in hp], axis=1)
    ccw = 0.5*RHO0*(P[1:nh+1]*dz[None]).sum(axis=1)       # (freq, point), J m-2 per cph
    cw = 0.5*RHO0*(P[n_t-nh:][::-1]*dz[None]).sum(axis=1)
    del P
    jb = np.digitize(latp, SPEC_BINS) - 1
    a_in = np.full((spec_mid.size, nh), np.nan)
    a_out = np.full((spec_mid.size, nh), np.nan)
    spec_n = np.array([(jb == j).sum() for j in range(spec_mid.size)])
    for j in range(spec_mid.size):
        if spec_n[j] >= SPEC_MIN:
            a_in[j] = ccw[:, jb == j].mean(axis=1)
            a_out[j] = cw[:, jb == j].mean(axis=1)
    spec_in[run], spec_out[run] = a_in, a_out
    spec_attrs = dict(spec_nt=n_t, spec_resolution_cph=1.0/(n_t*dth),
                      spec_start=str(tp[0])[:16], spec_end=str(tp[-1])[:16],
                      spec_columns=int(latp.size))
    print(f'{run}: spectra of {latp.size} columns, {n_t} records')

# --------------------------------------------------------------------------
# the best track and the virtual moorings
# --------------------------------------------------------------------------
track = None
if os.path.exists(OLD_TRACK):
    shutil.copyfile(OLD_TRACK, OUTDIR + 'fig1_track.csv')
    track = pd.read_csv(OLD_TRACK, parse_dates=['time'])
else:
    try:
        import tropycal.tracks as tracks
        ib = tracks.TrackDataset(basin='all', source='ibtracs',
                                 ibtracs_mode='jtwc_neumann', catarina=True)
        st = ib.get_storm(('catarina', 2004))
        track = pd.DataFrame({'time': pd.to_datetime(st.time), 'lon': st.lon, 'lat': st.lat,
                              'vmax_ms': np.asarray(st.vmax, float)*0.514444, 'type': st.type})
        track.to_csv(OUTDIR + 'fig1_track.csv', index=False)
    except Exception as err:
        print('track not available:', err)

if os.path.exists(OLD_LABELS):
    sta = pd.read_csv(OLD_LABELS)[['name', 'lon', 'lat']]
else:
    sta = pd.DataFrame([dict(name=k, lon=float(lon[min(e//sub, ny-1), min(x//sub, nx-1)]),
                             lat=float(lat[min(e//sub, ny-1), min(x//sub, nx-1)]))
                        for k, (x, e) in BASE_MOORINGS.items()])
sta = sta.sort_values('lat', ascending=False).reset_index(drop=True)
jj = [np.unravel_index(np.argmin((lon - r.lon)**2 + (lat - r.lat)**2), lon.shape)
      for r in sta.itertuples()]
sta['depth_m'] = [float(h[j]) for j in jj]
sta['dc_planetary_cph'] = [float(dc_plan[j]*RAD2CPH) for j in jj]
for i, run in enumerate(RUNS):
    for k, ep in enumerate(eps):
        sta[f'dc_{run}_{ep}_cph'] = [float(dc_ep[i, k][j]*RAD2CPH) for j in jj]
sta.to_csv(OUTDIR + 'fig1_stations.csv', index=False)

# --------------------------------------------------------------------------
# the outlines of the two model grids
# --------------------------------------------------------------------------
def outline(path):
    g = xr.open_dataset(path)
    lo, la = g.lon_rho.values, g.lat_rho.values
    edge_lo = np.concatenate([lo[0, :], lo[:, -1], lo[-1, ::-1], lo[::-1, 0]])
    edge_la = np.concatenate([la[0, :], la[:, -1], la[-1, ::-1], la[::-1, 0]])
    return edge_lo, edge_la, lo.shape


grids = {}
for name, path in (('child', CHILD_GRID), ('parent', PARENT_GRID)):
    if not os.path.exists(path):
        hits = sorted(glob.glob(os.path.join(os.path.dirname(path), '*.nc')))
        path = hits[0] if hits else None
    if path is None:
        print(f'no {name} grid file found, its outline is left out of panel (a)')
        continue
    try:
        grids[name] = outline(path) + (path,)
        print(f'{name} grid {grids[name][2]} from {path}')
    except Exception as err:
        print(f'{name} grid not read from {path}:', err)

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars=dict(
        lon=(('y', 'x'), lon), lat=(('y', 'x'), lat), h=(('y', 'x'), h),
        ocean=(('y', 'x'), ocean), slope=(('y', 'x'), slope), sector=(('y', 'x'), sector),
        zof=(('run', 'epoch', 'y', 'x'), zof_ep), dc=(('run', 'epoch', 'y', 'x'), dc_ep),
        free_hov=(('run', 'time_b', 'lat_bin'), free_hov),
        free_pole=(('run', 'time_b'), free_pole), forb_equa=(('run', 'time_b'), forb_equa),
        strain_frac=(('run', 'time_b'), strain_frac),
        free_prof=(('run', 'epoch', 'lat_bin'), free_prof),
        free_plan=(('lat_bin',), free_plan),
        agree_r=(('pair', 'time_b'), agree_r), agree_sign=(('pair', 'time_b'), agree_s),
        agree_sign_foot=(('pair', 'time_b'), agree_s_foot),
        tau_hov=(('time', 'lat_bin'), tau_hov),
        env=(('time',), env['FULL']), env_smooth=(('time',), env_s['FULL'])),
    coords=dict(run=RUNS, epoch=eps, time=t, time_b=tb, lat_bin=lat_mid,
                pair=[f'{a}-{b}' for a, b in pairs], y=np.arange(ny), x=np.arange(nx)),
    attrs=dict(omega_M2=OMEGA_M2, lat_crit=LAT_CRIT, lp_hours=LP_HOURS, l_meso=L_MESO,
               rossby_limit=ROSSBY_LIMIT, slope_band=str(SLOPE_BAND), epochs=str(EPOCHS),
               region_eta=str(REGION_ETA), sub=sub, landfall=str(LANDFALL),
               t_spring=','.join(str(v)[:16] for v in t[i_spring]),
               t_neap=','.join(str(v)[:16] for v in t[i_neap]),
               phase_landfall=tide['phase_landfall'],
               t_wind_peak=str(wind.get('t_peak', ''))[:16],
               t_wind_off=str(wind.get('t_off', ''))[:16]))
for r, s in tau_foot.items():
    ds[f'tau_foot_{r}'] = (('time',), s)
if spec_in:
    sr = [r for r in RUNS if r in spec_in]
    ds['spec_in'] = (('run_spec', 'lat_spec', 'freq'), np.stack([spec_in[r] for r in sr]))
    ds['spec_out'] = (('run_spec', 'lat_spec', 'freq'), np.stack([spec_out[r] for r in sr]))
    ds['spec_n'] = (('lat_spec',), spec_n)
    ds = ds.assign_coords(run_spec=sr, lat_spec=spec_mid, freq=freq)
    ds.attrs.update(spec_attrs)
for name, (elo, ela, shp, path) in grids.items():
    ds[f'{name}_lon'] = ((f'edge_{name}',), elo)
    ds[f'{name}_lat'] = ((f'edge_{name}',), ela)
    ds.attrs[f'{name}_grid'] = f'{shp[0]} x {shp[1]} rho points from {path}'
ds.to_netcdf(OUTDIR + 'fig1_v5.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['FIGURE 1 REPORT (full_v5)  --  the margin, the storm, the tide and the resonance',
     '=' * 78]
L.append(f'Record {str(t[0])[:16]} to {str(t[-1])[:16]}, {nt} hourly records, '
         f'experiments {", ".join(RUNS)}')
L.append(f'Grid of {ny} x {nx} blocks of {sub} cells, '
         f'{np.nanmean(dx)/1e3:.2f} x {np.nanmean(dy)/1e3:.2f} km')
for name in grids:
    L.append(f'{name} grid: ' + ds.attrs[f'{name}_grid'])
L.append(f'omega_M2/2 = {0.5*OMEGA_M2*RAD2CPH:.5f} cph, planetary critical latitude '
         f'{LAT_CRIT:.2f} deg')
L.append(f'Balanced vorticity: (g/f) del2 of the elevation low passed at {LP_HOURS:.0f} h, '
         f'Gaussian sigma {0.5*L_MESO/1e3:.1f} km, kept every {STEP_H} h, '
         f'{int(edge.sum())} fields blanked at the ends')
L.append('')
L.append('CHECK OF THE GEOSTROPHIC VORTICITY AGAINST THE MODEL VORTICITY OF new_full')
if check:
    L.append(pd.DataFrame(check).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    L.append('   spearman and same_regime are the two numbers that decide. The geostrophic field is')
    L.append('   adopted when the rank correlation is above about 0.8 and nine tenths of the slope')
    L.append('   keeps its regime. Below that the model vorticity has to be extracted time resolved.')
else:
    L.append('   not made, the earlier file was not found')
L.append('')
L.append(f'DETUNING BY WINDOW ON THE SLOPE BAND ({SLOPE_BAND[0]:.0f} to {SLOPE_BAND[1]:.0f} m)')
L.append(f'{"run":5s} {"win":4s} {"free poleward":>14s} {"forb equatorward":>17s} '
         f'{"strained":>9s} {"zeta/f p05":>11s} {"p95":>7s} {"fields":>7s}')
for (run, ep), s in ep_stats.items():
    L.append(f'{run:5s} {ep:4s} {s["free_pole"]:14.3f} {s["forb_equa"]:17.3f} '
             f'{s["strained"]:9.3f} {s["p05"]:+11.3f} {s["p95"]:+7.3f} {s["nrec"]:7d}')
L.append('   under the planetary f alone the first column is zero and the second is zero')
L.append('   slope |zeta/f| before coarse-graining: '
         + ', '.join(f'{r} {v:.2f}' for r, v in zof_raw_absmax.items()))
L.append('')
L.append('HOW MOBILE THE REGIMES ARE (time resolved, between 16 March and 7 April)')
core_b = (tb >= np.datetime64(EPOCHS['T1'][0])) & (tb < np.datetime64(EPOCHS['T3'][1])) & ~edge
for i, run in enumerate(RUNS):
    a, b = free_pole[i, core_b], forb_equa[i, core_b]
    L.append(f'   {run}: poleward free fraction {np.nanmin(a):.3f} to {np.nanmax(a):.3f}, '
             f'equatorward forbidden fraction {np.nanmin(b):.3f} to {np.nanmax(b):.3f}')
flips = []
for i, run in enumerate(RUNS):
    sg = np.sign(np.stack([detuning(zof[run][n])[slope] for n in np.where(core_b)[0]]))
    flips.append(f'{run} {np.mean(np.abs(np.diff(sg, axis=0)).sum(axis=0) > 0):.3f}')
L.append('   share of the slope that changes regime at least once: ' + ', '.join(flips))
L.append('')
L.append('AGREEMENT BETWEEN EXPERIMENTS (slope band, mean by window)')
L.append(f'{"pair":11s} ' + ' '.join(f'{"r " + ep:>8s} {"same " + ep:>8s} {"foot " + ep:>8s}'
                                     for ep in eps))
for k, (a, b) in enumerate(pairs):
    row = f'{a + "-" + b:11s} '
    for ep, (t0, t1) in EPOCHS.items():
        sel = (tb >= np.datetime64(t0)) & (tb < np.datetime64(t1)) & ~edge
        row += (f'{np.nanmean(agree_r[k, sel]):8.3f} {np.nanmean(agree_s[k, sel]):8.3f} '
                f'{np.nanmean(agree_s_foot[k, sel]):8.3f} ')
    L.append(row)
L.append('   r is the pattern correlation of the balanced vorticity, same is the share of the')
L.append('   slope with the same sign of the detuning, foot is that share over the southern sector.')
L.append('   TIDE-BACK isolates what the tide does to the balanced flow, WIND-BACK what the storm')
L.append('   does, and one minus the FULL-WIND share bounds the area on which a difference between')
L.append('   the two wind forced experiments compares two different regimes.')
L.append('')
L.append('WIND STRESS')
if wind:
    same = str(wind['t_off'])[:13] == T_OFF[:13]
    L.append(f'   the windows are cut at T_OFF = {T_OFF}, the wind stop measured here is '
             f'{str(wind["t_off"])[:16]}, ' + ('they agree' if same else
                                              'THEY DIFFER, set T_OFF to the measured hour and rerun'))
if wind:
    L.append(f'   footprint mean peaks at {wind["peak"]:.2f} Pa on {str(wind["t_peak"])[:16]}, '
             f'largest block value on the slope {wind["max_point"]:.2f} Pa')
    L.append(f'   falls below {100*WIND_OFF_FRAC:.0f}% of that peak on {str(wind["t_off"])[:16]} '
             f'(the earlier version gave 28 March 11:00 from the profile columns)')
    if 'stress_rel_diff' in wind:
        L.append(f'   rms difference of the stress vector between FULL and WIND over the footprint '
                 f'in T2: {100*wind["stress_rel_diff"]:.2f}% of the rms stress')
        L.append('   a value near zero means the stress does not depend on the surface current, so')
        L.append('   the two experiments receive the same momentum flux and differ only in the work')
        L.append('   that flux does on a different surface current, which Figure 6 measures.')
else:
    L.append('   no stress in the surface cache of the joint experiment')
L.append('')
L.append('TIDE')
L.append('   spring: ' + ', '.join(str(v)[:16] for v in t[i_spring]))
L.append('   neap:   ' + ', '.join(str(v)[:16] for v in t[i_neap]))
L.append(f'   semidiurnal elevation envelope (median over the slope) {tide["env_min"]:.3f} to '
         f'{tide["env_max"]:.3f} m, {tide["env_landfall"]:.3f} m at landfall, '
         f'{100*tide["phase_landfall"]:.0f}% of the range')
if 'r_full_tide' in tide:
    L.append(f'   correlation of the envelope between FULL and TIDE {tide["r_full_tide"]:.3f}')
L.append('')
L.append('BAND GEOMETRY')
if spec_in:
    L.append(f'   spectra from {spec_attrs["spec_columns"]} columns, {spec_attrs["spec_start"]} to '
             f'{spec_attrs["spec_end"]}, resolution {spec_attrs["spec_resolution_cph"]:.5f} cph '
             f'({24*spec_attrs["spec_resolution_cph"]:.4f} cpd)')
    L.append(f'   {"lat":>7s} {"cols":>5s} {"f cpd":>7s} {"M2-f cpd":>9s} {"|2f-M2| cpd":>12s} '
             f'{"bins apart":>11s} {"peak cpd FULL":>14s}')
    band = (freq*24 > 0.6) & (freq*24 < 1.4)
    for j, la in enumerate(spec_mid):
        fc = 2*OMEGA_E*abs(np.sin(np.radians(la)))*RAD2CPH*24
        m2 = OMEGA_M2*RAD2CPH*24
        pk = (freq[band][np.nanargmax(spec_in['FULL'][j][band])]*24
              if 'FULL' in spec_in and np.isfinite(spec_in['FULL'][j]).any() else np.nan)
        L.append(f'   {la:7.2f} {spec_n[j]:5d} {fc:7.3f} {m2 - fc:9.3f} {abs(2*fc - m2):12.3f} '
                 f'{abs(2*fc - m2)/(24*spec_attrs["spec_resolution_cph"]):11.1f} {pk:14.3f}')
    L.append('   bins apart is the separation of f and M2 - f in units of the frequency resolution,')
    L.append('   and peak is the frequency of the largest near-inertial sense density of FULL between')
    L.append('   0.6 and 1.4 cpd, which shows where the wake sits against f and against M2/2.')
    L.append(f'   O1 0.9295, K1 1.0027, M2/2 {0.5*OMEGA_M2*RAD2CPH*24:.4f} cpd')
else:
    L.append('   no profile cache found, panel (c) will be empty')
L.append('')
L.append('STATIONS')
L.append(sta.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
if track is not None and 'type' in track:
    hu = track[track['type'] == 'HU']
    L += ['', 'TRACK', f'   record {track.time.iloc[0]} to {track.time.iloc[-1]}',
          f'   hurricane status {hu.time.iloc[0]} to {hu.time.iloc[-1]}' if len(hu) else
          '   no hurricane status in the track',
          f'   peak wind {track.vmax_ms.max():.1f} m s-1']
with open(OUTDIR + 'fig1_report.txt', 'w') as fh:
    fh.write('\n'.join(L))
print('\n'.join(L))