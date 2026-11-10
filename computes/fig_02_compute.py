"""
fig_02_compute.py  (full_v5, revision 4)
----------------------------------------
Figure 2. The kinetic energy of the hurricane wake with and without the
internal tide, each measured against the experiment that shares its balanced
flow.

The question this figure answers
--------------------------------
Figure 1 leaves a storm crossing a slope where the boundary current has moved
the subharmonic resonance. Does the wake hold more kinetic energy where the
internal tide is present than the same wind leaves without it?

Why the wake is measured inside matched pairs
---------------------------------------------
check_vorticity.py shows the four experiments falling into two pairs that share
their eddies, FULL with TIDE and WIND with BACK (pattern correlation of the
balanced vorticity 0.94 and 0.92 through the storm window), while a run with
the tide and a run without it are uncorrelated by the time the storm arrives.
The wake is therefore the velocity difference inside each pair,

    with the tide     u_w = u_FULL - u_TIDE ,
    without the tide  u_w = u_WIND - u_BACK ,

band-passed around half the tidal frequency, and its kinetic energy is the
depth integral of rho0 |u_w|^2 / 2. The difference removes from the joint run
the diurnal tide that sits inside the band (O1 and K1), which TIDE carries on
the same eddies, and removes from both pairs the band kinetic energy the
boundary current holds without any forcing. What the eddies that still differ
inside a pair leave behind is measured as the floor before the storm.

The band. The wake rings at the effective inertial frequency, 0.93 to 1.06 cpd
along the footprint before the vorticity shifts it, so the wake band is 0.80 to
1.20 of half the tidal frequency (0.77 to 1.16 cpd). The 0.90 to 1.10 band of
the earlier version is carried in the report for continuity.

The comparison. Each wake is binned by the detuning of its own balanced flow,
so wakes at equal distance from resonance are set against each other, and
each is divided by the near-inertial wind work it received (D'Asaro 1985,
Alford 2001, Alford 2020). The wind work is formed on the wake current itself,
so a wake the tide strengthens while the wind still blows also receives more
work. The ratio of kinetic energies is therefore the measure of the gain and
the ratio of efficiencies its lower bound, and the wind work difference is a
term of the budget of Figure 6.

Inputs
  ../new_full/fig6_cache_<RUN>/     u and v on the 3 km slope tiles, 18 March to
                                    9 April, from fig_06_extract.py, four runs
  ./surface/surface_<RUN>.nc        stress of FULL and WIND, from extract_surface.py
  ./fig1_out/fig1_v5.nc             the hour the wind stops, from fig_01_compute.py

Outputs (in OUTDIR)
  fig2_wake.nc          maps, group series with intervals, binned curves, tables
  fig2_wake_table.csv   the group table, both bands
  fig2_wake_report.txt

Run under nohup. The pre-pass that forms the detuning reads every cache once
without filtering (about a minute a tile), the main pass filters CHUNK_Z levels
at a time (about a minute a tile on Rio).
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu

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
SURF = './surface/surface_{run}.nc'
FIG1 = './fig1_out/fig1_v5.nc'
OUTDIR = './fig2_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
PAIRS = {'tide': ('FULL', 'TIDE'), 'notide': ('WIND', 'BACK')}
PAIR_DC = {'tide': 'FULL', 'notide': 'WIND'}       # the flow whose detuning each wake sees
T_OFF_DEFAULT = '2004-03-28T05:00'                 # wind below 15% of its peak (Figure 1)
T_STORM0, T_WAKE1 = '2004-03-24', '2004-04-07'
REGION_ETA = {'South': (0, 420), 'Central': (421, 900), 'North': (901, 10**6)}
SLOPE_BAND = (200.0, 2000.0)
EDGE_BUFFER = 60                                   # 1 km cells clear of the sponge

T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
BANDS = {'ni': (0.80, 1.20),                       # the wake band, times omega_M2/2
         'sub': (0.90, 1.10)}                      # the band of the earlier version
MAIN = 'ni'                                        # the band the figure draws
M2_BAND = (0.95, 1.05)                             # times omega_M2
Z_UPPER = -150.0
L_MESO = 25.0e3
SMOOTH_H = 25                                      # about one inertial period here
CHUNK_Z = 8

DC_EDGES = np.arange(-0.008, 0.0061, 0.001)        # cph
LAT_EDGES = np.arange(-33.0, -20.99, 0.5)          # latitude bins of the slope profile
MIN_LAT_PTS = 40                                   # slope points a latitude bin must hold
MIN_BIN = 150                                      # slope points a detuning bin must hold
BLOCK = 8                                          # 8 x 8 points of 3 km, about 24 km
N_BOOT = 2000
BAND_STEP_H = 6                                    # hours between the bootstrap bands in time
MARGINS = [0.0, 0.0005, 0.001]                     # cph around exact resonance
LAG_DAYS = [1, 3, 5, 7]                            # days after the wind stops
MIN_WORK = 0.05                                    # share of the footprint wind work a group needs
rng = np.random.default_rng(7)

# --------------------------------------------------------------------------
# grid and slab geometry
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h, pm, pn, f_rho = ref.h.values, ref.pm.values, ref.pn.values, ref.f.values
lon_rho, lat_rho, mask_rho = ref.lon_rho.values, ref.lat_rho.values, ref.mask_rho.values
angle = ref.angle.values if 'angle' in ref else np.zeros_like(h)
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
    raise SystemExit(f'no slab cache for {run}, run fig_06_extract.py {run} in ../new_full')


def to_hour(x):
    """Record stamps on the nearest hour. The model writes each hourly record a
    fraction of a second off the hour and the caches kept that stamp in different
    ways, truncated in the slabs and rounded in the surface files."""
    x = np.asarray(x).astype('datetime64[ns]')
    return (x + np.timedelta64(30, 'm')).astype('datetime64[h]').astype('datetime64[ns]')


cache = {r: find_cache(r) for r in RUNS}
geo = np.load(cache['FULL'] + 'geometry.npz', allow_pickle=True)
for r in RUNS:
    g = np.load(cache[r] + 'geometry.npz', allow_pickle=True)
    assert int(g['ntiles']) == int(geo['ntiles']) and np.array_equal(g['time'], geo['time']), \
        f'the slab cache of {r} is not on the geometry of FULL'
SUB = int(geo['SUB'])
eta_s, xi_s = geo['eta_s'], geo['xi_s']
t = to_hour(geo['time'])
nt = t.size
dt_h = float(np.mean(np.diff(t)/np.timedelta64(1, 'h')))
nyq = 0.5/dt_h
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
print(f'{ntiles} tiles, {nt} records from {str(t[0])[:16]} to {str(t[-1])[:16]}, '
      f'sub grid {ny_s} x {nx_s}', flush=True)

sub2 = np.ix_(np.minimum(eta_s, neta - 1), np.minimum(xi_s, nxi - 1))
lon_s, lat_s, h_s, mask_s, f_s = lon_rho[sub2], lat_rho[sub2], h[sub2], mask_rho[sub2], f_rho[sub2]
area_s = (SUB/pm[sub2])*(SUB/pn[sub2])
inside = ((eta_s[:, None] >= EDGE_BUFFER) & (eta_s[:, None] <= neta - EDGE_BUFFER)
          & (xi_s[None, :] >= EDGE_BUFFER) & (xi_s[None, :] <= nxi - EDGE_BUFFER))
slope_s = (h_s > SLOPE_BAND[0]) & (h_s < SLOPE_BAND[1]) & (mask_s == 1) & inside
sector_s = np.full((ny_s, nx_s), -1)
for code, (lo, hi) in enumerate(REGION_ETA.values()):
    sector_s[(eta_s >= lo) & (eta_s <= hi), :] = code
SECTORS = list(REGION_ETA)
foot = slope_s & (sector_s == 0)

# blocks of about 24 km, the unit of every bootstrap
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
blk_ids = np.unique(blk[slope_s])
blk_index = {b: i for i, b in enumerate(blk_ids)}
nblk = blk_ids.size
blk_of = np.vectorize(lambda b: blk_index.get(b, -1))(blk)
print(f'{nblk} blocks hold slope points')

# --------------------------------------------------------------------------
# windows, set on the hour the wind stops
# --------------------------------------------------------------------------
t_off = T_OFF_DEFAULT
if os.path.exists(FIG1):
    t_off = xr.open_dataset(FIG1).attrs.get('t_wind_off', '') or T_OFF_DEFAULT
T_OFF = np.datetime64(t_off)
WINDOWS = {'pre': (str(t[0] + np.timedelta64(24, 'h'))[:13], T_STORM0),
           'storm': (T_STORM0, t_off),
           'wake': (t_off, T_WAKE1)}
W = {k: (t >= np.datetime64(a0)) & (t < np.datetime64(a1)) for k, (a0, a1) in WINDOWS.items()}
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
print('windows', WINDOWS, flush=True)


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


half = 0.5*OMEGA_M2*RAD2CPH
SOS = {b: band_sos(lo*half, hi*half) for b, (lo, hi) in BANDS.items()}
SOS['M2'] = band_sos(M2_BAND[0]*2*half, M2_BAND[1]*2*half)


def bp(x, name):
    sos, pad = SOS[name]
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


# --------------------------------------------------------------------------
# pre-pass. The detuning each wake sees, from the wake-window mean velocity of
# the upper 150 m of its own experiment, coarse-grained as in Figure 1. It is
# formed first because the classes it defines are needed while the main pass
# accumulates the block series.
# --------------------------------------------------------------------------
zeta_w = {r: np.full((ny_s, nx_s), np.nan, np.float32) for r in RUNS}
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = z_of(s_rho, Cs_r, h_t)
    w150 = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0)*(zr > Z_UPPER), 0, -1)
    w150 = (w150/np.maximum(w150.sum(axis=-1, keepdims=True), 1e-6)).astype(np.float32)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    kw = np.where(W['wake'])[0]
    for run in RUNS:
        u = np.load(cache[run] + f'u_t{k:03d}.npy', mmap_mode='r')
        v = np.load(cache[run] + f'v_t{k:03d}.npy', mmap_mode='r')
        Um = (np.nan_to_num(np.asarray(u[kw[0]:kw[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
        Vm = (np.nan_to_num(np.asarray(v[kw[0]:kw[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
        zt = np.full(Um.shape, np.nan, np.float32)
        zt[1:-1, 1:-1] = ((Vm[1:-1, 2:] - Vm[1:-1, :-2])/(2*dxi[1:-1, 1:-1])
                          - (Um[2:, 1:-1] - Um[:-2, 1:-1])/(2*deta[1:-1, 1:-1]))
        zeta_w[run][b0:b1, cols] = zt[rows]
    print(f'pre-pass tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)

sig = 0.5*L_MESO/float(np.nanmean(SUB/pm))
dc_w, zof_w = {}, {}
for run in RUNS:
    g = np.isfinite(zeta_w[run])
    num = gaussian_filter(np.where(g, zeta_w[run], 0.0), sig, mode='nearest')
    den = gaussian_filter(g.astype(float), sig, mode='nearest')
    zs = np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)
    zof_w[run] = zs/f_s
    dc_w[run] = np.abs(f_s + 0.5*zs) - 0.5*OMEGA_M2
CLASSES = {'free': {p: dc_w[PAIR_DC[p]] < 0 for p in PAIRS},
           'trapped': {p: dc_w[PAIR_DC[p]] > 0 for p in PAIRS}}

# --------------------------------------------------------------------------
# stress from the surface caches
# --------------------------------------------------------------------------
stress = {}
for run in ('FULL', 'WIND'):
    d = xr.open_dataset(SURF.format(run=run))
    ts = to_hour(d.ocean_time.values)
    k = np.clip(np.searchsorted(ts, t), 0, ts.size - 1)
    assert np.array_equal(ts[k], t), \
        (f'the surface cache of {run} does not hold the slab records, it runs '
         f'{str(ts[0])[:16]} to {str(ts[-1])[:16]} against {str(t[0])[:16]} to {str(t[-1])[:16]}')
    stress[run] = (d.taue.values[k], d.taun.values[k])
    ny_b, nx_b = d.sizes['y'], d.sizes['x']
print('stress read for FULL and WIND', flush=True)

# --------------------------------------------------------------------------
# main pass
# --------------------------------------------------------------------------
QMAP = ([f'Ew{b}_{p}' for b in BANDS for p in PAIRS]
        + [f'Esub_{r}' for r in RUNS] + [f'M2_{r}' for r in RUNS])
QSER = QMAP + [f'Pi_{p}' for p in PAIRS]
maps = {(q, w): np.full((ny_s, nx_s), np.nan, np.float32) for q in QMAP for w in WINDOWS}
work = {p: np.full((ny_s, nx_s), np.nan, np.float32) for p in PAIRS}
series = {q: np.zeros((nt, len(SECTORS))) for q in QSER}
area_sec = np.zeros(len(SECTORS))
# block series of the wake kinetic energy, by pair and by class of its own flow
Eblk = {(p, c): np.zeros((nt, nblk), np.float64) for p in PAIRS for c in ('all', 'free', 'trapped')}
Ablk = {(p, c): np.zeros(nblk) for p in PAIRS for c in ('all', 'free', 'trapped')}
blk_sector = np.full(nblk, -1)

t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    an = angle[np.ix_(jy, jx)]
    jb = np.minimum((ys - 1)//3, ny_b - 1)             # the 3 km block of the surface cache
    ib = np.minimum(xs//3, nx_b - 1)

    mm = {r: (np.load(cache[r] + f'u_t{k:03d}.npy', mmap_mode='r'),
              np.load(cache[r] + f'v_t{k:03d}.npy', mmap_mode='r')) for r in RUNS}
    nz = mm['FULL'][0].shape[-1]
    shp = mm['FULL'][0].shape[:3]
    E = {q: np.zeros(shp, np.float32) for q in QMAP}
    top = {}
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        dzc = dz[..., z0:z1]
        raw, sub = {}, {}
        for run in RUNS:
            u = np.nan_to_num(np.asarray(mm[run][0][..., z0:z1], dtype=np.float32))
            v = np.nan_to_num(np.asarray(mm[run][1][..., z0:z1], dtype=np.float32))
            sub[run] = (bp(u, 'sub'), bp(v, 'sub'))
            E[f'Esub_{run}'] += 0.5*RHO0*((sub[run][0]**2 + sub[run][1]**2)*dzc).sum(axis=-1)
            um, vm = bp(u, 'M2'), bp(v, 'M2')
            E[f'M2_{run}'] += 0.5*RHO0*((um**2 + vm**2)*dzc).sum(axis=-1)
            raw[run] = (u, v)
            del um, vm
        for p, (a_, b_) in PAIRS.items():
            du, dv = sub[a_][0] - sub[b_][0], sub[a_][1] - sub[b_][1]
            E[f'Ewsub_{p}'] += 0.5*RHO0*((du**2 + dv**2)*dzc).sum(axis=-1)
            du = bp(raw[a_][0] - raw[b_][0], 'ni')
            dv = bp(raw[a_][1] - raw[b_][1], 'ni')
            E[f'Ewni_{p}'] += 0.5*RHO0*((du**2 + dv**2)*dzc).sum(axis=-1)
            if z1 == nz:
                top[p] = (du[..., -1].copy(), dv[..., -1].copy())          # the surface level
        del raw, sub, du, dv

    Pi = {}
    for p, (a_, _b) in PAIRS.items():
        te, tn = stress[a_]
        te, tn = te[:, jb][:, :, ib], tn[:, jb][:, :, ib]
        tx = te*np.cos(an) + tn*np.sin(an)                # east and north to the grid axes
        ty = -te*np.sin(an) + tn*np.cos(an)
        Pi[p] = np.nan_to_num(tx*top[p][0] + ty*top[p][1])                 # W m-2
        work[p][b0:b1, cols] = (Pi[p][WORK].sum(axis=0)*dt_h*3600.0)[rows]

    sl, sec, ar = slope_s[b0:b1, cols], sector_s[b0:b1, cols], area_s[b0:b1, cols]
    bo = blk_of[b0:b1, cols]
    for q in QSER:
        x = (E[q] if q in E else Pi[q[3:]])[:, rows]
        if q in E:
            for w in WINDOWS:
                maps[(q, w)][b0:b1, cols] = x[W[w]].mean(axis=0)
        for j in range(len(SECTORS)):
            m = sl & (sec == j)
            series[q][:, j] += (x[:, m]*ar[m]).sum(axis=1)
    for j in range(len(SECTORS)):
        area_sec[j] += ar[sl & (sec == j)].sum()
    for p in PAIRS:
        x = E[f'Ew{MAIN}_{p}'][:, rows]
        for c in ('all', 'free', 'trapped'):
            m = sl & (bo >= 0) if c == 'all' else sl & (bo >= 0) & CLASSES[c][p][b0:b1, cols]
            if not m.any():
                continue
            ids = bo[m]
            uniq, inv = np.unique(ids, return_inverse=True)
            onehot = (inv[:, None] == np.arange(uniq.size)[None, :]).astype(np.float64)
            Eblk[(p, c)][:, uniq] += (x[:, m]*ar[m]) @ onehot
            Ablk[(p, c)][uniq] += ar[m] @ onehot
    m = sl & (bo >= 0)
    blk_sector[bo[m]] = sec[m]
    print(f'tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)

# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def block_sums(sel, num, den):
    ok = sel & np.isfinite(num) & np.isfinite(den)
    if not ok.any():
        return np.zeros(0), np.zeros(0), 0
    _ids, inv = np.unique(blk[ok], return_inverse=True)
    return (np.bincount(inv, weights=(num*area_s)[ok]), np.bincount(inv, weights=(den*area_s)[ok]),
            int(ok.sum()))


def ratio_boot(sn, sd):
    if sn.size < 3 or sd.sum() <= 0:
        return np.nan, np.full(N_BOOT, np.nan)
    i = rng.integers(0, sn.size, size=(N_BOOT, sn.size))
    with np.errstate(divide='ignore', invalid='ignore'):
        return sn.sum()/sd.sum(), sn[i].sum(axis=1)/sd[i].sum(axis=1)


def groups_for(p, margin=0.0):
    dcc = dc_w[PAIR_DC[p]]*RAD2CPH
    return {'footprint free': foot & (dcc < -margin), 'footprint trapped': foot & (dcc > margin),
            'footprint': foot, 'central': slope_s & (sector_s == 1),
            'north': slope_s & (sector_s == 2)}


GROUPS = list(groups_for('tide'))
Ew = {b: {p: maps[(f'Ew{b}_{p}', 'wake')] for p in PAIRS} for b in BANDS}
ones = {p: np.where(np.isfinite(Ew[MAIN][p]), 1.0, np.nan) for p in PAIRS}

# kinetic energy and efficiency of each wake against the detuning of its own flow
centres = 0.5*(DC_EDGES[1:] + DC_EDGES[:-1])
curve_E = {p: np.full((centres.size, 3), np.nan) for p in PAIRS}        # value, p05, p95
curve_eff = {p: np.full((centres.size, 3), np.nan) for p in PAIRS}
npts = {p: np.zeros(centres.size, int) for p in PAIRS}
for p in PAIRS:
    dcc = dc_w[PAIR_DC[p]]*RAD2CPH
    for i in range(centres.size):
        sel = foot & (dcc >= DC_EDGES[i]) & (dcc < DC_EDGES[i + 1])
        sn, sa, n = block_sums(sel, Ew[MAIN][p], ones[p])
        npts[p][i] = n
        if n < MIN_BIN:
            continue
        val, bs = ratio_boot(sn, sa)
        curve_E[p][i] = [val, np.nanpercentile(bs, 5), np.nanpercentile(bs, 95)]
        sn, sd, n = block_sums(sel, Ew[MAIN][p], work[p])
        if sd.sum() > 0:
            val, bs = ratio_boot(sn, sd)
            curve_eff[p][i] = [val, np.nanpercentile(bs, 5), np.nanpercentile(bs, 95)]

# the two wakes and their ratio by latitude along the whole slope band
lat_mid = 0.5*(LAT_EDGES[1:] + LAT_EDGES[:-1])
lat_prof = {p: np.full(lat_mid.size, np.nan) for p in PAIRS}
lat_ratio = np.full((lat_mid.size, 3), np.nan)                      # value, p05, p95
for i in range(lat_mid.size):
    sel = slope_s & (lat_s >= LAT_EDGES[i]) & (lat_s < LAT_EDGES[i + 1])
    sums, boots = {}, {}
    for p in PAIRS:
        sn, sa, n = block_sums(sel, Ew[MAIN][p], ones[p])
        if n < MIN_LAT_PTS:
            break
        lat_prof[p][i] = sn.sum()/sa.sum()
        sums[p], boots[p] = ratio_boot(sn, sa)
    if len(boots) == 2:
        with np.errstate(divide='ignore', invalid='ignore'):
            bs = boots['tide']/boots['notide']
        lat_ratio[i] = [lat_prof['tide'][i]/lat_prof['notide'][i],
                        np.nanpercentile(bs, 5), np.nanpercentile(bs, 95)]

rows_tab = []
gratio = np.full((len(GROUPS), 2, 3), np.nan)        # group, (kinetic energy, efficiency), (value p05 p95)
for b in BANDS:
    for gi, gname in enumerate(GROUPS):
        rec = dict(band=b, group=gname)
        eb, nb_, blocks = {}, {}, {}
        for p in PAIRS:
            sel = groups_for(p)[gname]
            sn, sd, n = block_sums(sel, Ew[b][p], work[p])
            _, sa, _ = block_sums(sel, Ew[b][p], ones[p])
            val, eb[p] = ratio_boot(sn, sd)
            _, nb_[p] = ratio_boot(sn, sa)
            rec[f'eff_{p}'] = val
            rec[f'E_{p}_Jm2'] = sn.sum()/sa.sum() if sa.size else np.nan
            rec[f'W_{p}_Jm2'] = sd.sum()/sa.sum() if sa.size else np.nan
            rec[f'n_{p}'] = n
            blocks[p] = sn[sd > 0]/sd[sd > 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            rb, ee = eb['tide']/eb['notide'], nb_['tide']/nb_['notide']
        rec['E_ratio'] = rec['E_tide_Jm2']/rec['E_notide_Jm2']
        rec['E_lo'], rec['E_hi'] = np.nanpercentile(ee, 5), np.nanpercentile(ee, 95)
        rec['eff_ratio'] = rec['eff_tide']/rec['eff_notide']
        rec['eff_lo'], rec['eff_hi'] = np.nanpercentile(rb, 5), np.nanpercentile(rb, 95)
        if blocks['tide'].size >= 5 and blocks['notide'].size >= 5:
            rec['hl_shift'] = float(np.median(blocks['tide'][:, None] - blocks['notide'][None, :]))
            rec['mw_p'] = float(mannwhitneyu(blocks['tide'], blocks['notide'],
                                             alternative='two-sided').pvalue)
        # the semidiurnal band, kept for the budget of Figure 6
        for a_, b_, name in (('FULL', 'TIDE', 'M2_FULL_over_TIDE'), ('WIND', 'BACK', 'M2_WIND_over_BACK')):
            sn, sd, n = block_sums(groups_for('tide' if a_ == 'FULL' else 'notide')[gname],
                                   maps[(f'M2_{a_}', 'wake')], maps[(f'M2_{b_}', 'wake')])
            val, bs = ratio_boot(sn, sd)
            rec[name], rec[name + '_lo'], rec[name + '_hi'] = val, np.nanpercentile(bs, 5), \
                np.nanpercentile(bs, 95)
        rows_tab.append(rec)
        if b == MAIN:
            gratio[gi, 0] = [rec['E_ratio'], rec['E_lo'], rec['E_hi']]
            gratio[gi, 1] = [rec['eff_ratio'], rec['eff_lo'], rec['eff_hi']]
tab = pd.DataFrame(rows_tab)
w_ref = float(tab[(tab.band == MAIN) & (tab.group == 'footprint')].W_tide_Jm2.iloc[0])
weak = (tab.W_tide_Jm2 < MIN_WORK*w_ref) | (tab.W_notide_Jm2 <= 0)
tab.loc[weak, ['eff_ratio', 'eff_lo', 'eff_hi', 'hl_shift', 'mw_p']] = np.nan
for gi, gname in enumerate(GROUPS):
    if bool(weak[(tab.band == MAIN) & (tab.group == gname)].iloc[0]):
        gratio[gi, 1] = np.nan
tab.to_csv(OUTDIR + 'fig2_wake_table.csv', index=False)

sens = []
for m in MARGINS:
    for cls in ('footprint free', 'footprint trapped'):
        vals = {}
        for p in PAIRS:
            sn, sa, n = block_sums(groups_for(p, m)[cls], Ew[MAIN][p], ones[p])
            vals[p] = sn.sum()/sa.sum() if sa.sum() > 0 else np.nan
        sens.append(dict(margin_cph=m, group=cls, E_ratio=vals['tide']/vals['notide']))

# --------------------------------------------------------------------------
# the gain through time, by group, with block-bootstrap bands
# --------------------------------------------------------------------------


def smooth(x, n=SMOOTH_H):
    k_ = np.ones(n)/n
    return np.apply_along_axis(lambda c: np.convolve(np.pad(c, n//2, mode='edge'), k_,
                                                     mode='valid')[:c.size], 0, x)


GROUP_BLOCKS = {'footprint free': ('free', blk_sector == 0), 'footprint trapped': ('trapped', blk_sector == 0),
                'footprint': ('all', blk_sector == 0), 'central': ('all', blk_sector == 1),
                'north': ('all', blk_sector == 2)}
kb = np.arange(0, nt, int(round(BAND_STEP_H/dt_h)))
ratio_t = np.full((len(GROUPS), nt), np.nan)
ratio_band = np.full((len(GROUPS), kb.size, 2), np.nan)
for gi, gname in enumerate(GROUPS):
    c, sel = GROUP_BLOCKS[gname]
    num, den = smooth(Eblk[('tide', c)][:, sel]), smooth(Eblk[('notide', c)][:, sel])
    an_, ad_ = Ablk[('tide', c)][sel], Ablk[('notide', c)][sel]
    if sel.sum() < 3 or ad_.sum() <= 0:
        continue
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_t[gi] = (num.sum(1)/an_.sum())/(den.sum(1)/ad_.sum())
        i = rng.integers(0, sel.sum(), size=(N_BOOT, sel.sum()))
        bs = ((num[kb][:, i].sum(-1)/an_[i].sum(-1))
              / (den[kb][:, i].sum(-1)/ad_[i].sum(-1)))               # (kb, N_BOOT)
    ratio_band[gi, :, 0] = np.nanpercentile(bs, 5, axis=1)
    ratio_band[gi, :, 1] = np.nanpercentile(bs, 95, axis=1)

fa = area_sec[0]
ser_s = {q: smooth(series[q][:, 0]/fa) for q in QSER}
cum = {p: np.cumsum(np.where(WORK, series[f'Pi_{p}'][:, 0]/fa, 0.0))*dt_h*3600.0 for p in PAIRS}
timing = {}
for p in PAIRS:
    e = ser_s[f'Ew{MAIN}_{p}']
    ipk = int(np.argmax(np.where(t >= np.datetime64(T_STORM0), e, -np.inf)))
    sel = (t > t[ipk]) & (t <= t[ipk] + np.timedelta64(4, 'D')) & (e > 0)
    tau = (-1.0/np.polyfit((t[sel] - t[ipk])/np.timedelta64(1, 'D'), np.log(e[sel]), 1)[0]
           if sel.sum() > 24 else np.nan)
    timing[p] = dict(t_peak=t[ipk], E_peak=float(e[ipk]), efold=float(tau),
                     floor=float(np.mean(e[W['pre']])), W_total=float(cum[p][-1]))
gi_foot = GROUPS.index('footprint')
lags = {d: float(ratio_t[gi_foot, int(np.argmin(np.abs(t - (T_OFF + np.timedelta64(24*d, 'h')))))])
        for d in LAG_DAYS}

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars={**{f'{q}_{w}': (('y', 'x'), maps[(q, w)]) for q in QMAP for w in WINDOWS},
               **{f'work_{p}': (('y', 'x'), work[p]) for p in PAIRS},
               **{f'dc_{r}': (('y', 'x'), dc_w[r]) for r in RUNS},
               **{f'zof_{r}': (('y', 'x'), zof_w[r]) for r in RUNS},
               **{f'ser_{q}': (('time', 'sector'), series[q]/area_sec[None, :]) for q in QSER},
               **{f'cumwork_{p}': (('time',), cum[p]) for p in PAIRS},
               **{f'curveE_{p}': (('dc_bin', 'stat'), curve_E[p]) for p in PAIRS},
               **{f'curveeff_{p}': (('dc_bin', 'stat'), curve_eff[p]) for p in PAIRS},
               **{f'npts_{p}': (('dc_bin',), npts[p]) for p in PAIRS},
               **{f'latprof_{p}': (('lat_bin',), lat_prof[p]) for p in PAIRS},
               'lat_ratio': (('lat_bin', 'stat'), lat_ratio),
               'ratio_t': (('group', 'time'), ratio_t),
               'ratio_band': (('group', 'time_band', 'edge'), ratio_band),
               'group_ratio': (('group', 'quantity', 'stat'), gratio),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'slope': (('y', 'x'), slope_s), 'sector_code': (('y', 'x'), sector_s),
               'area': (('y', 'x'), area_s), 'mask': (('y', 'x'), mask_s)},
    coords=dict(time=t, time_band=t[kb], sector=SECTORS, dc_bin=centres, lat_bin=lat_mid,
                stat=['value', 'p05', 'p95'], edge=['p05', 'p95'], group=GROUPS,
                quantity=['kinetic energy with over without the tide',
                          'kinetic energy per unit wind work with over without the tide']),
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=t_off,
               bands=str(BANDS), main_band=MAIN, m2_band=str(M2_BAND), pairs=str(PAIRS),
               block_points=BLOCK, n_boot=N_BOOT, smooth_h=SMOOTH_H, omega_M2=OMEGA_M2))
ds.to_netcdf(OUTDIR + 'fig2_wake.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
cpd = 24*half
L = ['FIGURE 2 REPORT (full_v5, revision 4)  --  wake kinetic energy with and without the tide',
     '=' * 86,
     f'slab caches, {nt} records {str(t[0])[:16]} to {str(t[-1])[:16]}, sub grid {ny_s} x {nx_s}',
     'wake with the tide u_FULL - u_TIDE, without the tide u_WIND - u_BACK, kinetic energy of',
     'the band-passed difference integrated over depth (J m-2)',
     'bands: ' + ', '.join(f'{b} {lo} to {hi} of omega_M2/2 ({lo*cpd:.3f} to {hi*cpd:.3f} cpd)'
                           for b, (lo, hi) in BANDS.items())
     + f', semidiurnal {M2_BAND[0]} to {M2_BAND[1]} of omega_M2. The figure draws the {MAIN} band',
     f'windows {WINDOWS}', f'wind work accumulated from {T_STORM0} to {T_WAKE1}',
     f'slope band {SLOPE_BAND}, footprint {fa/1e6:.0f} km2, {nblk} blocks of {BLOCK} points, '
     f'{N_BOOT} bootstrap draws, intervals 5 to 95%', '']
L.append('1. THE FOOTPRINT THROUGH TIME (footprint mean, 25 h running mean)')
for p in PAIRS:
    d = timing[p]
    L.append(f'   {p:7s} peak {d["E_peak"]:.1f} J m-2 on {str(d["t_peak"])[:16]}, e-folding over the '
             f'next 4 days {d["efold"]:.2f} d, floor before the storm {d["floor"]:.2f} J m-2 '
             f'({100*d["floor"]/d["E_peak"]:.1f}% of the peak), wind work {d["W_total"]:.1f} J m-2')
L.append(f'   wind stops {t_off}. Wake kinetic energy with over without the tide '
         + ', '.join(f'{lags[d]:.2f} at +{d} d' for d in LAG_DAYS))
L.append('   the floor is what the eddies that still differ inside a pair leave in the band before')
L.append('   any storm, and it is the same in both pairs when the diurnal tide has cancelled')
L.append('')
L.append('2. THE GAIN THROUGH TIME BY GROUP (ratio with over without, 5 to 95% band every 6 h)')
for d in (-2, 0, 1, 2, 3, 5, 7, 9):
    j = int(np.argmin(np.abs(t - (T_OFF + np.timedelta64(24*d, 'h')))))
    jb_ = int(np.argmin(np.abs(t[kb] - t[j])))
    L.append(f'   {d:+d} d: ' + ', '.join(f'{g} {ratio_t[gi, j]:.2f} [{ratio_band[gi, jb_, 0]:.2f} '
                                         f'{ratio_band[gi, jb_, 1]:.2f}]' for gi, g in enumerate(GROUPS)))
L.append('')
L.append('3. THE WAKE AGAINST THE DIFFERENCE OF KINETIC ENERGIES (footprint, wake window, sub band)')
for p, (a, b) in PAIRS.items():
    ew = np.nansum((maps[(f'Ewsub_{p}', 'wake')]*area_s)[foot])/np.nansum(area_s[foot])
    de = np.nansum(((maps[(f'Esub_{a}', 'wake')] - maps[(f'Esub_{b}', 'wake')])*area_s)[foot]) \
        / np.nansum(area_s[foot])
    L.append(f'   {p:7s} energy of the velocity difference {ew:9.2f}   difference of energies {de:9.2f}')
L.append('')
L.append('4. GROUPS (wake window means, efficiency = wake kinetic energy over accumulated wind work)')
show = ['band', 'group', 'n_tide', 'n_notide', 'E_tide_Jm2', 'E_notide_Jm2', 'E_ratio', 'E_lo', 'E_hi',
        'W_tide_Jm2', 'W_notide_Jm2', 'eff_ratio', 'eff_lo', 'eff_hi', 'hl_shift', 'mw_p']
L.append(tab[show].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
L.append('   E_ratio is the measure of the gain. eff_ratio is its lower bound, because the wind work')
L.append('   is formed on the wake current and grows with a wake the tide strengthens while the wind')
L.append(f'   blows. It is withheld where a group received under {100*MIN_WORK:.0f}% of the footprint')
L.append('   work or negative work. The wind work difference is a term of the budget of Figure 6.')
L.append('')
L.append('5. THE SEMIDIURNAL BAND (wake window, kinetic energy ratios inside each pair, for Figure 6)')
L.append(tab[tab.band == MAIN][['group', 'M2_FULL_over_TIDE', 'M2_FULL_over_TIDE_lo', 'M2_FULL_over_TIDE_hi',
                                'M2_WIND_over_BACK', 'M2_WIND_over_BACK_lo', 'M2_WIND_over_BACK_hi']]
         .to_string(index=False, float_format=lambda x: f'{x:.4g}'))
L.append('   FULL over TIDE holds the storm-made semidiurnal motion, the loss or gain of the tide and')
L.append('   their cross terms. WIND over BACK holds the storm-made semidiurnal motion alone (direct')
L.append('   forcing and the 2f harmonic of the wake, which at this latitude is the tidal frequency).')
L.append('')
L.append(f'6. KINETIC ENERGY AND EFFICIENCY BY DETUNING OVER THE FOOTPRINT ({MAIN} band)')
L.append(f'{"dc cph":>8s} {"n tide":>7s} {"E tide":>9s} {"interval":>19s} {"n no":>6s} {"E no":>9s} '
         f'{"interval":>19s} {"eff tide":>9s} {"eff no":>9s}')
for i, c in enumerate(centres):
    a, b = curve_E['tide'][i], curve_E['notide'][i]
    L.append(f'{c:8.4f} {npts["tide"][i]:7d} {a[0]:9.4g} {a[1]:8.4g} to {a[2]:7.4g} '
             f'{npts["notide"][i]:6d} {b[0]:9.4g} {b[1]:8.4g} to {b[2]:7.4g} '
             f'{curve_eff["tide"][i][0]:9.4g} {curve_eff["notide"][i][0]:9.4g}')
L.append('')
L.append(f'7. THE TWO WAKES BY LATITUDE ALONG THE SLOPE BAND ({MAIN} band, wake window, J m-2, ratio with 5 to 95%)')
L.append(f'{"lat":>7s} {"E tide":>9s} {"E no":>9s} {"ratio":>7s} {"interval":>17s}')
for i, c in enumerate(lat_mid):
    if np.isfinite(lat_ratio[i, 0]):
        L.append(f'{c:7.2f} {lat_prof["tide"][i]:9.4g} {lat_prof["notide"][i]:9.4g} {lat_ratio[i, 0]:7.3f} '
                 f'{lat_ratio[i, 1]:7.3f} to {lat_ratio[i, 2]:6.3f}')
L.append('')
L.append('8. SENSITIVITY OF THE CLASS RATIO TO A MARGIN AROUND EXACT RESONANCE')
L.append(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: f'{x:.4g}'))
L.append('')
L.append('9. DETUNING OF THE WAKE WINDOW FROM THE VELOCITY (slope band)')
for r in RUNS:
    d = dc_w[r][slope_s]*RAD2CPH
    L.append(f'   {r}: share with negative detuning {np.nanmean(d < 0):.3f}, poleward of 28.8S '
             f'{np.nanmean(d[lat_s[slope_s] < -28.8] < 0):.3f}')
L.append('')
L.append('READING THE RESULT')
L.append('   question 1 is answered yes if E_ratio over the footprint has its interval above one,')
L.append('   the ratio of section 2 opens at or after the wind stops, the floor is a small share of')
L.append('   the peak, and the north control sits at one.')
with open(OUTDIR + 'fig2_wake_report.txt', 'w') as fh:
    fh.write('\n'.join(L))
print('\n'.join(L))