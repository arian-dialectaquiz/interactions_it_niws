"""
figS9_compute.py  (full_v5, revision 3)  Supplementary Figure S9, step 2
------------------------------------------------------------------------
Is the 3 km sampling of the 1 km model enough for the works of the paper?

Every energy term of the paper is formed on every third point of the 1 km
grid, with centred differences over 6 km (fig_03_compute.py). Here the same
terms are formed on one footprint tile cached at every point of the model
(figS9_extract.py), at 1 km and at coarser spacings, on identical columns,
records and levels. Every spacing is taken from the same fields, so the
comparison between spacings is exact whatever the fields are.

Test 1, the quantity of the paper
  wake    the near-inertial band of FULL - TIDE
  tide    the semidiurnal band of TIDE
  The tide run on disk (avg_TIDES_ONLY_V2_part*.nc) is the same configuration
  run again for a longer period. It follows the run of the paper closely
  (figS9_check_tide.py: correlation 0.998 on 18 March, 0.995 at the wind stop,
  0.95 in April). For a comparison between spacings that is enough, because
  1 km and 3 km come from the same fields. How close the repeated run is to the
  paper is measured at 3 km, where the slab cache of the paper holds its own
  tide run on the same tile (the anchor below).
  Parts 2 and 3 of the V2 files hold hourly means centred half an hour off the
  hour. Their records are interpolated linearly in time to the hour before any
  filter, from the time stamps of the files.

Test 2, FULL alone
  wake    the near-inertial band of FULL
  tide    the semidiurnal band of FULL
  It uses no tide run at all and its 1 km cache reproduces the paper exactly,
  so it is an independent check of test 1.

Variants of both tests, on the same tile and region
  s1  s2  s3  s4  s6    every point, every second, third, fourth and sixth
                        point, no averaging. s3 is the sampling of the paper
  avg3                  the filtered fields averaged over 3 x 3 points and
                        then taken every third point

The anchor at 3 km, from the slab caches of the paper on this tile
  it     the works of the internal tide of the paper on its wake
  sm     the works of the storm-made semidiurnal motion on that wake
  full   the works of FULL alone, which must equal variant s3 of test 2

Checks
  every third point of the 1 km cache of FULL against the slab cache (exact)
  the works of FULL alone from the slab cache against variant s3 (exact)
  the repeated tide run against the tide run of the paper at 3 km (reported)

Inputs
  ./figS9_cache_FULL/  ./figS9_cache_TIDE/   the tile at 1 km (figS9_extract.py)
  ../new_full/fig6_cache_FULL/ and _TIDE/    slab caches of the paper
  ./fig2_out/fig2_wake.nc                    footprint, windows
  the history file of FULL                   grid metrics
  avg_TIDES_ONLY_V2_part*.nc                 time stamps only

Outputs (in OUTDIR)
  figS9_sampling.nc  figS9_sampling_report.txt

Run under nohup. One pass over the levels, CHUNK_Z at a time, about twice the
time of the first version.

  cd ~/dd_waves/paper_3/full_v5
  nohup python -u figS9_compute.py > log_figS9.txt 2>&1 &
  tail -f log_figS9.txt
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter

# --------------------------------------------------------------------------
# paths and settings
# --------------------------------------------------------------------------
ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
TIDE_FILES = 'avg_TIDES_ONLY_V2_part*.nc'           # for the time stamps of the tide records
NEW_FULL = '../new_full/'
SLAB = {'FULL': [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/'],
        'TIDE': [NEW_FULL + 'fig6_cache_TIDE/']}
CACHE1 = {'FULL': './figS9_cache_FULL/', 'TIDE': './figS9_cache_TIDE/'}
FIG2 = './fig2_out/fig2_wake.nc'
OUTDIR = './figS9_out/'
os.makedirs(OUTDIR, exist_ok=True)

T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2, as Figures 2 and 3
M2_BAND = (0.95, 1.05)                             # times omega_M2
STRIDES = [1, 2, 3, 4, 6]                          # points of the 1 km grid between samples
PAPER = 3                                          # the sampling of the paper
CHUNK_Z = 2
SMOOTH_H = 25
Q = ['sig', 'del', 'z', 'T']
QN = Q + ['P']                                     # P is the sum of the three works
VARIANTS = [f's{s}' for s in STRIDES] + ['avg3']
STEP = {**{f's{s}': s for s in STRIDES}, 'avg3': 3}
TESTS = ['paper', 'full']                          # test 1 and test 2
SOURCES = ['it', 'sm', 'full']                     # the anchor at 3 km

# --------------------------------------------------------------------------
# grid, the tile at 1 km, the footprint and the windows
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h, pm, pn = ref.h.values, ref.pm.values, ref.pn.values
hc, Vt = float(ref.hc.values), int(ref.Vtransform.values)
s_rho, s_w, Cs_r, Cs_w = ref.s_rho.values, ref.s_w.values, ref.Cs_r.values, ref.Cs_w.values
neta, nxi = h.shape


def z_of(s, C, hh):
    s, C = s[:, None, None], C[:, None, None]
    return hh*((hc*s + hh*C)/(hc + hh)) if Vt == 2 else hc*s + (hh - hc)*C


def to_hour(x):
    x = np.asarray(x).astype('datetime64[ns]')
    return (x + np.timedelta64(30, 'm')).astype('datetime64[h]').astype('datetime64[ns]')


g1 = {r: np.load(CACHE1[r] + 'geometry.npz', allow_pickle=True) for r in CACHE1}
for r in g1:
    assert bool(g1[r]['complete']), f'the 1 km cache of {r} is not complete, rerun figS9_extract.py {r}'
assert int(g1['FULL']['tile']) == int(g1['TIDE']['tile']) and np.array_equal(g1['FULL']['time'], g1['TIDE']['time']), \
    'the 1 km caches of FULL and TIDE are not on the same tile and records'
G = g1['FULL']
k_tile, SUB = int(G['tile']), int(G['SUB'])
assert SUB == PAPER, 'the slab caches are not on the sampling named in PAPER'
ys, xs, ys1, xs1 = G['ys'], G['xs'], G['ys1'], G['xs1']
t = to_hour(G['time'])
nt = t.size
dt_h = float((t[1] - t[0])/np.timedelta64(1, 'h'))
nyq = 0.5/dt_h
ny1, nx1 = ys1.size, xs1.size
jy, jx = np.minimum(ys1, neta - 1), np.minimum(xs1, nxi - 1)
h1, pm1, pn1 = h[np.ix_(jy, jx)], pm[np.ix_(jy, jx)], pn[np.ix_(jy, jx)]
area1 = 1.0/(pm1*pn1)
zr = np.moveaxis(z_of(s_rho, Cs_r, h1), 0, -1)                               # (ny, nx, nz)
dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h1), axis=0), 0, -1).astype(np.float32)
nz = zr.shape[-1]

slab = {r: next((d for d in SLAB[r] if os.path.exists(d + f'u_t{k_tile:03d}.npy')), None) for r in SLAB}
for r, d in slab.items():
    if d is None:
        raise SystemExit(f'no slab cache of {r} for tile {k_tile}')
geo = np.load(slab['FULL'] + 'geometry.npz', allow_pickle=True)
assert np.array_equal(to_hour(geo['time']), t), 'the 1 km cache and the slab cache are not on the same records'
b0, b1, c0, hlo = int(geo['b0'][k_tile]), int(geo['b1'][k_tile]), int(geo['c0'][k_tile]), int(geo['hlo'][k_tile])
f2 = xr.open_dataset(FIG2)
foot_s = f2.slope.values.astype(bool) & (f2.sector_code.values == 0)
ft = np.zeros((ys.size, xs.size), bool)                                       # footprint on the slab tile
ft[hlo:hlo + (b1 - b0), :] = foot_s[b0:b1, c0:c0 + xs.size]
iy = np.minimum((np.arange(ny1) + SUB//2)//SUB, ys.size - 1)                  # nearest slab point of each 1 km point
ix = np.minimum((np.arange(nx1) + SUB//2)//SUB, xs.size - 1)
reg1 = ft[np.ix_(iy, ix)]
MARGIN = max(STRIDES)
reg1[:MARGIN], reg1[-MARGIN:], reg1[:, :MARGIN], reg1[:, -MARGIN:] = False, False, False, False
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
print(f'tile {k_tile}, 1 km block {ny1} x {nx1}, {nz} levels, {nt} records, region {int(reg1.sum())} points '
      f'({area1[reg1].sum()/1e6:.0f} km2), work window {T_STORM0} to {T_WAKE1}', flush=True)
if reg1.sum() == 0:
    raise SystemExit('the tile holds no footprint point away from its edge')

# --------------------------------------------------------------------------
# the time stamps of the tide records. A record that is not centred on the
# hour is replaced by the linear interpolation to the hour between itself and
# the record before it
# --------------------------------------------------------------------------
stamp = {}
for f in sorted(glob.glob(ROOT + TIDE_FILES)):
    raw = xr.open_dataset(f).ocean_time.values.astype('datetime64[ns]')
    for x in raw:
        stamp.setdefault(to_hour(x).item(), x)                                # the first part that holds the hour
if all(x.item() in stamp for x in t):
    t_true = np.array([stamp[x.item()] for x in t])
    off_h = (t_true - t)/np.timedelta64(1, 's')/3600.0                         # hours, stamp minus the hour
else:
    print('the time stamps of the tide files were not found, the records are taken as centred on the hour',
          flush=True)
    off_h = np.zeros(nt)
tt_h = np.arange(nt)*dt_h + off_h                                              # true time of each record, hours
W_T = np.ones(nt)                                                              # weight of the record itself
W_T[1:] = (np.arange(1, nt)*dt_h - tt_h[:-1])/(tt_h[1:] - tt_h[:-1])
W_T = np.clip(W_T, 0.0, 1.0)
SHIFTED = np.where(np.abs(off_h) > 0.01)[0]
if SHIFTED.size:
    print(f'tide records off the hour: {SHIFTED.size} of {nt}, from {str(t[SHIFTED[0]])[:13]} to '
          f'{str(t[SHIFTED[-1]])[:13]}, by {np.median(off_h[SHIFTED])*60:+.0f} min at the median. '
          f'They are interpolated to the hour', flush=True)
else:
    print('every tide record is centred on the hour', flush=True)


def on_the_hour(x):
    """Linear interpolation in time of the tide records to the hour (axis 0)."""
    if SHIFTED.size == 0:
        return x
    out = x.copy()
    k = np.where(W_T < 1.0 - 1e-6)[0]
    out[k] = x[k - 1] + W_T[k].reshape((-1,) + (1,)*(x.ndim - 1)).astype(np.float32)*(x[k] - x[k - 1])
    return out


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


half = 0.5*OMEGA_M2*RAD2CPH
SOS = {'ni': band_sos(NI_BAND[0]*half, NI_BAND[1]*half),
       'M2': band_sos(M2_BAND[0]*2*half, M2_BAND[1]*2*half)}


def bp(x, name):
    sos, pad = SOS[name]
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def d_xi(a, dxi):
    out = np.full_like(a, np.nan)
    out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2])/(2*dxi[None, :, 1:-1, None])
    return out


def d_eta(a, deta):
    out = np.full_like(a, np.nan)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2])/(2*deta[None, 1:-1, :, None])
    return out


def d_z(a, z):
    out = np.empty_like(a)
    zz = z[None]
    out[..., 1:-1] = (a[..., 2:] - a[..., :-2])/(zz[..., 2:] - zz[..., :-2])
    out[..., 0] = (a[..., 1] - a[..., 0])/(zz[..., 1] - zz[..., 0])
    out[..., -1] = (a[..., -1] - a[..., -2])/(zz[..., -1] - zz[..., -2])
    return out


def load1(run, var, z0, z1):
    a = np.load(CACHE1[run] + f'{var}.npy', mmap_mode='r')
    x = np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))
    return on_the_hour(x) if run == 'TIDE' else x


def load3(run, var, z0, z1):
    """Levels z0 to z1 of a slab variable. A variable held on the w levels is averaged to the rho levels."""
    a = np.load(slab[run] + f'{var}_t{k_tile:03d}.npy', mmap_mode='r')
    if a.shape[-1] == s_w.size:
        x = np.asarray(a[..., z0:z1 + 1], dtype=np.float32)
        return np.nan_to_num(0.5*(x[..., 1:] + x[..., :-1]))
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


def sample(x, var):
    """A field on the grid of a variant. The block mean is taken over 3 x 3 points
    centred on the points of the paper."""
    s = STEP[var]
    if var == 'avg3':
        x = uniform_filter(x, size=(1, 3, 3, 1) if x.ndim == 4 else (3, 3), mode='nearest')
    return x[:, ::s, ::s] if x.ndim == 4 else x[::s, ::s]


def works(a_u, a_v, a_w, b_u, b_v, az_u, az_v, bz_u, bz_v, dxi, deta):
    """The three works of the flow b on the wake a and the tide side of the triad, per unit
    volume, as in fig_03_compute.py. Also the strain variance of b and the squared speed of a."""
    ux, uy, vx, vy = d_xi(b_u, dxi), d_eta(b_u, deta), d_xi(b_v, dxi), d_eta(b_v, deta)
    sn, ss, dl = ux - vy, vx + uy, ux + vy
    q1, q2, q0 = a_u*a_u - a_v*a_v, 2*a_u*a_v, a_u*a_u + a_v*a_v
    adv_u = a_u*d_xi(a_u, dxi) + a_v*d_eta(a_u, deta) + a_w*az_u
    adv_v = a_u*d_xi(a_v, dxi) + a_v*d_eta(a_v, deta) + a_w*az_v
    return ({'sig': -0.5*RHO0*(sn*q1 + ss*q2), 'del': -0.5*RHO0*dl*q0, 'z': -RHO0*a_w*(a_u*bz_u + a_v*bz_v),
             'T': -RHO0*(b_u*adv_u + b_v*adv_v)}, sn*sn + ss*ss, q0)


# check, every third point of the 1 km cache of FULL against the slab cache of FULL
rec = np.unique(np.linspace(0, nt - 1, 8).astype(int))
chk = {}
for var in ('u', 'v', 'w'):
    a = np.load(CACHE1['FULL'] + f'{var}.npy', mmap_mode='r')
    c = np.load(slab['FULL'] + f'{var}_t{k_tile:03d}.npy', mmap_mode='r')
    chk[var] = float(np.nanmax(np.abs(np.asarray(a[rec][:, ::SUB, ::SUB]) - np.asarray(c[rec]))))
print('check, 1 km cache of FULL against the slab cache of FULL, largest difference '
      + ', '.join(f'{v} {chk[v]:.2e}' for v in chk) + ' m s-1', flush=True)
if max(chk.values()) > 1e-6:
    raise SystemExit('the 1 km cache of FULL does not reproduce the slab cache of the paper, stop')

# --------------------------------------------------------------------------
# the pass over the levels
# --------------------------------------------------------------------------
grid = {v: (sample(pm1, f's{STEP[v]}'), sample(pn1, f's{STEP[v]}')) for v in VARIANTS}
shape = {v: grid[v][0].shape for v in VARIANTS}
P = {(k, v, q): np.zeros((nt,) + shape[v], np.float32) for k in TESTS for v in VARIANTS for q in Q}
s2 = {(k, v): np.zeros(shape[v]) for k in TESTS for v in VARIANTS}            # strain variance of the tide side
ke = {(k, v): np.zeros(shape[v]) for k in TESTS for v in VARIANTS}            # J m-2, kinetic energy of the wake side
g3 = grid[f's{PAPER}']
B = {(s, q): np.zeros((nt,) + g3[0].shape, np.float32) for s in SOURCES for q in Q}   # the anchor at 3 km
dz3, zr3 = dz[::PAPER, ::PAPER], zr[::PAPER, ::PAPER]
m3 = reg1[::PAPER, ::PAPER]
tide_num, tide_den, tide_xy = 0.0, 0.0, 0.0                                    # the repeated tide run against the paper

t0 = pd.Timestamp.now()
chunks = list(range(0, nz, CHUNK_Z))
for n, z0 in enumerate(chunks):
    z1 = min(z0 + CHUNK_Z, nz)
    z0e, z1e = max(z0 - 1, 0), min(z1 + 1, nz)                                 # one level of overlap
    sl = slice(z0 - z0e, z0 - z0e + (z1 - z0))
    zc_e = zr[..., z0e:z1e]

    # the fields at 1 km
    uF, vF, wF = load1('FULL', 'u', z0e, z1e), load1('FULL', 'v', z0e, z1e), load1('FULL', 'w', z0e, z1e)
    uT, vT, wT = load1('TIDE', 'u', z0e, z1e), load1('TIDE', 'v', z0e, z1e), load1('TIDE', 'w', z0e, z1e)
    fields = {'paper': dict(au=bp(uF - uT, 'ni'), av=bp(vF - vT, 'ni'), aw=bp(wF - wT, 'ni'),
                            bu=bp(uT, 'M2'), bv=bp(vT, 'M2')),
              'full': dict(au=bp(uF, 'ni'), av=bp(vF, 'ni'), aw=bp(wF, 'ni'), bu=bp(uF, 'M2'), bv=bp(vF, 'M2'))}
    del uF, vF, wF, uT, vT, wT
    for k in TESTS:
        fk = fields[k]
        az_u, az_v = d_z(fk['au'], zc_e)[..., sl], d_z(fk['av'], zc_e)[..., sl]
        bz_u, bz_v = d_z(fk['bu'], zc_e)[..., sl], d_z(fk['bv'], zc_e)[..., sl]
        au, av, aw, bu, bv = (fk[x][..., sl] for x in ('au', 'av', 'aw', 'bu', 'bv'))
        for v in VARIANTS:
            s = STEP[v]
            dzc = sample(dz[..., z0:z1][None], v)[0]
            term, sv2, q0 = works(sample(au, v), sample(av, v), sample(aw, v), sample(bu, v), sample(bv, v),
                                  sample(az_u, v), sample(az_v, v), sample(bz_u, v), sample(bz_v, v),
                                  s/grid[v][0], s/grid[v][1])
            for q in Q:
                P[(k, v, q)] += np.nan_to_num((term[q]*dzc).sum(axis=-1))
            s2[(k, v)] += np.nan_to_num((sv2[WORK]*dzc).sum(axis=-1).mean(axis=0))
            ke[(k, v)] += (0.5*RHO0*q0[WORK]*dzc).sum(axis=-1).mean(axis=0)
            del term, sv2, q0
        del az_u, az_v, bz_u, bz_v, au, av, aw, bu, bv
    tide1 = (fields['paper']['bu'][:, ::PAPER, ::PAPER][..., sl], fields['paper']['bv'][:, ::PAPER, ::PAPER][..., sl])
    del fields

    # the anchor at 3 km, from the slab caches of the paper
    zc3, dzc3 = zr3[..., z0e:z1e], dz3[..., z0:z1]
    u3 = {r: load3(r, 'u', z0e, z1e) for r in ('FULL', 'TIDE')}
    v3 = {r: load3(r, 'v', z0e, z1e) for r in ('FULL', 'TIDE')}
    w3 = {r: load3(r, 'w', z0e, z1e) for r in ('FULL', 'TIDE')}
    wake = (bp(u3['FULL'] - u3['TIDE'], 'ni'), bp(v3['FULL'] - v3['TIDE'], 'ni'), bp(w3['FULL'] - w3['TIDE'], 'ni'))
    alone = (bp(u3['FULL'], 'ni'), bp(v3['FULL'], 'ni'), bp(w3['FULL'], 'ni'))
    src = {'it': (wake, bp(u3['TIDE'], 'M2'), bp(v3['TIDE'], 'M2')),
           'sm': (wake, bp(u3['FULL'] - u3['TIDE'], 'M2'), bp(v3['FULL'] - v3['TIDE'], 'M2')),
           'full': (alone, bp(u3['FULL'], 'M2'), bp(v3['FULL'], 'M2'))}
    for s_, ((a_u, a_v, a_w), b_u, b_v) in src.items():
        term, _, _ = works(a_u[..., sl], a_v[..., sl], a_w[..., sl], b_u[..., sl], b_v[..., sl],
                           d_z(a_u, zc3)[..., sl], d_z(a_v, zc3)[..., sl], d_z(b_u, zc3)[..., sl],
                           d_z(b_v, zc3)[..., sl], PAPER/g3[0], PAPER/g3[1])
        for q in Q:
            B[(s_, q)] += np.nan_to_num((term[q]*dzc3).sum(axis=-1))
        del term
    # the semidiurnal band of the repeated tide run against that of the paper, work window, region
    for x1, x3 in ((tide1[0], src['it'][1][..., sl]), (tide1[1], src['it'][2][..., sl])):
        a_, c_ = x1[WORK][:, m3].astype(np.float64), x3[WORK][:, m3].astype(np.float64)
        tide_xy, tide_num, tide_den = tide_xy + np.sum(a_*c_), tide_num + np.sum(a_*a_), tide_den + np.sum(c_*c_)
    del u3, v3, w3, wake, alone, src, tide1

    el = (pd.Timestamp.now() - t0).total_seconds()
    left = el/(n + 1)*(len(chunks) - n - 1)
    print(f'levels {z0 + 1} to {z1} of {nz}, {el/60:.1f} min, about {left/60:.0f} min left, finish near '
          f'{(pd.Timestamp.now() + pd.Timedelta(seconds=left)).strftime("%d %b %H:%M")}', flush=True)

# --------------------------------------------------------------------------
# means over the region, accumulated works and the factors of the rate
# --------------------------------------------------------------------------
depth = dz.sum(axis=-1)
series = np.zeros((len(TESTS), len(VARIANTS), len(QN), nt))                   # W m-2
strain_rms, ke_mean = np.zeros((len(TESTS), len(VARIANTS))), np.zeros((len(TESTS), len(VARIANTS)))
npts = np.zeros(len(VARIANTS), int)
for i, v in enumerate(VARIANTS):
    s = STEP[v]
    m = reg1[::s, ::s]
    wgt = area1[::s, ::s][m]
    wgt = wgt/wgt.sum()
    npts[i] = int(m.sum())
    dep = np.maximum(depth[::s, ::s], 1e-6)
    for c, k in enumerate(TESTS):
        for j, q in enumerate(Q):
            series[c, i, j] = (P[(k, v, q)][:, m]*wgt).sum(axis=1)
        series[c, i, len(Q)] = series[c, i, 0] + series[c, i, 1] + series[c, i, 2]
        strain_rms[c, i] = float(np.sqrt(np.sum((s2[(k, v)]/dep)[m]*wgt)))
        ke_mean[c, i] = float(np.sum(ke[(k, v)][m]*wgt))
J = series[..., WORK].sum(axis=-1)*dt_h*3600.0                                # J m-2 over the work window
after = (t >= T_OFF) & WORK
J_after = series[..., after].sum(axis=-1)*dt_h*3600.0
i1, i3 = VARIANTS.index('s1'), VARIANTS.index(f's{PAPER}')
corr = np.array([[[np.corrcoef(series[c, i, j][WORK], series[c, i1, j][WORK])[0, 1] for j in range(len(QN))]
                  for i in range(len(VARIANTS))] for c in range(len(TESTS))])

w3n = area1[::PAPER, ::PAPER][m3]
w3n = w3n/w3n.sum()
anchor = np.zeros((len(SOURCES), len(QN), nt))
for i, s_ in enumerate(SOURCES):
    for j, q in enumerate(Q):
        anchor[i, j] = (B[(s_, q)][:, m3]*w3n).sum(axis=1)
    anchor[i, len(Q)] = anchor[i, 0] + anchor[i, 1] + anchor[i, 2]
Ja = anchor[:, :, WORK].sum(axis=2)*dt_h*3600.0
k_it, k_sm, k_fu = (SOURCES.index(x) for x in ('it', 'sm', 'full'))
cP, cF = TESTS.index('paper'), TESTS.index('full')
jS, jP, jT = QN.index('sig'), QN.index('P'), QN.index('T')
chk_s3 = float(np.max(np.abs(anchor[k_fu] - series[cF, i3])))                 # W m-2, must vanish
tide_corr = float(tide_xy/np.sqrt(tide_num*tide_den))                         # repeated tide against the paper's
tide_amp = float(np.sqrt(tide_num/tide_den))
rep = {q: (J[cP, i3, j]/Ja[k_it, j] if Ja[k_it, j] != 0 else np.nan) for j, q in enumerate(QN)}
rep_corr = {q: float(np.corrcoef(series[cP, i3, j][WORK], anchor[k_it, j][WORK])[0, 1]) for j, q in enumerate(QN)}

# pattern of the accumulated strain work, the points of the paper against the same points at 1 km
A1 = (P[('paper', 's1', 'sig')][WORK].sum(axis=0)*dt_h*3600.0)[::PAPER, ::PAPER][m3]
A3 = (P[('paper', f's{PAPER}', 'sig')][WORK].sum(axis=0)*dt_h*3600.0)[m3]
pattern = float(np.corrcoef(A1, A3)[0, 1]) if m3.sum() > 2 else np.nan

# the ratio of 1 km to 3 km and the work of the paper carried to 1 km
r13 = {k: {q: (J[c, i1, j]/J[c, i3, j] if J[c, i3, j] != 0 else np.nan) for j, q in enumerate(QN)}
       for c, k in enumerate(TESTS)}

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars={'series': (('test', 'variant', 'term', 'time'), series), 'J': (('test', 'variant', 'term'), J),
               'J_after': (('test', 'variant', 'term'), J_after),
               'corr_with_1km': (('test', 'variant', 'term'), corr),
               'strain_rms': (('test', 'variant'), strain_rms), 'ke': (('test', 'variant'), ke_mean),
               'spacing_km': (('variant',), np.array([STEP[v] for v in VARIANTS], float)),
               'npts': (('variant',), npts),
               'anchor_series': (('source', 'term', 'time'), anchor), 'anchor_J': (('source', 'term'), Ja)},
    coords=dict(test=TESTS, variant=VARIANTS, term=QN, time=t, source=SOURCES),
    attrs=dict(tile=k_tile, windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(f2.attrs['t_off']),
               paper_variant=f's{PAPER}', region_km2=float(area1[reg1].sum()/1e6), pattern_corr_sig=pattern,
               smooth_h=SMOOTH_H, check_u=chk['u'], check_v=chk['v'], check_w=chk['w'], check_s3=chk_s3,
               tide_corr=tide_corr, tide_amp=tide_amp, repeat_over_paper_P=rep['P'], repeat_over_paper_sig=rep['sig'],
               repeat_corr_sig=rep_corr['sig'], records_off_the_hour=int(SHIFTED.size)))
ds.to_netcdf(OUTDIR + 'figS9_sampling.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
HEAD = ' '.join(f'{c:>19s}' for c in ('strain', 'divergence', 'vertical shear', 'tide side', 'sum of the works'))


def table(c):
    out = ['   variant  spacing  points ' + HEAD]
    for i, v in enumerate(VARIANTS):
        out.append(f'   {v:7s} {STEP[v]:5d} km {npts[i]:6d} '
                   + ' '.join((f'{J[c, i, j]:+11.2f} ({J[c, i, j]/J[c, i1, j]:5.2f})' if J[c, i1, j] != 0
                               else f'{J[c, i, j]:+11.2f} (  nan)') for j in range(len(QN))))
    return out


L = ['Supplementary Figure S9. Is the 3 km sampling of the 1 km model enough for the works of the paper (revision 3).',
     f'Tile {k_tile} at 1 km, {ny1} x {nx1} points, region {int(reg1.sum())} points ({area1[reg1].sum()/1e6:.0f} km2), '
     f'work window {T_STORM0} to {T_WAKE1}.',
     f'CHECK 1. Every third point of the 1 km cache of FULL against the slab cache of the paper, largest difference '
     f'u {chk["u"]:.2e}, v {chk["v"]:.2e}, w {chk["w"]:.2e} m s-1.',
     f'CHECK 2. The works of FULL alone from the slab cache against variant s3 of test 2, largest difference of the '
     f'hourly series {chk_s3:.2e} W m-2 (rounding if the two agree).',
     f'Tide records interpolated to the hour: {SHIFTED.size} of {nt}.', '',
     '1. THE REPEATED TIDE RUN AGAINST THE TIDE RUN OF THE PAPER, at 3 km on this tile',
     f'   semidiurnal band of the two runs over the work window: correlation {tide_corr:.3f}, amplitude ratio '
     f'{tide_amp:.3f}',
     '   works over the work window (J m-2)                ' + HEAD,
     '   tide run of the paper (slab caches)              ' + ' '.join(f'{Ja[k_it, j]:+19.2f}' for j in range(len(QN))),
     '   repeated tide run, variant s3                    '
     + ' '.join(f'{J[cP, i3, j]:+19.2f}' for j in range(len(QN))),
     '   repeated over paper                              ' + ' '.join(f'{rep[q]:19.2f}' for q in QN),
     '   correlation of the hourly series                 ' + ' '.join(f'{rep_corr[q]:19.3f}' for q in QN),
     '   (for reference, storm-made motion of the paper   '
     + ' '.join(f'{Ja[k_sm, j]:+19.2f}' for j in range(len(QN))) + ')', '',
     '2. TEST 1, THE QUANTITY OF THE PAPER AGAINST THE SPACING. Internal tide on the wake FULL - TIDE,',
     '   accumulated works over the work window (J m-2) and ratio to 1 km'] + table(cP)
L += [f'   the sampling of the paper is s{PAPER}', '',
      '3. TEST 2, FULL ALONE AGAINST THE SPACING. Semidiurnal band of FULL on its near-inertial band'] + table(cF)
L += ['', '4. AFTER THE WIND STOPS (J m-2), strain, divergence, vertical shear, tide side, sum of the works']
for c, k in enumerate(TESTS):
    for i, v in enumerate(VARIANTS):
        L.append(f'   test {c + 1} {v:7s} ' + ' '.join(f'{J_after[c, i, j]:+11.2f}' for j in range(len(QN))))
L += ['', '5. CORRELATION OF THE HOURLY SERIES WITH THE 1 km SERIES OVER THE WORK WINDOW',
      '   test    variant ' + ' '.join(f'{q:>8s}' for q in QN)]
for c, k in enumerate(TESTS):
    for i, v in enumerate(VARIANTS):
        L.append(f'   test {c + 1}  {v:7s} ' + ' '.join(f'{corr[c, i, j]:8.3f}' for j in range(len(QN))))
L += ['', '6. THE FACTORS OF THE RATE. Depth-mean rms strain of the tide side (1e-6 s-1), kinetic energy of the wake',
      '   side (J m-2), each with its ratio to 1 km, and -T/P',
      '   test    variant   rms strain  ratio   kinetic energy  ratio    -T/P']
for c, k in enumerate(TESTS):
    for i, v in enumerate(VARIANTS):
        L.append(f'   test {c + 1}  {v:7s} {strain_rms[c, i]*1e6:10.3f} {strain_rms[c, i]/strain_rms[c, i1]:7.2f} '
                 f'{ke_mean[c, i]:14.1f} {ke_mean[c, i]/ke_mean[c, i1]:7.2f} '
                 f'{-J[c, i, jT]/J[c, i, jP] if J[c, i, jP] != 0 else np.nan:8.2f}')
L += ['', f'7. PATTERN. Accumulated strain work of test 1 at the points of the paper against the same points at 1 km, '
      f'correlation {pattern:.3f}', '',
      '8. THE ANSWER IN ONE LINE',
      f'   3 km over 1 km, sum of the works: test 1 {1/r13["paper"]["P"]:.2f}, test 2 {1/r13["full"]["P"]:.2f}. '
      f'Strain work: test 1 {1/r13["paper"]["sig"]:.2f}, test 2 {1/r13["full"]["sig"]:.2f}.',
      f'   work of the internal tide of the paper on this tile at 3 km {Ja[k_it, jP]:+.1f} J m-2, carried to 1 km with '
      f'the ratio of test 1 {Ja[k_it, jP]*r13["paper"]["P"]:+.1f} J m-2', '',
      'READING. Section 1 says how close the repeated tide run is to the paper at the sampling of the paper.',
      'Sections 2 and 3 are the result. A ratio of s3 near one in both tests says the works do not depend on the',
      'sampling. If s3 departs from s1 and avg3 does not, the difference is aliasing of scales under 6 km. If both',
      'depart, the tide holds strain under 6 km that the centred difference of the paper does not see, and Section 8',
      'says by how much the work of the paper would change.']
open(OUTDIR + 'figS9_sampling_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('saved', OUTDIR + 'figS9_sampling.nc')