"""
fig_03_compute.py  (full_v5, revision 3)
----------------------------------------
Figure 3. The work of the tidal strain on the wake, from the momentum equation
of the near-inertial band inside the joint experiment.

The wake with the tide is the near-inertial band of u_FULL - u_TIDE (Figure 2).
The semidiurnal band of FULL is split exactly into two sources,

    u_M2(FULL) = u_M2(TIDE) + u_M2(FULL - TIDE),

the internal tide carried by the same eddies without the storm, and the
semidiurnal motion the storm makes inside the pair (the 2f harmonic of the
wake, direct forcing, and the change the storm makes to the tide). Revision 1
used u_M2(FULL - WIND + BACK), which adds the harmonic of the other wake with
an unrelated phase. Its strain work (-311 J m-2) and that of the internal tide
(+390 J m-2) had opposite signs and uncorrelated maps (rank correlation 0.04),
so the strain of the revision 1 tide was set by storm-made motion at the
scales of the wake. Revision 1 is reproduced here as the sum
W(TIDE) + W(FULL - TIDE) - W(WIND - BACK).

Kinetic energy transfer into the wake band, per unit volume, is the production
term of the filtered budget (McComas and Bretherton 1977, Young et al. 2008,
MacKinnon et al. 2013, Barkan et al. 2021, Wang et al. 2025),

    P = - rho0 a_i a_j db_i/dx_j ,     a the wake, b the source, i horizontal,

split into the strain work -(rho0/2)[sigma_n (u^2 - v^2) + 2 sigma_s u v], the
divergence work -(rho0/2) delta (u^2 + v^2) and the vertical shear work
-rho0 w (u du_b/dz + v dv_b/dz). The tide side of the same triad is

    T = - rho0 b_i a_j da_i/dx_j ,

and P + T = - rho0 d(a_i a_j b_i)/dx_j, a flux divergence, so the internal
tide loses over the footprint what the wake gains from it, up to the flux
through the footprint boundary. Revision 1 formed the tide side with the wrong
pair of factors (-rho0 b_i b_j da_i/dx_j, a triad with no resonant mean).

Tests
  the budget of the gain. The works of the internal tide and of the storm-made
    motion and the near-inertial wind work difference between the pairs,
    against the tendency of (wake with - wake without).
  the triad. T of the internal tide against its P, footprint and blocks.
  the phase null. The internal tide shifted by -48 to +48 h against the wake
    with the tide and against the wake without it (u_WIND - u_BACK, which never
    met the tide). The second is the work a wake and a tide do with no
    feedback on each other.
  dose and response. The gain of each 24 km block against the strain work of
    the internal tide on it, Spearman against a circular-shift level, and the
    partial correlation holding the wind work difference and the wake kinetic
    energy fixed.
  bilinearity. The strain work against the strain amplitude times the wake
    kinetic energy, Theil and Sen slope (one in the theory).
  the wind work route. The wind work difference of each block against the
    difference of detuning between FULL and WIND in the storm window, and the
    ratio of wind work with over without the tide where the two flows have
    the same detuning. A ratio that stays above one there belongs to the tide
    and not to the eddies of the two pairs.

Inputs
  ../new_full/fig6_cache_<RUN>/     u, v and w on the 3 km slope tiles, four runs
  ./fig2_out/fig2_wake.nc           wake maps, wind work, series, geometry
                                    (revision 4 of fig_02_compute.py)

Outputs (in OUTDIR)
  fig3_strain.nc  fig3_blocks.csv  fig3_strain_report.txt

Run under nohup. A short pre-pass for the storm-window vorticity, then one pass
over the tiles, CHUNK_Z levels at a time.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, theilslopes, rankdata

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
OUTDIR = './fig3_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2, as Figure 2
M2_BAND = (0.95, 1.05)                             # times omega_M2
Z_UPPER = -150.0                                   # the layer of the balanced vorticity
L_MESO = 25.0e3
CHUNK_Z = 4
SMOOTH_H = 25
SHIFTS_H = np.arange(-48, 49)                      # the tide against the wake, hours
Z_EDGES = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000.0])
N_BOOT = 2000
N_DOSE = 8                                         # bins of accumulated work (quantiles)
MIN_BLOCKS = 6
rng = np.random.default_rng(11)

# the works, each accumulated over depth
Q = ['sig_it', 'del_it', 'z_it',                   # internal tide on the wake with the tide
     'sig_sm', 'del_sm', 'z_sm',                   # storm-made semidiurnal motion of the pair
     'sig_ot',                                     # storm-made motion of the other pair (revision 1)
     'sig_kin',                                    # internal tide on the wake without the tide
     'T_it']                                       # tide side of the triad, internal tide
NAMES = {'sig_it': 'strain work of the internal tide', 'del_it': 'divergence work of the internal tide',
         'z_it': 'vertical shear work of the internal tide',
         'sig_sm': 'strain work of the storm-made motion', 'del_sm': 'divergence work of the storm-made motion',
         'z_sm': 'vertical shear work of the storm-made motion',
         'sig_ot': 'strain work of the storm-made motion of WIND - BACK',
         'sig_kin': 'strain work of the internal tide on the wake without the tide',
         'T_it': 'tide side of the triad (internal tide)'}

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
dt_h = float((t[1] - t[0])/np.timedelta64(1, 'h'))
nyq = 0.5/dt_h
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
HAVE_W = all(os.path.exists(cache[r] + 'w_t000.npy') for r in ('FULL', 'TIDE'))
print(f'{ntiles} tiles, {nt} records, sub grid {ny_s} x {nx_s}, vertical velocity '
      f'{"cached" if HAVE_W else "not cached, the vertical terms are skipped"}', flush=True)
f_s = f_rho[np.ix_(np.minimum(eta_s, neta - 1), np.minimum(xi_s, nxi - 1))]

f2 = xr.open_dataset(FIG2)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s and f2.sizes['time'] == nt, \
    'fig2_wake.nc is not on the slab geometry, rerun fig_02_compute.py'
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
foot = slope_s & (sector_s == 0)
dc_full = f2.dc_FULL.values*RAD2CPH                   # cph, wake window
CLASS = {'free': foot & (dc_full < 0), 'trapped': foot & (dc_full > 0)}
Ew = {p: f2[f'Ewni_{p}_wake'].values for p in ('tide', 'notide')}
gain_map = Ew['tide'] - Ew['notide']                  # J m-2, wake-window mean
work_pair = {p: f2[f'work_{p}'].values for p in ('tide', 'notide')}
wind_diff = work_pair['tide'] - work_pair['notide']
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
STORM = (t >= np.datetime64(WINDOWS['storm'][0])) & (t < np.datetime64(WINDOWS['storm'][1]))
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
foot_area = float(area_s[foot].sum())
print(f'footprint {foot_area/1e6:.0f} km2, work window {T_STORM0} to {T_WAKE1}', flush=True)


def band_sos(lo_cph, hi_cph):
    sos = butter(4, [lo_cph/nyq, hi_cph/nyq], btype='band', output='sos')
    return sos, int(min(3.0/((hi_cph - lo_cph)*dt_h), nt - 1))


half = 0.5*OMEGA_M2*RAD2CPH
SOS = {'ni': band_sos(NI_BAND[0]*half, NI_BAND[1]*half),
       'M2': band_sos(M2_BAND[0]*2*half, M2_BAND[1]*2*half)}


def bp(x, name):
    sos, pad = SOS[name]
    return sosfiltfilt(sos, x, axis=0, padlen=pad).astype(np.float32)


def tile_geometry(k):
    ys, xs = geo[f'ys{k:03d}'], geo[f'xs{k:03d}']
    b0, b1, c0, hlo = int(geo['b0'][k]), int(geo['b1'][k]), int(geo['c0'][k]), int(geo['hlo'][k])
    jy, jx = np.minimum(ys, neta - 1), np.minimum(xs, nxi - 1)
    return ys, xs, b0, b1, c0, slice(hlo, hlo + (b1 - b0)), slice(c0, c0 + xs.size), jy, jx


def d_xi(a, dxi):
    out = np.full_like(a, np.nan)
    out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2])/(2*dxi[None, :, 1:-1, None])
    return out


def d_eta(a, deta):
    out = np.full_like(a, np.nan)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2])/(2*deta[None, 1:-1, :, None])
    return out


def d_z(a, z):
    """Centred difference along the level axis on the level depths z (ny, nx, nz),
    one-sided at the ends."""
    out = np.empty_like(a)
    zz = z[None]
    out[..., 1:-1] = (a[..., 2:] - a[..., :-2])/(zz[..., 2:] - zz[..., :-2])
    out[..., 0] = (a[..., 1] - a[..., 0])/(zz[..., 1] - zz[..., 0])
    out[..., -1] = (a[..., -1] - a[..., -2])/(zz[..., -1] - zz[..., -2])
    return out


def load(run, var, k, z0, z1):
    """Levels z0 to z1 of a cached variable. A variable held on the w levels is
    averaged to the rho levels."""
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    if a.shape[-1] == s_w.size:
        x = np.asarray(a[..., z0:z1 + 1], dtype=np.float32)
        return np.nan_to_num(0.5*(x[..., 1:] + x[..., :-1]))
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


# --------------------------------------------------------------------------
# pre-pass. Detuning of FULL and WIND in the storm window, from the mean velocity
# of the upper 150 m coarse-grained to 25 km as in Figure 2, for the test of
# the wind work route
# --------------------------------------------------------------------------
zeta_s = {r: np.full((ny_s, nx_s), np.nan, np.float32) for r in ('FULL', 'WIND')}
ks = np.where(STORM)[0]
t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = z_of(s_rho, Cs_r, h_t)
    w150 = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0)*(zr > Z_UPPER), 0, -1)
    w150 = (w150/np.maximum(w150.sum(axis=-1, keepdims=True), 1e-6)).astype(np.float32)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    for run in zeta_s:
        u = np.load(cache[run] + f'u_t{k:03d}.npy', mmap_mode='r')
        v = np.load(cache[run] + f'v_t{k:03d}.npy', mmap_mode='r')
        Um = (np.nan_to_num(np.asarray(u[ks[0]:ks[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
        Vm = (np.nan_to_num(np.asarray(v[ks[0]:ks[-1] + 1], dtype=np.float32)).mean(axis=0)*w150).sum(-1)
        zt = np.full(Um.shape, np.nan, np.float32)
        zt[1:-1, 1:-1] = ((Vm[1:-1, 2:] - Vm[1:-1, :-2])/(2*dxi[1:-1, 1:-1])
                          - (Um[2:, 1:-1] - Um[:-2, 1:-1])/(2*deta[1:-1, 1:-1]))
        zeta_s[run][b0:b1, cols] = zt[rows]
    print(f'pre-pass tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)
sig_px = 0.5*L_MESO/float(np.nanmean(SUB/pm))
dc_storm = {}
for run, z_ in zeta_s.items():
    g = np.isfinite(z_)
    num = gaussian_filter(np.where(g, z_, 0.0), sig_px, mode='nearest')
    den = gaussian_filter(g.astype(float), sig_px, mode='nearest')
    zs = np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)
    dc_storm[run] = (np.abs(f_s + 0.5*zs) - 0.5*OMEGA_M2)*RAD2CPH             # cph

# --------------------------------------------------------------------------
# the pass over the tiles
# --------------------------------------------------------------------------
Pint = {q: np.full((ny_s, nx_s), np.nan, np.float32) for q in Q}             # J m-2 over WORK
strain_amp = np.full((ny_s, nx_s), np.nan, np.float32)                       # s-1, internal tide
series = {q: np.zeros(nt) for q in Q}                                         # W, footprint sum
null_with = np.zeros((SHIFTS_H.size, nt))                                     # W, internal tide shifted
null_without = np.zeros((SHIFTS_H.size, nt))
PROF = ['sig_it', 'del_it', 'z_it', 'sm', 'T_it']
nzb = Z_EDGES.size - 1
prof_num = {(q, c): np.zeros(nzb) for q in PROF for c in CLASS}
prof_den = {c: np.zeros(nzb) for c in CLASS}


def strain_parts(bu, bv, dxi, deta):
    """Normal strain, shear strain and divergence of a horizontal field."""
    ux, uy, vx, vy = d_xi(bu, dxi), d_eta(bu, deta), d_xi(bv, dxi), d_eta(bv, deta)
    return ux - vy, vx + uy, ux + vy


t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = np.moveaxis(z_of(s_rho, Cs_r, h_t), 0, -1)                           # (ny, nx, nz)
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    nz = zr.shape[-1]
    ny_t, nx_t = h_t.shape
    ar_t = np.zeros((ny_t, nx_t))
    ar_t[rows, :] = area_s[b0:b1, cols]
    ft = np.zeros((ny_t, nx_t), bool)
    ft[rows, :] = foot[b0:b1, cols]
    cls_t = {c: np.zeros((ny_t, nx_t), bool) for c in CLASS}
    for c in CLASS:
        cls_t[c][rows, :] = CLASS[c][b0:b1, cols]

    P = {q: np.zeros((nt, ny_t, nx_t), np.float32) for q in Q}
    s2 = np.zeros((ny_t, nx_t), np.float64)
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        z0e, z1e = max(z0 - 1, 0), min(z1 + 1, nz)                             # one level of overlap
        sl = slice(z0 - z0e, z0 - z0e + (z1 - z0))
        zc_e = zr[..., z0e:z1e]
        dzc = dz[..., z0:z1]
        u = {r: load(r, 'u', k, z0e, z1e) for r in RUNS}
        v = {r: load(r, 'v', k, z0e, z1e) for r in RUNS}
        # the two wakes and the three semidiurnal sources
        au, av = bp(u['FULL'] - u['TIDE'], 'ni'), bp(v['FULL'] - v['TIDE'], 'ni')
        ku, kv = bp(u['WIND'] - u['BACK'], 'ni')[..., sl], bp(v['WIND'] - v['BACK'], 'ni')[..., sl]
        src = {'it': (bp(u['TIDE'], 'M2'), bp(v['TIDE'], 'M2')),
               'sm': (bp(u['FULL'] - u['TIDE'], 'M2'), bp(v['FULL'] - v['TIDE'], 'M2')),
               'ot': (bp(u['WIND'] - u['BACK'], 'M2'), bp(v['WIND'] - v['BACK'], 'M2'))}
        del u, v
        aw = (bp(load('FULL', 'w', k, z0e, z1e) - load('TIDE', 'w', k, z0e, z1e), 'ni')[..., sl]
              if HAVE_W else None)
        # the tide side of the triad needs the wake gradients, horizontal and vertical
        ax_u, ay_u, ax_v, ay_v = (d_xi(au, dxi)[..., sl], d_eta(au, deta)[..., sl],
                                  d_xi(av, dxi)[..., sl], d_eta(av, deta)[..., sl])
        az_u, az_v = (d_z(au, zc_e)[..., sl], d_z(av, zc_e)[..., sl]) if HAVE_W else (None, None)
        au, av = au[..., sl], av[..., sl]
        q1, q2, q0 = au*au - av*av, 2*au*av, au*au + av*av
        term = {}
        for s_, (bu, bv) in src.items():
            sn, ss, dl = strain_parts(bu, bv, dxi, deta)
            sn, ss, dl = sn[..., sl], ss[..., sl], dl[..., sl]
            term[f'sig_{s_}'] = -0.5*RHO0*(sn*q1 + ss*q2)
            if s_ == 'ot':
                continue
            term[f'del_{s_}'] = -0.5*RHO0*dl*q0
            if HAVE_W:
                bz_u, bz_v = d_z(bu, zc_e)[..., sl], d_z(bv, zc_e)[..., sl]
                term[f'z_{s_}'] = -RHO0*aw*(au*bz_u + av*bz_v)
                del bz_u, bz_v
            else:
                term[f'z_{s_}'] = np.zeros_like(q0)
            if s_ == 'it':
                bu_, bv_ = bu[..., sl], bv[..., sl]
                term['sig_kin'] = -0.5*RHO0*(sn*(ku*ku - kv*kv) + ss*(2*ku*kv))
                adv_u = au*ax_u + av*ay_u + (aw*az_u if HAVE_W else 0.0)
                adv_v = au*ax_v + av*ay_v + (aw*az_v if HAVE_W else 0.0)
                term['T_it'] = -RHO0*(bu_*adv_u + bv_*adv_v)
                del adv_u, adv_v, bu_, bv_
                s2 += np.nan_to_num(((sn*sn + ss*ss)[WORK]*dzc).sum(axis=-1).mean(axis=0))
                # the phase null, footprint points of the chunk
                fp = ft & np.isfinite(sn[0, ..., 0])
                if fp.any():
                    wgt = (dzc*ar_t[..., None])[fp]
                    Q1, Q2 = (q1[:, fp]*wgt).reshape(nt, -1), (q2[:, fp]*wgt).reshape(nt, -1)
                    K1 = ((ku*ku - kv*kv)[:, fp]*wgt).reshape(nt, -1)
                    K2 = ((2*ku*kv)[:, fp]*wgt).reshape(nt, -1)
                    Sn, Ss = sn[:, fp].reshape(nt, -1), ss[:, fp].reshape(nt, -1)
                    for i, s in enumerate(SHIFTS_H):
                        s = int(round(s/dt_h))
                        a_ = slice(max(s, 0), nt + min(s, 0))                     # tide at t + s
                        b_ = slice(max(-s, 0), nt - max(s, 0))
                        null_with[i, b_] += -0.5*RHO0*(np.einsum('tp,tp->t', Sn[a_], Q1[b_])
                                                       + np.einsum('tp,tp->t', Ss[a_], Q2[b_]))
                        null_without[i, b_] += -0.5*RHO0*(np.einsum('tp,tp->t', Sn[a_], K1[b_])
                                                          + np.einsum('tp,tp->t', Ss[a_], K2[b_]))
                    del Q1, Q2, K1, K2, Sn, Ss
            del sn, ss, dl
        del src, ax_u, ay_u, ax_v, ay_v, az_u, az_v, aw, ku, kv
        for q in Q:
            P[q] += np.nan_to_num((term[q]*dzc).sum(axis=-1))
        # profiles by depth bin and class
        term['sm'] = term['sig_sm'] + term['del_sm'] + term['z_sm']
        zc = -zr[..., z0:z1]
        ok_col = np.isfinite(term['sig_it'][0, ..., 0])
        for c in CLASS:
            m = cls_t[c] & ok_col
            if not m.any():
                continue
            zb = np.clip(np.digitize(zc[m], Z_EDGES) - 1, 0, nzb - 1)
            vol = dzc[m]*ar_t[m][:, None]
            prof_den[c] += np.bincount(zb.ravel(), weights=(vol*WORK.sum()).ravel(), minlength=nzb)
            for q in PROF:
                tw = np.nan_to_num(term[q][WORK][:, m].sum(axis=0))
                prof_num[(q, c)] += np.bincount(zb.ravel(), weights=(tw*vol).ravel(), minlength=nzb)
        del term, q1, q2, q0, au, av
    for q in Q:
        x = P[q][:, rows]
        Pint[q][b0:b1, cols] = x[WORK].sum(axis=0)*dt_h*3600.0
        m = foot[b0:b1, cols] & np.isfinite(x[0])
        series[q] += np.nan_to_num(x[:, m]*area_s[b0:b1, cols][m]).sum(axis=1)
    strain_amp[b0:b1, cols] = np.sqrt(2.0*s2[rows]/np.maximum(dz.sum(axis=-1)[rows], 1e-6))
    print(f'tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)

# --------------------------------------------------------------------------
# footprint series and the budget of the gain
# --------------------------------------------------------------------------


def smooth(x, n=SMOOTH_H):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


def J(x, sel=WORK):
    return float(np.sum(x[sel])*dt_h*3600.0)


ser = {q: series[q]/foot_area for q in Q}                                       # W m-2
ser['it'] = ser['sig_it'] + ser['del_it'] + ser['z_it']
ser['sm'] = ser['sig_sm'] + ser['del_sm'] + ser['z_sm']
ser['rev1'] = ser['sig_it'] + ser['sig_sm'] - ser['sig_ot']
sec = list(f2.sector.values).index('South')
E_gain = (f2.ser_Ewni_tide.values[:, sec] - f2.ser_Ewni_notide.values[:, sec])  # J m-2
dEdt = np.gradient(smooth(E_gain), dt_h*3600.0)                                  # W m-2
dPi = f2.ser_Pi_tide.values[:, sec] - f2.ser_Pi_notide.values[:, sec]           # W m-2
kw = np.where(WORK)[0]
after = (t >= T_OFF) & WORK
budget = {'work of the internal tide': J(ser['it']), '  strain': J(ser['sig_it']),
          '  divergence': J(ser['del_it']), '  vertical shear': J(ser['z_it']),
          'work of the storm-made motion': J(ser['sm']), '  strain ': J(ser['sig_sm']),
          '  divergence ': J(ser['del_sm']), '  vertical shear ': J(ser['z_sm']),
          'wind work difference': J(dPi), 'tendency of the gain': J(dEdt),
          'gain over the window': float(smooth(E_gain)[kw[-1]] - smooth(E_gain)[kw[0]])}
budget['residual'] = (budget['gain over the window'] - budget['work of the internal tide']
                      - budget['work of the storm-made motion'] - budget['wind work difference'])
budget_storm = {k_: J(ser[q], WORK & ~after) for k_, q in (('internal tide', 'it'), ('storm-made', 'sm'))}
budget_storm['wind work difference'] = J(dPi, WORK & ~after)
budget_storm['tendency of the gain'] = J(dEdt, WORK & ~after)
budget_after = {k_: J(ser[q], after) for k_, q in (('internal tide', 'it'), ('storm-made', 'sm'))}
budget_after['wind work difference'] = J(dPi, after)
budget_after['tendency of the gain'] = J(dEdt, after)
pos_hours = {q: int(np.sum(ser[q][WORK] > 0)) for q in ('sig_it', 'it', 'sm')}
triad = dict(P_it=J(ser['it']), T_it=J(ser['T_it']))
triad['minus_T_over_P'] = -triad['T_it']/triad['P_it'] if triad['P_it'] != 0 else np.nan

# the null, integrated over the work window
null_with_int = np.nansum(null_with[:, WORK], axis=1)*dt_h*3600.0/foot_area
null_without_int = np.nansum(null_without[:, WORK], axis=1)*dt_h*3600.0/foot_area
i0 = int(np.where(SHIFTS_H == 0)[0][0])
null = dict(with_actual=float(null_with_int[i0]), with_rank=float(np.mean(null_with_int <= null_with_int[i0])),
            with_amp=float(0.5*(null_with_int.max() - null_with_int.min())),
            without_actual=float(null_without_int[i0]),
            without_rank=float(np.mean(null_without_int <= null_without_int[i0])),
            without_amp=float(0.5*(null_without_int.max() - null_without_int.min())))

# --------------------------------------------------------------------------
# blocks, dose and response, bilinearity, the triad, the wind work route
# --------------------------------------------------------------------------
ddc = dc_storm['FULL'] - dc_storm['WIND']
ids = np.unique(blk[foot & np.isfinite(Pint['sig_it'])])
rows_b = []
for b in ids:
    m = foot & (blk == b) & np.isfinite(Pint['sig_it']) & np.isfinite(gain_map)
    if m.sum() < BLOCK*BLOCK//4:
        continue
    a = area_s[m]

    def mean_(x):
        return float(np.nansum(x[m]*a)/a[np.isfinite(x[m])].sum()) if np.isfinite(x[m]).any() else np.nan

    rec = dict(block=int(b), lon=mean_(lon_s), lat=mean_(lat_s), n=int(m.sum()),
               free_share=float(np.mean(dc_full[m] < 0)))
    for q in Q:
        rec[f'work_{q}'] = mean_(Pint[q])
    rec['work_it'] = rec['work_sig_it'] + rec['work_del_it'] + rec['work_z_it']
    rec['work_sm'] = rec['work_sig_sm'] + rec['work_del_sm'] + rec['work_z_sm']
    rec.update(gain=mean_(gain_map), wind_diff=mean_(wind_diff), W_tide=mean_(work_pair['tide']),
               W_notide=mean_(work_pair['notide']), Ew_tide=mean_(Ew['tide']),
               Ew_notide=mean_(Ew['notide']), strain_amp=mean_(strain_amp),
               ddc_storm=mean_(ddc), dc_storm_full=mean_(dc_storm['FULL']))
    rows_b.append(rec)
blocks = pd.DataFrame(rows_b)
blocks['class'] = np.where(blocks.free_share > 0.5, 'free', 'trapped')
blocks.to_csv(OUTDIR + 'fig3_blocks.csv', index=False)
nb = len(blocks)


def shift_level(x, y, n=N_BOOT, q=95):
    """Circular-shift significance level of a Spearman correlation."""
    r = np.array([abs(spearmanr(np.roll(x, s), y).correlation)
                  for s in rng.integers(1, x.size - 1, size=n)])
    return float(np.percentile(r, q))


def partial_spearman(x, y, zs):
    """Rank correlation of x and y with the ranks of every z removed by least squares."""
    Z = np.column_stack([np.ones(x.size)] + [rankdata(z) for z in zs])
    res = [r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0] for r in (rankdata(x), rankdata(y))]
    return float(np.corrcoef(res[0], res[1])[0, 1])


def boot_ratio(num, den, n=N_BOOT):
    i = rng.integers(0, num.size, size=(n, num.size))
    bs = num[i].sum(1)/den[i].sum(1)
    return float(num.sum()/den.sum()), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


stats = {}
dose = np.full((N_DOSE, 5), np.nan)
if nb >= MIN_BLOCKS:
    x, y = blocks.work_sig_it.values, blocks.gain.values
    zw, ze = blocks.wind_diff.values, blocks.Ew_tide.values
    stats['rho'] = float(spearmanr(x, y).correlation)
    stats['rho_level'] = shift_level(x, y)
    stats['rho_partial'] = partial_spearman(x, y, [zw, ze])
    stats['rho_total_it'] = float(spearmanr(blocks.work_it.values, y).correlation)
    stats['rho_sm'] = float(spearmanr(blocks.work_sm.values, y).correlation)
    stats['rho_wind'] = float(spearmanr(zw, y).correlation)
    stats['rho_wind_level'] = shift_level(zw, y)
    edges = np.quantile(x, np.linspace(0, 1, N_DOSE + 1))
    for i in range(N_DOSE):
        m = (x >= edges[i]) & ((x <= edges[i + 1]) if i == N_DOSE - 1 else (x < edges[i + 1]))
        if m.sum() < 3:
            continue
        bs = np.median(y[m][rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))], axis=1)
        dose[i] = [np.median(x[m]), np.median(y[m]), np.percentile(bs, 5), np.percentile(bs, 95), m.sum()]
    pos = (x > 0) & (blocks.strain_amp.values > 0) & (ze > 0)
    if pos.sum() >= MIN_BLOCKS:
        xb, yb = np.log10(blocks.strain_amp.values[pos]*ze[pos]), np.log10(x[pos])
        ts = theilslopes(yb, xb)
        stats.update(bil_slope=float(ts[0]), bil_lo=float(ts[2]), bil_hi=float(ts[3]),
                     bil_rho=float(spearmanr(xb, yb).correlation), bil_level=shift_level(xb, yb),
                     bil_n=int(pos.sum()))
    # the triad over the blocks
    stats['rho_triad'] = float(spearmanr(blocks.work_it.values, -blocks.work_T_it.values).correlation)
    stats['rho_triad_level'] = shift_level(blocks.work_it.values, -blocks.work_T_it.values)
    # the wind work route
    ok = np.isfinite(blocks.ddc_storm.values)
    if ok.sum() >= MIN_BLOCKS:
        dd = blocks.ddc_storm.values[ok]
        stats['rho_wind_ddc'] = float(spearmanr(dd, zw[ok]).correlation)
        stats['rho_wind_absddc'] = float(spearmanr(np.abs(dd), zw[ok]).correlation)
        stats['rho_wind_ddc_level'] = shift_level(dd, zw[ok])
        thr = float(np.quantile(np.abs(dd), 1/3))
        matched = np.abs(dd) <= thr
        Wt, Wn = blocks.W_tide.values[ok], blocks.W_notide.values[ok]
        good = Wn > 0
        stats['ddc_threshold'] = thr
        stats['wind_ratio_all'] = boot_ratio(Wt[good], Wn[good])
        stats['wind_ratio_matched'] = boot_ratio(Wt[good & matched], Wn[good & matched]) \
            if (good & matched).sum() >= 3 else (np.nan,)*3
        stats['wind_ratio_unmatched'] = boot_ratio(Wt[good & ~matched], Wn[good & ~matched]) \
            if (good & ~matched).sum() >= 3 else (np.nan,)*3
        stats['n_matched'] = int((good & matched).sum())
    for c in ('free', 'trapped'):
        m = blocks['class'].values == c
        if m.sum() >= 3:
            stats[f'work_it_{c}'] = float(np.mean(blocks.work_it.values[m]))
            stats[f'work_sm_{c}'] = float(np.mean(blocks.work_sm.values[m]))
            stats[f'gain_{c}'] = float(np.mean(y[m]))
            stats[f'n_{c}'] = int(m.sum())

# --------------------------------------------------------------------------
# profiles, W m-3 averaged over the window
# --------------------------------------------------------------------------
prof = np.full((len(PROF), len(CLASS), nzb), np.nan)
for i, q in enumerate(PROF):
    for j, c in enumerate(CLASS):
        with np.errstate(divide='ignore', invalid='ignore'):
            prof[i, j] = np.where(prof_den[c] > 0, prof_num[(q, c)]/prof_den[c], np.nan)

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
SER = Q + ['it', 'sm', 'rev1']
ds = xr.Dataset(
    data_vars={**{f'Pint_{q}': (('y', 'x'), Pint[q]) for q in Q},
               'strain_amp': (('y', 'x'), strain_amp), 'gain': (('y', 'x'), gain_map.astype(np.float32)),
               'wind_diff': (('y', 'x'), wind_diff.astype(np.float32)),
               **{f'dc_storm_{r}': (('y', 'x'), dc_storm[r].astype(np.float32)) for r in dc_storm},
               **{f'ser_{q}': (('time',), ser[q]) for q in SER},
               'ser_dEdt': (('time',), dEdt), 'ser_dPi': (('time',), dPi), 'ser_Egain': (('time',), E_gain),
               'null_with': (('shift',), null_with_int), 'null_without': (('shift',), null_without_int),
               'null_with_t': (('shift', 'time'), (null_with/foot_area).astype(np.float32)),
               'null_without_t': (('shift', 'time'), (null_without/foot_area).astype(np.float32)),
               'prof': (('term', 'cls', 'zbin'), prof), 'dose': (('dose_bin', 'dose_stat'), dose),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'slope': (('y', 'x'), slope_s), 'foot': (('y', 'x'), foot),
               'dc_FULL': (('y', 'x'), dc_full), 'area': (('y', 'x'), area_s),
               'mask': (('y', 'x'), f2.mask.values)},
    coords=dict(time=t, shift=SHIFTS_H, term=PROF, cls=list(CLASS),
                zbin=0.5*(Z_EDGES[1:] + Z_EDGES[:-1]), dose_bin=np.arange(N_DOSE),
                dose_stat=['work_med', 'gain_med', 'gain_p05', 'gain_p95', 'n']),
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF)[:16],
               ni_band=str(NI_BAND), m2_band=str(M2_BAND), smooth_h=SMOOTH_H, have_w=int(HAVE_W),
               z_edges=str(list(Z_EDGES)), foot_area_m2=foot_area, block_points=BLOCK,
               n_boot=N_BOOT, omega_M2=OMEGA_M2, budget=str(budget), budget_storm=str(budget_storm),
               budget_after=str(budget_after), triad=str(triad), null=str(null), stats=str(stats),
               names=str(NAMES)))
ds.to_netcdf(OUTDIR + 'fig3_strain.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['FIGURE 3 REPORT (full_v5, revision 3)  --  the work of the tidal strain on the wake',
     '=' * 84,
     f'slab caches, {nt} records, footprint {foot_area/1e6:.0f} km2, {nb} blocks of {BLOCK} points',
     'wake u_FULL - u_TIDE (near-inertial band). Sources in the semidiurnal band: the internal tide',
     'u_TIDE and the storm-made motion u_FULL - u_TIDE, whose sum is the semidiurnal band of FULL',
     f'work window {T_STORM0} to {T_WAKE1}, wind stops {str(T_OFF)[:16]}, vertical velocity '
     f'{"cached" if HAVE_W else "NOT cached (vertical terms zero)"}', '',
     '1. THE BUDGET OF THE GAIN OVER THE FOOTPRINT (J m-2 over the work window)']
for k_, val in budget.items():
    L.append(f'   {k_:32s} {val:10.1f}')
L.append('   storm window: ' + ', '.join(f'{k_} {v:.1f}' for k_, v in budget_storm.items()))
L.append('   after the wind stops: ' + ', '.join(f'{k_} {v:.1f}' for k_, v in budget_after.items()))
L.append(f'   hours of positive footprint work in the window ({int(WORK.sum())}): strain of the internal '
         f'tide {pos_hours["sig_it"]}, work of the internal tide {pos_hours["it"]}, storm-made {pos_hours["sm"]}')
L.append(f'   revision 1 strain work reproduced as W(TIDE) + W(FULL - TIDE) - W(WIND - BACK): {J(ser["rev1"]):.1f}')
L.append('   the residual holds the work against the balanced flow, radiation and dissipation.')
L.append('')
L.append('2. THE TRIAD (internal tide, footprint, J m-2). P + T is a flux divergence')
L.append(f'   wake side P {triad["P_it"]:.1f}, tide side T {triad["T_it"]:.1f}, -T/P {triad["minus_T_over_P"]:.2f}')
if 'rho_triad' in stats:
    L.append(f'   blocks, Spearman of P against -T {stats["rho_triad"]:.2f} (level {stats["rho_triad_level"]:.2f})')
L.append('')
L.append(f'3. THE PHASE NULL (the internal tide shifted by {SHIFTS_H[0]} to {SHIFTS_H[-1]} h, strain work, J m-2)')
L.append(f'   wake with the tide     actual {null["with_actual"]:8.1f}, rank {null["with_rank"]:.2f}, '
         f'half range {null["with_amp"]:.1f}')
L.append(f'   wake without the tide  actual {null["without_actual"]:8.1f}, rank {null["without_rank"]:.2f}, '
         f'half range {null["without_amp"]:.1f}')
L.append('   shift (h), with, without:')
for i in range(0, SHIFTS_H.size, 3):
    L.append(f'   {SHIFTS_H[i]:+4d} {null_with_int[i]:9.1f} {null_without_int[i]:9.1f}')
L.append('')
L.append('4. DOSE AND RESPONSE OVER THE BLOCKS (gain against the strain work of the internal tide)')
if 'rho' in stats:
    L.append(f'   Spearman {stats["rho"]:.2f} (level {stats["rho_level"]:.2f}), partial on the wind work '
             f'difference and the wake kinetic energy {stats["rho_partial"]:.2f}')
    L.append(f'   gain against the total work of the internal tide {stats["rho_total_it"]:.2f}, against the work of the '
             f'storm-made motion {stats["rho_sm"]:.2f}, against the wind work difference {stats["rho_wind"]:.2f} '
             f'(level {stats["rho_wind_level"]:.2f})')
    L.append(f'{"work_med":>10s} {"gain_med":>10s} {"p05":>9s} {"p95":>9s} {"n":>4s}')
    for row in dose:
        if np.isfinite(row[0]):
            L.append(f'{row[0]:10.1f} {row[1]:10.1f} {row[2]:9.1f} {row[3]:9.1f} {int(row[4]):4d}')
    for c in ('free', 'trapped'):
        if f'n_{c}' in stats:
            L.append(f'   {c:8s} blocks {stats[f"n_{c}"]:3d}, work of the internal tide {stats[f"work_it_{c}"]:.1f}, '
                     f'storm-made {stats[f"work_sm_{c}"]:.1f}, gain {stats[f"gain_{c}"]:.1f}')
L.append('')
L.append('5. BILINEARITY (strain work of the internal tide against strain amplitude times wake kinetic energy)')
if 'bil_slope' in stats:
    L.append(f'   Theil and Sen slope {stats["bil_slope"]:.2f} [{stats["bil_lo"]:.2f}, {stats["bil_hi"]:.2f}], '
             f'Spearman {stats["bil_rho"]:.2f} (level {stats["bil_level"]:.2f}), {stats["bil_n"]} blocks of positive work')
L.append('')
L.append('6. THE WIND WORK ROUTE (storm-window detuning of FULL minus WIND, blocks)')
if 'rho_wind_ddc' in stats:
    L.append(f'   wind work difference against the difference of detuning: Spearman {stats["rho_wind_ddc"]:.2f} '
             f'(level {stats["rho_wind_ddc_level"]:.2f}), against its magnitude {stats["rho_wind_absddc"]:.2f}')
    for k_ in ('all', 'matched', 'unmatched'):
        v = stats[f'wind_ratio_{k_}']
        L.append(f'   wind work with over without the tide, {k_:9s} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]')
    L.append(f'   matched blocks have |difference of detuning| under {stats["ddc_threshold"]*1e3:.2f} x 1e-3 cph '
             f'({stats["n_matched"]} blocks)')
    L.append('   the ratios keep only blocks where the wind did positive work on the wake without the tide.')
    L.append('   The footprint ratio of Figure 2 (1.7) is carried by the blocks where it did negative work,')
    L.append('   where the wind met an inertial current already turning against it.')
L.append('')
L.append('7. THE WORKS BY DEPTH (W m-3 times 1e6, mean over the window and the class)')
L.append(f'{"depth m":>9s}' + ''.join(f'{q + " " + c:>16s}' for q in PROF for c in CLASS))
for j in range(nzb):
    L.append(f'{Z_EDGES[j]:4.0f}-{Z_EDGES[j + 1]:4.0f}'
             + ''.join(f'{1e6*prof[i, jc, j]:16.3f}' for i in range(len(PROF)) for jc in range(len(CLASS))))
L.append('')
L.append('READING THE RESULT')
L.append('   question 2 is answered yes if the work of the internal tide has the sign of the gain and')
L.append('   lies within a factor of two of it, the tide side mirrors it (-T/P near one), the wake with')
L.append('   the tide sits in the upper part of its phase null while the wake without the tide sits')
L.append('   near zero, and the gain rises with the work across the blocks once the wind work')
L.append('   difference and the wake kinetic energy are held fixed. The wind work route belongs to the')
L.append('   tide if the wind work ratio stays above one where the two flows share their detuning.')
open(OUTDIR + 'fig3_strain_report.txt', 'w').write('\n'.join(L))
print('\n'.join(L))
print('saved', OUTDIR + 'fig3_strain.nc')