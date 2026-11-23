"""
figS10_compute.py  (full_v5)
----------------------------
Is the tide that does work on the wake the internal tide?

The tide of Figure 3 is the semidiurnal band of the velocity of TIDE at every
level. That velocity holds the barotropic tide (the depth mean) and the
internal tide (the deviation from the depth mean). The work on the wake is
linear in the velocity of the tide, so it splits exactly into two parts,

    b = b_bt + b_bc,   b_bt = (1/H) sum_z b dz,   b_bc = b - b_bt,

    P(b) = P(b_bt) + P(b_bc),      T(b) = T(b_bt) + T(b_bc),

with the same operators as fig_03_compute.py: the strain work, the divergence
work and the vertical shear work on the wake u_FULL - u_TIDE (near-inertial
band), and the tide side of the triad. The depth mean has no vertical shear,
so the vertical shear work belongs to the internal tide alone.

What is written
  1. the works of each part over the footprint and the work window, with the
     part of the storm window and the part after the wind subsides, and the
     share of the internal tide in each
  2. the same by class of the flow (below and above resonance)
  3. the tide side of each part and the ratio of loss to work
  4. the kinetic energy of the semidiurnal band of TIDE in each part and the
     strain amplitude of each part
  5. the 97 shifted tides for each part (the strain work of the tide shifted
     by -48 to +48 h on the same wake) and the place of the actual tide
  6. the strain work of each part by depth bin and class
  7. the maps of the strain work of each part and their pattern correlation
     with the strain work of the whole tide
  8. a check. The sum of the two parts against the series of fig3_strain.nc

Inputs
  ../new_full/fig6_cache_<RUN>/     u, v and w of FULL and TIDE on the slope tiles
  ./fig2_out/fig2_wake.nc           geometry, classes and windows
  ./fig3_out/fig3_strain.nc         for the check (used if it is there)

Outputs (in ./figS10_out/)
  figS10_tide_parts.nc   figS10_tide_parts_report.txt

Nothing is extracted from the model. About the time of fig_03_compute.py.

  cd ~/dd_waves/paper_3/full_v5
  nohup python -u figS10_compute.py > log_figS10_compute.txt 2>&1 &
  tail -f log_figS10_compute.txt
  cat figS10_out/figS10_tide_parts_report.txt
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------------------------------
# paths and settings (those of fig_03_compute.py)
# --------------------------------------------------------------------------
ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
NEW_FULL = '../new_full/'
CACHES = {'TIDE': [NEW_FULL + 'fig6_cache_TIDE/'],
          'FULL': [NEW_FULL + 'fig6_cache_FULL/', NEW_FULL + 'fig6_cache/']}
FIG2 = './fig2_out/fig2_wake.nc'
FIG3 = './fig3_out/fig3_strain.nc'
OUTDIR = './figS10_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['TIDE', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2
M2_BAND = (0.95, 1.05)                             # times omega_M2
CHUNK_Z = 4
SHIFTS_H = np.arange(-48, 49)
Z_EDGES = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000.0])

PARTS = ['bt', 'bc']                               # barotropic tide and internal tide
PART_NAME = {'bt': 'barotropic tide (depth mean)', 'bc': 'internal tide (deviation from the depth mean)',
             'all': 'whole tide (as in Figure 3)'}
Q = ['sig_bt', 'del_bt', 'T_bt', 'sig_bc', 'del_bc', 'z_bc', 'T_bc']

# --------------------------------------------------------------------------
# grid, slab geometry and the Figure 2 fields
# --------------------------------------------------------------------------
ref = xr.open_dataset(HIS)
h, pm, pn = ref.h.values, ref.pm.values, ref.pn.values
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
HAVE_W = all(os.path.exists(cache[r] + 'w_t000.npy') for r in RUNS)
print(f'{ntiles} tiles, {nt} records, slab {ny_s} x {nx_s}, vertical velocity '
      f'{"cached" if HAVE_W else "not cached, the vertical terms are skipped"}', flush=True)

f2 = xr.open_dataset(FIG2)
assert f2.sizes['y'] == ny_s and f2.sizes['x'] == nx_s, 'fig2_wake.nc is not on the slab geometry'
area_s, slope_s, sector_s = f2.area.values, f2.slope.values.astype(bool), f2.sector_code.values
lon_s, lat_s, h_s = f2.lon.values, f2.lat.values, f2.h.values
foot = slope_s & (sector_s == 0)
dc_full = f2.dc_FULL.values*RAD2CPH
CLASS = {'free': foot & (dc_full < 0), 'trapped': foot & (dc_full > 0)}          # below and above resonance
CLASS_NAME = {'free': 'below resonance', 'trapped': 'above resonance'}
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
AFTER = WORK & (t >= T_OFF)
BEFORE = WORK & ~AFTER
foot_area = float(area_s[foot].sum())
class_area = {c: float(area_s[m].sum()) for c, m in CLASS.items()}
print(f'footprint {foot_area/1e6:.0f} km2, work window {T_STORM0} to {T_WAKE1}, wind subsides {T_OFF}', flush=True)


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


def d_z(a, z):
    out = np.empty_like(a)
    zz = z[None]
    out[..., 1:-1] = (a[..., 2:] - a[..., :-2])/(zz[..., 2:] - zz[..., :-2])
    out[..., 0] = (a[..., 1] - a[..., 0])/(zz[..., 1] - zz[..., 0])
    out[..., -1] = (a[..., -1] - a[..., -2])/(zz[..., -1] - zz[..., -2])
    return out


def strain_parts(bu, bv, dxi, deta):
    """Normal strain, shear strain and divergence of a horizontal field."""
    ux, uy, vx, vy = d_xi(bu, dxi), d_eta(bu, deta), d_xi(bv, dxi), d_eta(bv, deta)
    return ux - vy, vx + uy, ux + vy


def load(run, var, k, z0, z1):
    """Levels z0 to z1 of a cached variable. A variable on the w levels is averaged to the rho levels."""
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    if a.shape[-1] == s_w.size:
        x = np.asarray(a[..., z0:z1 + 1], dtype=np.float32)
        return np.nan_to_num(0.5*(x[..., 1:] + x[..., :-1]))
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


def clock(t0, done, total):
    """Elapsed minutes, minutes left and the hour of the end."""
    el = (pd.Timestamp.now() - t0).total_seconds()/60.0
    left = el*(total - done)/max(done, 1)
    return f'{el:.1f} min elapsed, about {left:.0f} min left, ends near {(pd.Timestamp.now() + pd.Timedelta(minutes=left)).strftime("%H:%M")}'


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------
Pint = {q: np.full((ny_s, nx_s), np.nan, np.float32) for q in Q}                  # J m-2 over WORK
series = {q: np.zeros(nt) for q in Q}                                              # W, footprint sum
series_c = {(q, c): np.zeros(nt) for q in Q for c in CLASS}                        # W, class sum
shifted = {p: np.zeros((SHIFTS_H.size, nt)) for p in PARTS}                        # W, strain work of the shifted tide
ke = {p: np.full((ny_s, nx_s), np.nan, np.float32) for p in PARTS}                 # J m-2, mean over WORK
s2map = {p: np.full((ny_s, nx_s), np.nan, np.float32) for p in PARTS + ['x']}      # s-2, depth mean of the squared strain
nzb = Z_EDGES.size - 1
prof_num = {(p, c): np.zeros(nzb) for p in PARTS for c in CLASS}
prof_den = {c: np.zeros(nzb) for c in CLASS}
nsteps = sum(int(np.ceil(np.load(cache['FULL'] + f'u_t{k:03d}.npy', mmap_mode='r').shape[-1]/CHUNK_Z))
             for k in range(ntiles))*2
step = 0
t0 = pd.Timestamp.now()

for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    zr = np.moveaxis(z_of(s_rho, Cs_r, h_t), 0, -1)                                # (ny, nx, nz)
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    H = dz.sum(axis=-1)
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

    # ---------------------------------------------------------------- pass 1. The depth mean of the tide
    Ub, Vb = np.zeros((nt, ny_t, nx_t), np.float64), np.zeros((nt, ny_t, nx_t), np.float64)
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        Ub += (load('TIDE', 'u', k, z0, z1)*dz[None, ..., z0:z1]).sum(axis=-1)
        Vb += (load('TIDE', 'v', k, z0, z1)*dz[None, ..., z0:z1]).sum(axis=-1)
        step += 1
    Ub, Vb = bp((Ub/H[None]).astype(np.float32), 'M2'), bp((Vb/H[None]).astype(np.float32), 'M2')
    sn_bt, ss_bt, dl_bt = strain_parts(Ub, Vb, dxi, deta)                            # (time, y, x)
    print(f'tile {k + 1} of {ntiles}, depth mean of the tide done, {clock(t0, step, nsteps)}', flush=True)

    # ---------------------------------------------------------------- pass 2. The works of each part
    P = {q: np.zeros((nt, ny_t, nx_t), np.float32) for q in Q}
    s2 = {p: np.zeros((ny_t, nx_t), np.float64) for p in PARTS + ['x']}
    kebc = np.zeros((ny_t, nx_t), np.float64)
    fp = ft & np.isfinite(sn_bt[0])
    Q1z = np.zeros((nt, int(fp.sum()))) if fp.any() else None                        # depth sums of the wake quadratics
    Q2z = np.zeros((nt, int(fp.sum()))) if fp.any() else None
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        z0e, z1e = max(z0 - 1, 0), min(z1 + 1, nz)                                   # one level of overlap
        sl = slice(z0 - z0e, z0 - z0e + (z1 - z0))
        zc_e = zr[..., z0e:z1e]
        dzc = dz[..., z0:z1]
        uT, vT = load('TIDE', 'u', k, z0e, z1e), load('TIDE', 'v', k, z0e, z1e)
        au, av = bp(load('FULL', 'u', k, z0e, z1e) - uT, 'ni'), bp(load('FULL', 'v', k, z0e, z1e) - vT, 'ni')
        bu, bv = bp(uT, 'M2') - Ub[..., None], bp(vT, 'M2') - Vb[..., None]          # the internal tide
        del uT, vT
        aw = (bp(load('FULL', 'w', k, z0e, z1e) - load('TIDE', 'w', k, z0e, z1e), 'ni')[..., sl]
              if HAVE_W else None)
        ax_u, ay_u, ax_v, ay_v = (d_xi(au, dxi)[..., sl], d_eta(au, deta)[..., sl],
                                  d_xi(av, dxi)[..., sl], d_eta(av, deta)[..., sl])
        az_u, az_v = (d_z(au, zc_e)[..., sl], d_z(av, zc_e)[..., sl]) if HAVE_W else (None, None)
        au, av = au[..., sl], av[..., sl]
        q1, q2, q0 = au*au - av*av, 2*au*av, au*au + av*av
        adv_u = au*ax_u + av*ay_u + (aw*az_u if HAVE_W else 0.0)
        adv_v = au*ax_v + av*ay_v + (aw*az_v if HAVE_W else 0.0)
        term = {}
        # the internal tide
        sn, ss, dl = strain_parts(bu, bv, dxi, deta)
        sn, ss, dl = sn[..., sl], ss[..., sl], dl[..., sl]
        term['sig_bc'] = -0.5*RHO0*(sn*q1 + ss*q2)
        term['del_bc'] = -0.5*RHO0*dl*q0
        term['z_bc'] = (-RHO0*aw*(au*d_z(bu, zc_e)[..., sl] + av*d_z(bv, zc_e)[..., sl]) if HAVE_W
                        else np.zeros_like(q0))
        term['T_bc'] = -RHO0*(bu[..., sl]*adv_u + bv[..., sl]*adv_v)
        # the barotropic tide (the same at every level)
        term['sig_bt'] = -0.5*RHO0*(sn_bt[..., None]*q1 + ss_bt[..., None]*q2)
        term['del_bt'] = -0.5*RHO0*dl_bt[..., None]*q0
        term['T_bt'] = -RHO0*(Ub[..., None]*adv_u + Vb[..., None]*adv_v)
        for q in Q:
            P[q] += np.nan_to_num((term[q]*dzc).sum(axis=-1))
        # squared strain and kinetic energy of the internal tide
        s2['bc'] += np.nan_to_num(((sn*sn + ss*ss)[WORK]*dzc).sum(axis=-1).mean(axis=0))
        s2['x'] += np.nan_to_num(((sn*sn_bt[..., None] + ss*ss_bt[..., None])[WORK]*dzc).sum(axis=-1).mean(axis=0))
        kebc += np.nan_to_num((0.5*RHO0*(bu[..., sl]**2 + bv[..., sl]**2)[WORK]*dzc).sum(axis=-1).mean(axis=0))
        # the shifted tides, footprint points of the chunk
        if fp.any():
            wgt = (dzc*ar_t[..., None])[fp]
            Q1, Q2 = (q1[:, fp]*wgt), (q2[:, fp]*wgt)                                 # (time, points, levels)
            Q1z += Q1.sum(axis=-1)
            Q2z += Q2.sum(axis=-1)
            Q1, Q2 = Q1.reshape(nt, -1), Q2.reshape(nt, -1)
            Sn, Ss = np.nan_to_num(sn[:, fp]).reshape(nt, -1), np.nan_to_num(ss[:, fp]).reshape(nt, -1)
            for i, s in enumerate(SHIFTS_H):
                s = int(round(s/dt_h))
                a_ = slice(max(s, 0), nt + min(s, 0))                                 # tide at t + s
                b_ = slice(max(-s, 0), nt - max(s, 0))
                shifted['bc'][i, b_] += -0.5*RHO0*(np.einsum('tp,tp->t', Sn[a_], Q1[b_])
                                                   + np.einsum('tp,tp->t', Ss[a_], Q2[b_]))
            del Q1, Q2, Sn, Ss
        # profiles of the strain work by depth bin and class
        zc = -zr[..., z0:z1]
        ok_col = np.isfinite(term['sig_bc'][0, ..., 0])
        for c in CLASS:
            m = cls_t[c] & ok_col
            if not m.any():
                continue
            zb = np.clip(np.digitize(zc[m], Z_EDGES) - 1, 0, nzb - 1)
            vol = dzc[m]*ar_t[m][:, None]
            prof_den[c] += np.bincount(zb.ravel(), weights=(vol*WORK.sum()).ravel(), minlength=nzb)
            for p in PARTS:
                tw = np.nan_to_num(term[f'sig_{p}'][WORK][:, m].sum(axis=0))
                prof_num[(p, c)] += np.bincount(zb.ravel(), weights=(tw*vol).ravel(), minlength=nzb)
        del term, q1, q2, q0, au, av, bu, bv, aw, adv_u, adv_v, sn, ss, dl
        step += 1
        print(f'   tile {k + 1} of {ntiles}, levels {z0} to {z1 - 1} of {nz}, {clock(t0, step, nsteps)}', flush=True)
    # the shifted barotropic tide, with the depth sums of the wake quadratics
    if fp.any():
        Sn, Ss = np.nan_to_num(sn_bt[:, fp]), np.nan_to_num(ss_bt[:, fp])
        for i, s in enumerate(SHIFTS_H):
            s = int(round(s/dt_h))
            a_ = slice(max(s, 0), nt + min(s, 0))
            b_ = slice(max(-s, 0), nt - max(s, 0))
            shifted['bt'][i, b_] += -0.5*RHO0*(np.einsum('tp,tp->t', Sn[a_], Q1z[b_])
                                               + np.einsum('tp,tp->t', Ss[a_], Q2z[b_]))
    # maps and series of the tile
    for q in Q:
        x = P[q][:, rows]
        Pint[q][b0:b1, cols] = x[WORK].sum(axis=0)*dt_h*3600.0
        m = foot[b0:b1, cols] & np.isfinite(sn_bt[0][rows])
        series[q] += np.nan_to_num(x[:, m]*area_s[b0:b1, cols][m]).sum(axis=1)
        for c in CLASS:
            mc = CLASS[c][b0:b1, cols] & np.isfinite(sn_bt[0][rows])
            series_c[(q, c)] += np.nan_to_num(x[:, mc]*area_s[b0:b1, cols][mc]).sum(axis=1)
    ok = np.isfinite(sn_bt[0])
    s2['bt'] = np.where(ok, np.nan_to_num((sn_bt*sn_bt + ss_bt*ss_bt)[WORK].mean(axis=0))*H, np.nan)
    ke['bt'][b0:b1, cols] = (0.5*RHO0*(Ub**2 + Vb**2)[WORK].mean(axis=0)*H)[rows]
    ke['bc'][b0:b1, cols] = kebc[rows]
    for p in PARTS + ['x']:
        s2map[p][b0:b1, cols] = (np.where(ok, s2[p], np.nan)/np.maximum(H, 1e-6))[rows]

# --------------------------------------------------------------------------
# footprint numbers
# --------------------------------------------------------------------------


def J(x, sel=WORK):
    return float(np.sum(x[sel])*dt_h*3600.0)


ser = {q: series[q]/foot_area for q in Q}                                          # W m-2
ser['z_bt'] = np.zeros(nt)
for p in PARTS:
    ser[f'P_{p}'] = ser[f'sig_{p}'] + ser[f'del_{p}'] + ser[f'z_{p}']
for q in ('sig', 'del', 'z', 'P', 'T'):
    ser[f'{q}_all'] = ser[f'{q}_bt'] + ser[f'{q}_bc']
ser_c = {}
for c in CLASS:
    for q in Q:
        ser_c[(q, c)] = series_c[(q, c)]/max(class_area[c], 1.0)
    ser_c[('z_bt', c)] = np.zeros(nt)
    for p in PARTS:
        ser_c[(f'P_{p}', c)] = ser_c[(f'sig_{p}', c)] + ser_c[(f'del_{p}', c)] + ser_c[(f'z_{p}', c)]
    for q in ('sig', 'del', 'z', 'P', 'T'):
        ser_c[(f'{q}_all', c)] = ser_c[(f'{q}_bt', c)] + ser_c[(f'{q}_bc', c)]

TERMS = ['sig', 'del', 'z', 'P', 'T']
TERM_NAME = {'sig': 'strain work', 'del': 'divergence work', 'z': 'vertical shear work', 'P': 'work on the wake (sum)',
             'T': 'tide side of the triad'}
SRC = ['all', 'bt', 'bc']
tab = {(q, s, w): J(ser[f'{q}_{s}'], sel) for q in TERMS for s in SRC
       for w, sel in (('work', WORK), ('storm', BEFORE), ('after', AFTER))}
tab_c = {(q, s, c): J(ser_c[(f'{q}_{s}', c)]) for q in TERMS for s in SRC for c in CLASS}


def share(a, b):
    return a/b if abs(b) > 1e-12 else np.nan


# the shifted tides: place of the actual tide among the 97 (1 for the largest work)
i0 = int(np.where(SHIFTS_H == 0)[0][0])
shifted['all'] = shifted['bt'] + shifted['bc']
place = {}
for p in PARTS + ['all']:
    for w, sel in (('work', WORK), ('after', AFTER)):
        x = np.nansum(shifted[p][:, sel], axis=1)*dt_h*3600.0/foot_area
        place[(p, w)] = dict(actual=float(x[i0]), place=int(1 + np.sum(x > x[i0])), amp=float(0.5*(x.max() - x.min())),
                             values=x)

# kinetic energy and strain of each part over the footprint
aw_ = np.where(foot, area_s, 0.0)


def fmean(a):
    g = foot & np.isfinite(a)
    return float(np.sum(a[g]*area_s[g])/max(np.sum(area_s[g]), 1.0))


KE = {p: fmean(ke[p]) for p in PARTS}
S2 = {p: fmean(s2map[p]) for p in PARTS + ['x']}
S2['all'] = S2['bt'] + S2['bc'] + 2*S2['x']
RMS = {p: np.sqrt(max(S2[p], 0.0)) for p in ('bt', 'bc', 'all')}

# pattern correlation of the maps of the strain work
Pint['sig_all'] = Pint['sig_bt'] + Pint['sig_bc']
g = foot & np.isfinite(Pint['sig_all'])
pat = {p: float(np.corrcoef(Pint[f'sig_{p}'][g], Pint['sig_all'][g])[0, 1]) for p in PARTS}

# the check against Figure 3
check = None
if os.path.exists(FIG3):
    f3 = xr.open_dataset(FIG3)
    check = {}
    for q, name in (('sig', 'ser_sig_it'), ('del', 'ser_del_it'), ('z', 'ser_z_it'), ('T', 'ser_T_it')):
        if name in f3:
            a, b_ = J(ser[f'{q}_all']), J(f3[name].values)
            check[q] = (a, b_, float(np.nanmax(np.abs(ser[f'{q}_all'] - f3[name].values))))

# profiles (W m-3)
prof = {(p, c): prof_num[(p, c)]/np.maximum(prof_den[c], 1e-9) for p in PARTS for c in CLASS}

# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
ds = xr.Dataset(
    data_vars={**{f'ser_{k_}': (('time',), v) for k_, v in ser.items()},
               **{f'ser_{k_}_{c}': (('time',), v) for (k_, c), v in ser_c.items()},
               **{f'J_{q}': (('y', 'x'), Pint[q]) for q in list(Q) + ['sig_all']},
               **{f'ke_{p}': (('y', 'x'), ke[p]) for p in PARTS},
               **{f's2_{p}': (('y', 'x'), s2map[p]) for p in PARTS + ['x']},
               **{f'shifted_{p}_t': (('shift', 'time'), (shifted[p]/foot_area).astype(np.float32)) for p in PARTS + ['all']},
               **{f'shifted_{p}_{w}': (('shift',), place[(p, w)]['values']) for p in PARTS + ['all'] for w in ('work', 'after')},
               **{f'prof_sig_{p}_{c}': (('zbin',), prof[(p, c)]) for p in PARTS for c in CLASS},
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'foot': (('y', 'x'), foot.astype(np.int8)), 'dc_FULL': (('y', 'x'), dc_full)},
    coords={'time': t, 'shift': SHIFTS_H, 'zbin': 0.5*(Z_EDGES[1:] + Z_EDGES[:-1])},
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF), z_edges=str(list(Z_EDGES)),
               table=str({f'{q}|{s}|{w}': v for (q, s, w), v in tab.items()}),
               table_class=str({f'{q}|{s}|{c}': v for (q, s, c), v in tab_c.items()}),
               place=str({f'{p}|{w}': (d['actual'], d['place'], d['amp']) for (p, w), d in place.items()}),
               ke=str(KE), rms_strain=str(RMS), pattern=str(pat), n_shifts=int(SHIFTS_H.size),
               share_bc_P=share(tab[('P', 'bc', 'work')], tab[('P', 'all', 'work')]),
               share_bc_sig=share(tab[('sig', 'bc', 'work')], tab[('sig', 'all', 'work')]),
               share_bc_T=share(tab[('T', 'bc', 'work')], tab[('T', 'all', 'work')])))
ds.to_netcdf(OUTDIR + 'figS10_tide_parts.nc')

L = ['figS10_compute.py. The barotropic tide and the internal tide in the work on the wake', '',
     'The tide is the semidiurnal band of the velocity of TIDE. Barotropic tide: its depth mean. Internal tide: the',
     'deviation from the depth mean. The works are linear in the tide, so the two parts add up to the work of Figure 3.',
     f'Footprint {foot_area/1e6:.0f} km2. Work window {T_STORM0} to {T_WAKE1}. The wind subsides {T_OFF}.', '']
L += ['1. WORKS OVER THE FOOTPRINT (J m-2). Columns: whole tide, barotropic tide, internal tide, share of the internal tide']
for w, name in (('work', 'the work window'), ('storm', 'the storm window'), ('after', 'after the wind subsides')):
    L.append(f'   {name}')
    for q in TERMS:
        a, b_, c_ = tab[(q, 'all', w)], tab[(q, 'bt', w)], tab[(q, 'bc', w)]
        L.append(f'      {TERM_NAME[q]:<26s} {a:+9.1f} {b_:+9.1f} {c_:+9.1f}    {share(c_, a):6.2f}')
L += ['', '2. BY CLASS OF THE FLOW, work window (J m-2, mean over the class). Whole tide, barotropic tide, internal tide, share']
for c in CLASS:
    L.append(f'   {CLASS_NAME[c]} ({class_area[c]/1e6:.0f} km2)')
    for q in TERMS:
        a, b_, c_ = tab_c[(q, 'all', c)], tab_c[(q, 'bt', c)], tab_c[(q, 'bc', c)]
        L.append(f'      {TERM_NAME[q]:<26s} {a:+9.1f} {b_:+9.1f} {c_:+9.1f}    {share(c_, a):6.2f}')
L += ['', '3. LOSS OF THE TIDE OVER THE WORK ON THE WAKE (-T/P), work window']
for s in SRC:
    L.append(f'   {PART_NAME[s]:<48s} {share(-tab[("T", s, "work")], tab[("P", s, "work")]):6.2f}')
L += ['', '4. THE TWO PARTS OF THE TIDE OVER THE FOOTPRINT, mean of the work window',
      f'   kinetic energy (J m-2): barotropic tide {KE["bt"]:.1f}, internal tide {KE["bc"]:.1f}, '
      f'internal share {KE["bc"]/(KE["bt"] + KE["bc"]):.2f}',
      f'   rms strain (1e-6 s-1): whole tide {RMS["all"]*1e6:.3f}, barotropic tide {RMS["bt"]*1e6:.3f}, '
      f'internal tide {RMS["bc"]*1e6:.3f}',
      f'   share of the squared strain: barotropic {S2["bt"]/S2["all"]:.2f}, internal {S2["bc"]/S2["all"]:.2f}, '
      f'cross term {2*S2["x"]/S2["all"]:.2f}']
L += ['', f'5. THE {SHIFTS_H.size} SHIFTED TIDES, strain work on the wake with the tide (J m-2). Actual tide, its place '
          f'(1 is the largest work), half range of the shifts']
for w, name in (('work', 'the work window'), ('after', 'after the wind subsides')):
    L.append(f'   {name}')
    for p in ('all', 'bt', 'bc'):
        d = place[(p, w)]
        L.append(f'      {PART_NAME[p]:<48s} {d["actual"]:+9.1f}   place {d["place"]:3d} of {SHIFTS_H.size}   '
                 f'half range {d["amp"]:8.1f}')
L += ['', '6. STRAIN WORK BY DEPTH (1e-6 W m-3, mean of the work window). Depth bin, barotropic and internal tide, '
          'below then above resonance']
for i in range(nzb):
    L.append(f'   {Z_EDGES[i]:5.0f} to {Z_EDGES[i + 1]:5.0f} m   '
             + '   '.join(f'{prof[(p, c)][i]*1e6:+8.3f}' for c in CLASS for p in PARTS))
L += ['', '7. MAPS. Pattern correlation over the footprint of the strain work of each part with that of the whole tide',
      f'   barotropic tide {pat["bt"]:.2f}, internal tide {pat["bc"]:.2f}']
L += ['', '8. CHECK. The sum of the two parts against fig3_strain.nc (J m-2 here, J m-2 in Figure 3, largest '
          'difference of the hourly series in W m-2)']
if check is None:
    L.append('   fig3_strain.nc was not found, no check')
else:
    for q, (a, b_, d) in check.items():
        L.append(f'   {TERM_NAME[q]:<26s} {a:+9.2f} {b_:+9.2f}   {d:.2e}')
L += ['', 'READING',
      f'   The internal tide did {share(tab[("P", "bc", "work")], tab[("P", "all", "work")]):.2f} of the work on the wake '
      f'({tab[("P", "bc", "work")]:+.0f} of {tab[("P", "all", "work")]:+.0f} J m-2) and '
      f'{share(tab[("sig", "bc", "work")], tab[("sig", "all", "work")]):.2f} of the strain work. '
      f'The barotropic tide did {share(tab[("P", "bt", "work")], tab[("P", "all", "work")]):.2f}.',
      f'   The internal tide holds {KE["bc"]/(KE["bt"] + KE["bc"]):.2f} of the semidiurnal kinetic energy and '
      f'{S2["bc"]/S2["all"]:.2f} of the squared strain over the footprint.']
with open(OUTDIR + 'figS10_tide_parts_report.txt', 'w') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nsaved {OUTDIR}figS10_tide_parts.nc and the report, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min')