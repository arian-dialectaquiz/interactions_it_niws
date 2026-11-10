"""
region_compute.py  (full_v5, revision 1)
----------------------------------------
The maps of Figures 2 to 5 over the whole bight, from the region cache of
extract_region.py (6 km blocks, all levels). The statistics of each figure
stay on the 3 km slope tiles and their own compute scripts. This script forms,
with the same bands, windows, filters and definitions,

  Figure 2   near-inertial kinetic energy of the wake with the tide
             (u_FULL - u_TIDE) and without it (u_WIND - u_BACK), window means
  Figure 2   detuning of the balanced flow, f_eff - omega/2 with
             f_eff = f + zeta/2, from the wake-window and storm-window mean
             velocity of the upper 150 m coarse-grained to 25 km, FULL and WIND
  Figure 3   tidal strain work on the wake accumulated over the work window,
             strain of the internal tide u_TIDE in the semidiurnal band
  Figure 4   Z = sum_z S_r A*^2 dz over the wake window, whose angle gives
             the phase of the wake against the tide and whose real part the
             strain work
  Figure 5   detuning of the wake from the drift of its own phase, and the
             anticlockwise strain amplitude |S_r| of the internal tide

Output  ./region_out/region_maps.nc

Run under nohup. Row bands of the region grid, CHUNK_Z levels at a time, about
the cost of one Figure 4 compute.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------
# paths and settings
# --------------------------------------------------------------------------
ROOT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/'
HIS = ROOT + 'avg_FULL_HC.nc'
NEW_FULL = '../new_full/'
CACHE = {r: NEW_FULL + f'region_cache_{r}/' for r in ('BACK', 'TIDE', 'WIND', 'FULL')}
FIG2 = './fig2_out/fig2_wake.nc'
OUTDIR = './region_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)
M2_BAND = (0.95, 1.05)
LP_H = 72.0
Z_UPPER = -150.0
L_MESO = 25.0e3
CHUNK_Z = 4
ROW_BANDS = 4

# --------------------------------------------------------------------------
# grid of the region cache, the records, the windows of Figure 2
# --------------------------------------------------------------------------
geo = np.load(CACHE['FULL'] + 'geometry.npz', allow_pickle=True)
for r in RUNS:
    g = np.load(CACHE[r] + 'geometry.npz', allow_pickle=True)
    assert np.array_equal(g['time'], geo['time']) and int(g['nz']) == int(geo['nz']), \
        f'the region cache of {r} is not on the geometry of FULL'
SUB, nz = int(geo['SUB']), int(geo['nz'])
levels = geo['levels']
lon, lat, h, f_r = geo['lon'], geo['lat'], geo['h'], geo['f']
pm, pn, mask = geo['pm'], geo['pn'], geo['mask'].astype(bool)
ny, nx = h.shape
t = np.asarray(geo['time']).astype('datetime64[ns]')
t = ((t + np.timedelta64(30, 'm')).astype('datetime64[h]')).astype('datetime64[ns]')
nt = t.size
dt_h = float((t[1] - t[0])/np.timedelta64(1, 'h'))
dt_s = dt_h*3600.0
nyq = 0.5/dt_h
tsec = ((t - t[0])/np.timedelta64(1, 's')).astype(np.float64)

ref = xr.open_dataset(HIS)
hc, Vt = float(ref.hc.values), int(ref.Vtransform.values)
s_w, Cs_w = ref.s_w.values, ref.Cs_w.values
nz_full = ref.sizes['s_rho']
stride = int(levels[1] - levels[0]) if levels.size > 1 else 1

f2 = xr.open_dataset(FIG2)
WINDOWS = eval(f2.attrs['windows'])
T_STORM0, T_WAKE1 = eval(f2.attrs['work_window'])
T_OFF = np.datetime64(f2.attrs['t_off'])
W = {k: (t >= np.datetime64(a0)) & (t < np.datetime64(a1)) for k, (a0, a1) in WINDOWS.items()}
WORK = (t >= np.datetime64(T_STORM0)) & (t < np.datetime64(T_WAKE1))
WAKE = (t >= T_OFF) & (t < np.datetime64(T_WAKE1))
print(f'region {ny} x {nx} at {SUB} km, {nz} levels, {nt} records, windows {WINDOWS}', flush=True)


def z_w_of(hh):
    s, C = s_w[:, None, None], Cs_w[:, None, None]
    return hh*((hc*s + hh*C)/(hc + hh)) if Vt == 2 else hc*s + (hh - hc)*C


zw_all = z_w_of(np.where(mask, h, 1.0))                                   # (nz_full + 1, ny, nx)
zr_all = 0.5*(zw_all[1:] + zw_all[:-1])
# thickness carried by each kept level (its own layer and the skipped ones below the next kept level)
dz_kept = np.stack([zw_all[min(k + stride, nz_full)] - zw_all[k] for k in levels])
dz = np.moveaxis(dz_kept, 0, -1).astype(np.float32)                        # (ny, nx, nz)
zr = np.moveaxis(zr_all[levels], 0, -1)
w150 = (dz*(zr > Z_UPPER)).astype(np.float32)
w150 = w150/np.maximum(w150.sum(axis=-1, keepdims=True), 1e-6)
dxi, deta = 1.0/pm, 1.0/pn


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


def d_xi(a, dxi_):
    out = np.full_like(a, np.nan)
    out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2])/(2*dxi_[None, :, 1:-1, None])
    return out


def d_eta(a, deta_):
    out = np.full_like(a, np.nan)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2])/(2*deta_[None, 1:-1, :, None])
    return out


def d_z(a, z):
    out = np.empty_like(a)
    zz = z[None]
    out[..., 1:-1] = (a[..., 2:] - a[..., :-2])/(zz[..., 2:] - zz[..., :-2])
    out[..., 0] = (a[..., 1] - a[..., 0])/(zz[..., 1] - zz[..., 0])
    out[..., -1] = (a[..., -1] - a[..., -2])/(zz[..., -1] - zz[..., -2])
    return out


MM = {r: {v: np.load(CACHE[r] + f'{v}.npy', mmap_mode='r') for v in ('u', 'v', 'w')} for r in RUNS}
car_half = np.exp(-0.5j*OMEGA_M2*tsec).astype(np.complex64)[:, None, None, None]
car_full = np.exp(-1.0j*OMEGA_M2*tsec).astype(np.complex64)[:, None, None, None]

# --------------------------------------------------------------------------
# pre-pass. The balanced vorticity of FULL and WIND in the wake and storm windows
# --------------------------------------------------------------------------
zeta = {}
for run in ('FULL', 'WIND'):
    for win in ('wake', 'storm'):
        kw = np.where(W[win])[0]
        Um = (np.asarray(MM[run]['u'][kw[0]:kw[-1] + 1]).mean(axis=0)*w150).sum(-1)
        Vm = (np.asarray(MM[run]['v'][kw[0]:kw[-1] + 1]).mean(axis=0)*w150).sum(-1)
        zt = np.full((ny, nx), np.nan)
        zt[1:-1, 1:-1] = ((Vm[1:-1, 2:] - Vm[1:-1, :-2])/(2*dxi[1:-1, 1:-1])
                          - (Um[2:, 1:-1] - Um[:-2, 1:-1])/(2*deta[1:-1, 1:-1]))
        zeta[(run, win)] = np.where(mask, zt, np.nan)
sig_px = 0.5*L_MESO/float(np.nanmean(dxi))
dc, zof = {}, {}
for key, z_ in zeta.items():
    g = np.isfinite(z_)
    num = gaussian_filter(np.where(g, z_, 0.0), sig_px, mode='nearest')
    den = gaussian_filter(g.astype(float), sig_px, mode='nearest')
    zs = np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)
    zof[key] = zs/f_r
    dc[key] = (np.abs(f_r + 0.5*zs) - 0.5*OMEGA_M2)*RAD2CPH                 # cph
print('pre-pass done', flush=True)

# --------------------------------------------------------------------------
# the pass, row bands and level chunks
# --------------------------------------------------------------------------
Ew = {(p, w): np.zeros((ny, nx), np.float32) for p in ('tide', 'notide') for w in WINDOWS}
Pint = {q: np.zeros((ny, nx), np.float32) for q in ('sig_it', 'del_it', 'z_it')}
Zsum = np.zeros((ny, nx), np.complex64)
Fsum = np.zeros((ny, nx), np.float64)
Esum = np.zeros((ny, nx), np.float64)
Sr2 = np.zeros((ny, nx), np.float64)
rows = np.array_split(np.arange(ny), ROW_BANDS)
t0 = pd.Timestamp.now()
for ib, rr in enumerate(rows):
    r0, r1 = int(rr[0]), int(rr[-1]) + 1
    re0, re1 = max(r0 - 1, 0), min(r1 + 1, ny)                              # one row of halo for gradients
    sl_r = slice(r0 - re0, r0 - re0 + (r1 - r0))
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        z0e, z1e = max(z0 - 1, 0), min(z1 + 1, nz)
        sl = slice(z0 - z0e, z0 - z0e + (z1 - z0))
        dzc = dz[re0:re1, :, z0:z1]
        zc_e = zr[re0:re1, :, z0e:z1e]
        u = {r: np.nan_to_num(np.asarray(MM[r]['u'][:, re0:re1, :, z0e:z1e], dtype=np.float32)) for r in RUNS}
        v = {r: np.nan_to_num(np.asarray(MM[r]['v'][:, re0:re1, :, z0e:z1e], dtype=np.float32)) for r in RUNS}
        # the two wakes in the near-inertial band
        au, av = bp(u['FULL'] - u['TIDE'], 'ni'), bp(v['FULL'] - v['TIDE'], 'ni')
        ku, kv = bp(u['WIND'] - u['BACK'], 'ni')[..., sl], bp(v['WIND'] - v['BACK'], 'ni')[..., sl]
        for p, (pu, pv) in (('tide', (au[..., sl], av[..., sl])), ('notide', (ku, kv))):
            e = 0.5*RHO0*((pu*pu + pv*pv)*dzc).sum(axis=-1)                   # (nt, nr, nx)
            for w in WINDOWS:
                Ew[(p, w)][r0:r1] += e[W[w]].mean(axis=0)[sl_r]
        del ku, kv
        # the internal tide in the semidiurnal band and its works on the wake
        bu, bv = bp(u['TIDE'], 'M2'), bp(v['TIDE'], 'M2')
        del u, v
        ux, uy, vx, vy = d_xi(bu, dxi[re0:re1]), d_eta(bu, deta[re0:re1]), d_xi(bv, dxi[re0:re1]), d_eta(bv, deta[re0:re1])
        sn, ss, dl = (ux - vy)[..., sl], (vx + uy)[..., sl], (ux + vy)[..., sl]
        del ux, uy, vx, vy
        aw = np.nan_to_num(np.asarray(MM['FULL']['w'][:, re0:re1, :, z0e:z1e], dtype=np.float32)
                           - np.asarray(MM['TIDE']['w'][:, re0:re1, :, z0e:z1e], dtype=np.float32))
        aw = bp(aw, 'ni')[..., sl]
        bz_u, bz_v = d_z(bu, zc_e)[..., sl], d_z(bv, zc_e)[..., sl]
        a_u, a_v = au[..., sl], av[..., sl]
        q1, q2, q0 = a_u*a_u - a_v*a_v, 2*a_u*a_v, a_u*a_u + a_v*a_v
        term = {'sig_it': -0.5*RHO0*(sn*q1 + ss*q2), 'del_it': -0.5*RHO0*dl*q0,
                'z_it': -RHO0*aw*(a_u*bz_u + a_v*bz_v)}
        for q in Pint:
            Pint[q][r0:r1] += (np.nan_to_num((term[q][WORK]*dzc).sum(axis=-1)).sum(axis=0)*dt_s)[sl_r]
        del term, q1, q2, q0, aw, bz_u, bz_v, dl
        # demodulation, the phase and the drift
        A = lp_complex((au + 1j*av)*car_half)[..., sl]
        Sr = lp_complex((sn + 1j*ss)*car_full)
        del au, av, sn, ss, bu, bv
        Zsum[r0:r1] += (((Sr[WAKE]*np.conj(A[WAKE])**2)*dzc).sum(axis=-1).sum(axis=0))[sl_r]
        dA = np.empty_like(A)
        dA[1:-1] = (A[2:] - A[:-2])/(2*dt_s)
        dA[0], dA[-1] = (A[1] - A[0])/dt_s, (A[-1] - A[-2])/dt_s
        Fsum[r0:r1] += ((((np.conj(A[WAKE])*dA[WAKE]).imag)*dzc).sum(axis=-1).sum(axis=0))[sl_r]
        Esum[r0:r1] += (((np.abs(A[WAKE])**2)*dzc).sum(axis=-1).sum(axis=0))[sl_r]
        Sr2[r0:r1] += (((np.abs(Sr[WORK])**2)*dzc).sum(axis=-1).mean(axis=0))[sl_r]
        del A, Sr, dA
    print(f'row band {ib + 1} of {ROW_BANDS}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)

nwake = int(WAKE.sum())
Z_win = Zsum/nwake
with np.errstate(divide='ignore', invalid='ignore'):
    drift = np.where(Esum > 0, Fsum/Esum, np.nan)*RAD2CPH                  # cph
    Sr_amp = np.sqrt(2.0*Sr2/np.maximum(dz.sum(axis=-1), 1e-6))*RAD2CPH     # cph
    Ew_win = 0.5*RHO0*Esum/nwake
land = ~mask
for a in (Z_win, drift, Sr_amp, Ew_win):
    a[land] = np.nan

ds = xr.Dataset(
    data_vars={**{f'Ew_{p}_{w}': (('y', 'x'), np.where(mask, Ew[(p, w)], np.nan)) for (p, w) in Ew},
               **{f'dc_{r}_{w}': (('y', 'x'), dc[(r, w)].astype(np.float32)) for (r, w) in dc},
               **{f'zof_{r}_{w}': (('y', 'x'), zof[(r, w)].astype(np.float32)) for (r, w) in zof},
               **{f'Pint_{q}': (('y', 'x'), np.where(mask, Pint[q], np.nan)) for q in Pint},
               'Zwin_re': (('y', 'x'), Z_win.real.astype(np.float32)),
               'Zwin_im': (('y', 'x'), Z_win.imag.astype(np.float32)),
               'drift_cph': (('y', 'x'), drift.astype(np.float32)), 'Sr_cph': (('y', 'x'), Sr_amp.astype(np.float32)),
               'Ew_win': (('y', 'x'), Ew_win.astype(np.float32)),
               'lon': (('y', 'x'), lon), 'lat': (('y', 'x'), lat), 'h': (('y', 'x'), h),
               'f': (('y', 'x'), f_r), 'mask': (('y', 'x'), mask.astype(np.int8))},
    attrs=dict(sub_km=SUB, windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF)[:16],
               ni_band=str(NI_BAND), m2_band=str(M2_BAND), lp_h=LP_H, omega_M2=OMEGA_M2, rho0=RHO0,
               note='maps over the whole domain for Figures 2 to 5, statistics stay on the slope tiles'))
ds.to_netcdf(OUTDIR + 'region_maps.nc')
print('saved', OUTDIR + 'region_maps.nc', flush=True)