"""
fig_04_compute.py  (full_v5, revision 4)
----------------------------------------
Figure 4. The phase of the wake against the tide sets the sign of the work.

With U = u + iv the wake with the tide (near-inertial band of u_FULL - u_TIDE)
and sigma_T = sigma_n + i sigma_s the strain of the internal tide (semidiurnal
band of u_TIDE, the source Figure 3 selected), inertial motion south of the
equator rotates anticlockwise, U = A exp(+i omega t / 2) with omega the M2
frequency, and the part of the strain that does steady work on it is the
component rotating the same way at omega, sigma_T = S_r exp(+i omega t)
+ S_l exp(-i omega t). The strain work (Figure 3) is then

    P_sigma = -(rho0/2) Re[ S_r A*^2 ] = (rho0/2) |S_r| |A|^2 cos(Phi),
    Phi = pi + arg(S_r) - 2 arg(A)                   (Young et al. 2008),

so the sign of the work is the cosine of one angle. The amplitudes are formed
by complex demodulation, A = lowpass[U exp(-i omega t / 2)] and
S_r = lowpass[sigma_T exp(-i omega t)], with a 72 h cut-off that keeps the
near-inertial band of Figure 2 and removes the clockwise parts. Over a water
column the phase that matters is that of Z = sum_z S_r A*^2 dz, because
-(rho0/2) Re Z is the strain work of the column and (rho0/2) |Z| the largest
work the column could do at any phase. Phi = pi + arg Z and cos(Phi) is the
phase efficiency of the column.

Also formed here for Figure 5: the detuning of the wake measured from the drift
of its own phase, omega_w - omega/2 = <Im(A* dA/dt)> / <|A|^2>.

Tests
  the cosine. The tendency of the wake kinetic energy of each block in the
    wake window, divided by (rho0/2)|Z| of the block, against Phi in bins,
    with cos(Phi) drawn with no fit and with a fitted amplitude and offset.
  the lock across the footprint. The resultant length of exp(i Phi) over
    the footprint blocks through time for the wake with the tide and for the
    kinematic pairing (the wake of WIND - BACK against the same strain), with
    the Rayleigh level for the number of blocks, and the amplitude-weighted
    phase of the footprint through time, arg(-sum_b Z_b).
  the lock in time. For each block the coherence of its phase over the wake
    window, |sum_t Z_b| / sum_t |Z_b|, which is one for a phase held fixed and
    falls as the phase turns, against the detuning of the flow, beside the
    same for the kinematic pairing and the value a phase drifting at twice the
    detuning would give over the window, |sinc(2 Delta_c T)|.
  the sense of rotation. |S_r| against |S_l| over the footprint, since only
    the anticlockwise strain can hold a phase against the wake.
  consistency. The strain work rebuilt from Z against the series of Figure 3.
  the detuning turns the phase. Phi of the blocks against the detuning of
    the balanced flow (Figure 2) at the hour the wind stops and 2, 4 and 6
    days later (amplitude-weighted circular means by bin with bootstrap bands),
    against the prediction that Phi drifts at twice the detuning from the
    snapshot at the wind stop, with no free parameter. The same through time
    as Phi against detuning and hours.
  the bridge to Figure 5. cos(Phi) of the wake window against the detuning
    of the balanced flow, in bins with block-bootstrap bands.

Inputs
  ../new_full/fig6_cache_<RUN>/   u and v on the slope tiles, four runs
  ./fig2_out/fig2_wake.nc         geometry, detuning, wake series
  ./fig3_out/fig3_strain.nc       the strain work series (consistency)
  ./fig3_out/fig3_blocks.csv      the block table of Figure 3 (merged)

Outputs (in OUTDIR)
  fig4_phase.nc  fig4_blocks.csv  fig4_phase_report.txt

Run under nohup. One pass over the tiles, CHUNK_Z levels at a time.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr, theilslopes

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
OUTDIR = './fig4_out/'
os.makedirs(OUTDIR, exist_ok=True)

RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
T_M2_H = 12.4206012
OMEGA_M2 = 2*np.pi/(T_M2_H*3600.0)
RHO0 = 1025.0
RAD2CPH = 3600.0/(2*np.pi)
NI_BAND = (0.80, 1.20)                             # times omega_M2/2, as Figure 2
M2_BAND = (0.95, 1.05)                             # times omega_M2
LP_H = 72.0                                        # demodulation cut-off (h)
CHUNK_Z = 5
SMOOTH_H = 25
LAT_EDGES = np.arange(-33.0, -20.99, 0.25)         # the Hovmoeller bins
PHI_BINS = 12
DC_EDGES = np.arange(-0.008, 0.0061, 0.001)        # cph
MIN_BIN = 5                                        # blocks a bin must hold
N_BOOT = 2000
N_SHIFT = 500
AMP_FLOOR = 0.10                                   # of the median |Z| for a phase to count
SNAP_DAYS = [0, 2, 4, 6]                           # snapshots after the wind stops
rng = np.random.default_rng(13)

# --------------------------------------------------------------------------
# grid, geometry, Figure 2 and Figure 3 fields
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
dt_s = dt_h*3600.0
nyq = 0.5/dt_h
ntiles = int(geo['ntiles'])
ny_s, nx_s = eta_s.size, xi_s.size
tsec = ((t - t[0])/np.timedelta64(1, 's')).astype(np.float64)
print(f'{ntiles} tiles, {nt} records, sub grid {ny_s} x {nx_s}', flush=True)

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
WAKE = (t >= T_OFF) & (t < np.datetime64(T_WAKE1))
BLOCK = int(f2.attrs['block_points'])
blk = (np.arange(ny_s)[:, None]//BLOCK)*1000 + (np.arange(nx_s)[None, :]//BLOCK)
foot_area = float(area_s[foot].sum())
f3 = xr.open_dataset(FIG3) if os.path.exists(FIG3) else None
blocks3 = pd.read_csv(FIG3_BLOCKS) if os.path.exists(FIG3_BLOCKS) else None

# blocks of the footprint, the same ids as Figures 2 and 3
ids = np.unique(blk[foot])
counts = {b: int(np.sum(foot & (blk == b))) for b in ids}
ids = np.array([b for b in ids if counts[b] >= BLOCK*BLOCK//4])
blk_index = {b: i for i, b in enumerate(ids)}
nblk = ids.size
blk_of = np.vectorize(lambda b: blk_index.get(b, -1))(blk)
blk_of = np.where(foot, blk_of, -1)
print(f'footprint {foot_area/1e6:.0f} km2, {nblk} blocks, windows {WINDOWS}', flush=True)


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
    out = np.full_like(a, np.nan)
    out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2])/(2*dxi[None, :, 1:-1, None])
    return out


def d_eta(a, deta):
    out = np.full_like(a, np.nan)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2])/(2*deta[None, 1:-1, :, None])
    return out


def load(run, var, k, z0, z1):
    a = np.load(cache[run] + f'{var}_t{k:03d}.npy', mmap_mode='r')
    return np.nan_to_num(np.asarray(a[..., z0:z1], dtype=np.float32))


# the demodulation carriers
car_half = np.exp(-0.5j*OMEGA_M2*tsec).astype(np.complex64)[:, None, None, None]
car_full = np.exp(-1.0j*OMEGA_M2*tsec).astype(np.complex64)[:, None, None, None]
car_back = np.exp(+1.0j*OMEGA_M2*tsec).astype(np.complex64)[:, None, None, None]

# --------------------------------------------------------------------------
# the pass over the tiles
# --------------------------------------------------------------------------
Z = np.zeros((nt, ny_s, nx_s), np.complex64)       # sum_z S_r A*^2 dz, the wake with the tide
Zk = np.zeros((nt, ny_s, nx_s), np.complex64)      # the same for the wake of WIND - BACK
E = np.zeros((nt, ny_s, nx_s), np.float32)         # (rho0/2) sum |A|^2 dz, J m-2
F = np.zeros((nt, ny_s, nx_s), np.float32)         # sum Im(A* dA/dt) dz, for the drift
Sr2 = np.zeros((ny_s, nx_s), np.float64)           # window mean of sum |S_r|^2 dz
Sl2 = np.zeros((ny_s, nx_s), np.float64)
Hcol = np.zeros((ny_s, nx_s), np.float64)
valid = np.zeros((ny_s, nx_s), bool)

t0 = pd.Timestamp.now()
for k in range(ntiles):
    ys, xs, b0, b1, c0, rows, cols, jy, jx = tile_geometry(k)
    h_t = h[np.ix_(jy, jx)]
    dz = np.moveaxis(np.diff(z_of(s_w, Cs_w, h_t), axis=0), 0, -1).astype(np.float32)
    dxi, deta = SUB/pm[np.ix_(jy, jx)], SUB/pn[np.ix_(jy, jx)]
    nz = dz.shape[-1]
    ny_t, nx_t = h_t.shape
    Zt = np.zeros((nt, ny_t, nx_t), np.complex64)
    Zkt = np.zeros((nt, ny_t, nx_t), np.complex64)
    Et = np.zeros((nt, ny_t, nx_t), np.float32)
    Ft = np.zeros((nt, ny_t, nx_t), np.float32)
    sr2 = np.zeros((ny_t, nx_t), np.float64)
    sl2 = np.zeros((ny_t, nx_t), np.float64)
    ok_t = np.ones((ny_t, nx_t), bool)
    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        dzc = dz[..., z0:z1]
        u = {r: load(r, 'u', k, z0, z1) for r in RUNS}
        v = {r: load(r, 'v', k, z0, z1) for r in RUNS}
        # the two wakes, demodulated at omega/2
        A = lp_complex((bp(u['FULL'] - u['TIDE'], 'ni') + 1j*bp(v['FULL'] - v['TIDE'], 'ni'))*car_half)
        K = lp_complex((bp(u['WIND'] - u['BACK'], 'ni') + 1j*bp(v['WIND'] - v['BACK'], 'ni'))*car_half)
        # the strain of the internal tide, demodulated at omega in both senses
        bu, bv = bp(u['TIDE'], 'M2'), bp(v['TIDE'], 'M2')
        del u, v
        ux, uy, vx, vy = d_xi(bu, dxi), d_eta(bu, deta), d_xi(bv, dxi), d_eta(bv, deta)
        sig = (ux - vy) + 1j*(vx + uy)
        del bu, bv, ux, uy, vx, vy
        bad = ~np.isfinite(sig[0, ..., 0])
        sig = np.nan_to_num(sig)
        Sr = lp_complex(sig*car_full)
        Sl = lp_complex(sig*car_back)
        del sig
        Ac2 = np.conj(A)**2
        Zt += ((Sr*Ac2)*dzc).sum(axis=-1)
        Zkt += ((Sr*np.conj(K)**2)*dzc).sum(axis=-1)
        Et += 0.5*RHO0*((np.abs(A)**2)*dzc).sum(axis=-1)
        dA = np.empty_like(A)
        dA[1:-1] = (A[2:] - A[:-2])/(2*dt_s)
        dA[0], dA[-1] = (A[1] - A[0])/dt_s, (A[-1] - A[-2])/dt_s
        Ft += ((np.conj(A)*dA).imag*dzc).sum(axis=-1)
        sr2 += ((np.abs(Sr[WORK])**2)*dzc).sum(axis=-1).mean(axis=0)
        sl2 += ((np.abs(Sl[WORK])**2)*dzc).sum(axis=-1).mean(axis=0)
        ok_t &= ~bad
        del A, K, Sr, Sl, Ac2, dA
    Z[:, b0:b1, cols] = Zt[:, rows]
    Zk[:, b0:b1, cols] = Zkt[:, rows]
    E[:, b0:b1, cols] = Et[:, rows]
    F[:, b0:b1, cols] = Ft[:, rows]
    Sr2[b0:b1, cols] = sr2[rows]
    Sl2[b0:b1, cols] = sl2[rows]
    Hcol[b0:b1, cols] = dz.sum(axis=-1)[rows]
    valid[b0:b1, cols] = ok_t[rows]
    print(f'tile {k + 1} of {ntiles}, {(pd.Timestamp.now() - t0).total_seconds()/60:.1f} min',
          flush=True)

good = foot & valid
Z[:, ~valid] = 0.0
Zk[:, ~valid] = 0.0


def smooth(x, n=SMOOTH_H, axis=0):
    k_ = np.ones(n)/n
    return np.apply_along_axis(lambda c: np.convolve(np.pad(c, n//2, mode='edge'), k_,
                                                     mode='valid')[:c.size], axis, x)


# --------------------------------------------------------------------------
# maps and the Hovmoeller
# --------------------------------------------------------------------------
Z_win = Z[WAKE].mean(axis=0)                                   # the effective phase of the window
Zk_win = Zk[WAKE].mean(axis=0)
snaps = {d: Z[int(np.argmin(np.abs(t - (T_OFF + np.timedelta64(24*d, 'h')))))] for d in SNAP_DAYS}
# the detuning of the wake measured from its own phase drift, wake window, cph (for Figure 5)
with np.errstate(divide='ignore', invalid='ignore'):
    drift_map = (F[WAKE].sum(axis=0)/(E[WAKE].sum(axis=0)/(0.5*RHO0)))*RAD2CPH
drift_map = np.where(valid & (E[WAKE].sum(axis=0) > 0), drift_map, np.nan).astype(np.float32)
Ew_map = np.where(valid, E[WAKE].mean(axis=0), np.nan).astype(np.float32)
amp_med = float(np.median(np.abs(Z_win[good])))
strain_amp = {'r': np.sqrt(Sr2/np.maximum(Hcol, 1e-6)), 'l': np.sqrt(Sl2/np.maximum(Hcol, 1e-6))}

lat_mid = 0.5*(LAT_EDGES[1:] + LAT_EDGES[:-1])
hov = np.zeros((nt, lat_mid.size), np.complex64)
hov_n = np.zeros(lat_mid.size, int)
for i in range(lat_mid.size):
    m = slope_s & valid & (lat_s >= LAT_EDGES[i]) & (lat_s < LAT_EDGES[i + 1])
    hov_n[i] = int(m.sum())
    if m.sum() >= 10:
        hov[:, i] = (Z[:, m]*area_s[m]).sum(axis=1)/area_s[m].sum()

# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
Zb = np.zeros((nt, nblk), np.complex64)
Zkb = np.zeros((nt, nblk), np.complex64)
Eb = np.zeros((nt, nblk))
Fb = np.zeros((nt, nblk))
Ab = np.zeros(nblk)
rows_b = []
for b, i in blk_index.items():
    m = good & (blk == b)
    if not m.any():
        continue
    a = area_s[m]
    Ab[i] = a.sum()
    Zb[:, i] = (Z[:, m]*a).sum(axis=1)/a.sum()
    Zkb[:, i] = (Zk[:, m]*a).sum(axis=1)/a.sum()
    Eb[:, i] = (E[:, m]*a).sum(axis=1)/a.sum()
    Fb[:, i] = (F[:, m]*a).sum(axis=1)/a.sum()
    zw = (Z_win[m]*a).sum()/a.sum()
    zsn = {d: (snaps[d][m]*a).sum()/a.sum() for d in SNAP_DAYS}
    rows_b.append(dict(block=int(b), lon=float(np.mean(lon_s[m])), lat=float(np.mean(lat_s[m])),
                       **{f'phi_d{d}': float(np.angle(-zsn[d])) for d in SNAP_DAYS},
                       **{f'amp_d{d}': float(np.abs(zsn[d]))*0.5*RHO0 for d in SNAP_DAYS},
                       n=int(m.sum()), dc=float(np.sum(dc_full[m]*a)/a.sum()),
                       free_share=float(np.mean(dc_full[m] < 0)),
                       phi=float(np.angle(-zw)), cosphi=float(np.cos(np.angle(-zw))),
                       amp=float(np.abs(zw))*0.5*RHO0,
                       work_from_Z=float(-0.5*RHO0*Zb[WORK, i].real.sum()*dt_s),
                       Sr=float(np.sum(strain_amp['r'][m]*a)/a.sum()),
                       Sl=float(np.sum(strain_amp['l'][m]*a)/a.sum()),
                       drift_cph=float(Fb[WAKE, i].sum()/np.maximum(Eb[WAKE, i].sum()/(0.5*RHO0), 1e-9))*RAD2CPH,
                       Ew=float(Eb[WAKE, i].mean())))
blocks = pd.DataFrame(rows_b)
blocks['class'] = np.where(blocks.free_share > 0.5, 'free', 'trapped')
if blocks3 is not None:
    blocks = blocks.merge(blocks3[['block', 'work_sig_it', 'work_it', 'gain', 'strain_amp']], on='block',
                          how='left')
blocks.to_csv(OUTDIR + 'fig4_blocks.csv', index=False)
nb = len(blocks)
bi = np.array([blk_index[b] for b in blocks.block.values])

# --------------------------------------------------------------------------
# the cosine test. Tendency of the wake kinetic energy over (rho0/2)|Z|,
# against Phi, over blocks and hours of the wake window
# --------------------------------------------------------------------------
Es = smooth(Eb, axis=0)
dEdt = np.gradient(Es, dt_s, axis=0)                                       # W m-2
Zs = Zb
amp = 0.5*RHO0*np.abs(Zs)
phi = np.angle(-Zs)                                                        # pi + arg Z
sel = WAKE[:, None] & (amp > AMP_FLOOR*0.5*RHO0*amp_med)
sel[:, ~np.isin(np.arange(nblk), bi)] = False
with np.errstate(divide='ignore', invalid='ignore'):
    yn = np.where(sel, dEdt/amp, np.nan)                                   # dimensionless rate, s-1
phi_edges = np.linspace(-np.pi, np.pi, PHI_BINS + 1)
phi_mid = 0.5*(phi_edges[1:] + phi_edges[:-1])


def cos_bins(y, ph, s, idx_blocks):
    out = np.full((PHI_BINS, 4), np.nan)                                   # median, p05, p95, n
    med = np.full((N_BOOT, PHI_BINS), np.nan)
    draws = rng.integers(0, idx_blocks.size, size=(N_BOOT, idx_blocks.size))
    for j in range(PHI_BINS):
        m = s & (ph >= phi_edges[j]) & (ph < phi_edges[j + 1])
        vals = y[m]
        if vals.size < 24:
            continue
        out[j, 0], out[j, 3] = np.nanmedian(vals), vals.size
        # block bootstrap, the blocks resampled and their hours kept together
        per_block = [y[:, b][m[:, b]] for b in idx_blocks]
        for d_ in range(N_BOOT):
            pool = np.concatenate([per_block[q] for q in draws[d_]])
            med[d_, j] = np.nanmedian(pool) if pool.size else np.nan
        out[j, 1], out[j, 2] = np.nanpercentile(med[:, j], 5), np.nanpercentile(med[:, j], 95)
    return out


cosine = cos_bins(yn, phi, sel, bi)
ok = np.isfinite(yn)
X = np.column_stack([np.cos(phi[ok]), np.ones(ok.sum())])
coef = np.linalg.lstsq(X, yn[ok], rcond=None)[0]
cos_fit = dict(a=float(coef[0]), b=float(coef[1]))
# the amplitude the theory gives is one (the rate is P_sigma / amplitude), so a near one
# and b the mean decay rate of the wake. Correlation with a circular-shift level in time.
rho_cos = float(spearmanr(np.cos(phi[ok]), yn[ok]).correlation)
lv = []
for _ in range(N_SHIFT):
    s_ = int(rng.integers(24, nt - 24))
    ph_s = np.roll(phi, s_, axis=0)
    lv.append(abs(spearmanr(np.cos(ph_s[ok]), yn[ok]).correlation))
rho_cos_level = float(np.percentile(lv, 95))

# --------------------------------------------------------------------------
# the lock. Resultant length over the blocks through time
# --------------------------------------------------------------------------


def resultant(Zs_, floor):
    R = np.full(nt, np.nan)
    mean_phi = np.full(nt, np.nan)
    n = np.zeros(nt, int)
    for j in range(nt):
        a_ = np.abs(Zs_[j, bi])
        m = a_ > floor
        n[j] = int(m.sum())
        if m.sum() >= 5:
            e = np.exp(1j*np.angle(-Zs_[j, bi][m]))
            R[j] = np.abs(e.mean())
            mean_phi[j] = np.angle(e.mean())
    return R, mean_phi, n


floor_Z = AMP_FLOOR*np.median(np.abs(Zb[WAKE][:, bi]))
R_with, phi_with, n_with = resultant(Zb, floor_Z)
R_without, phi_without, n_without = resultant(Zkb, AMP_FLOOR*np.median(np.abs(Zkb[WAKE][:, bi])))
rayleigh = np.sqrt(-np.log(0.05)/np.maximum(n_with, 1))
lock = dict(R_with_wake=float(np.nanmean(R_with[WAKE])), R_without_wake=float(np.nanmean(R_without[WAKE])),
            R_with_storm=float(np.nanmean(R_with[WORK & ~WAKE])),
            rayleigh_wake=float(np.nanmean(rayleigh[WAKE])),
            mean_phi_wake=float(np.angle(np.nanmean(np.exp(1j*phi_with[WAKE])))),
            hours_above=int(np.nansum((R_with > rayleigh)[WAKE])), hours_wake=int(WAKE.sum()))

# weighted phase of the footprint over the window and the rose
zf = Z_win[good]
foot_phase = dict(phi=float(np.angle(-(zf*area_s[good]).sum())),
                  cos=float(np.cos(np.angle(-(zf*area_s[good]).sum()))),
                  share_pos=float(np.sum(area_s[good]*(np.cos(np.angle(-zf)) > 0))/area_s[good].sum()))
rose_edges = np.linspace(-np.pi, np.pi, 25)
rose = {}
for name, zz in (('with', Z_win), ('without', Zk_win)):
    ph = np.angle(-zz[good])
    w = np.abs(zz[good])*area_s[good]
    rose[name] = np.histogram(ph, bins=rose_edges, weights=w)[0]/w.sum()

# --------------------------------------------------------------------------
# the bridge to Figure 5. cos(Phi) against the detuning of the flow
# --------------------------------------------------------------------------
centres = 0.5*(DC_EDGES[1:] + DC_EDGES[:-1])
curve = np.full((centres.size, 4), np.nan)
for i in range(centres.size):
    m = (blocks.dc.values >= DC_EDGES[i]) & (blocks.dc.values < DC_EDGES[i + 1])
    if m.sum() < MIN_BIN:
        continue
    zz = blocks.amp.values[m]*np.exp(1j*blocks.phi.values[m])             # amplitude-weighted
    c = np.cos(np.angle(zz.sum()))
    draws = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
    bs = np.cos(np.angle(zz[draws].sum(axis=1)))
    curve[i] = [c, np.percentile(bs, 5), np.percentile(bs, 95), m.sum()]
# Phi against the detuning of the flow at the snapshots, amplitude-weighted
# circular means by bin, bootstrap bands as the spread of the mean angle


def circ_bins(ph_b, amp_b, dc_b):
    out = np.full((centres.size, 4), np.nan)
    for i in range(centres.size):
        m = (dc_b >= DC_EDGES[i]) & (dc_b < DC_EDGES[i + 1]) & np.isfinite(ph_b)
        if m.sum() < MIN_BIN:
            continue
        zz = amp_b[m]*np.exp(1j*ph_b[m])
        mu = np.angle(zz.sum())
        draws = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        dev = np.angle(zz[draws].sum(axis=1)*np.exp(-1j*mu))              # spread about the mean
        out[i] = [mu, mu + np.percentile(dev, 5), mu + np.percentile(dev, 95), m.sum()]
    return out


phase_dc = np.stack([circ_bins(blocks[f'phi_d{d}'].values, blocks[f'amp_d{d}'].values, blocks.dc.values)
                     for d in SNAP_DAYS])
# the prediction from the snapshot at the wind stop, Phi drifting at twice the detuning
pred_dc = np.stack([phase_dc[0, :, 0] - 2*(2*np.pi*centres)*(24.0*d) for d in SNAP_DAYS])
pred_dc = np.angle(np.exp(1j*pred_dc))
# the drift of the block phases between the snapshots against the detuning
dphi = np.stack([np.angle(np.exp(1j*(blocks[f'phi_d{SNAP_DAYS[j + 1]}'].values
                                     - blocks[f'phi_d{SNAP_DAYS[j]}'].values)))
                 for j in range(len(SNAP_DAYS) - 1)])
hours = np.diff(np.array(SNAP_DAYS))*24.0
blocks['drift_snap_cph'] = -(dphi/hours[:, None]).mean(axis=0)/(2*2*np.pi)      # cph
# Phi against detuning and time (amplitude-weighted over the blocks of each bin)
hovdc = np.zeros((nt, centres.size), np.complex64)
hovdc_n = np.zeros(centres.size, int)
for i in range(centres.size):
    m = (blocks.dc.values >= DC_EDGES[i]) & (blocks.dc.values < DC_EDGES[i + 1])
    hovdc_n[i] = int(m.sum())
    if m.sum() >= MIN_BIN:
        hovdc[:, i] = Zb[:, bi[m]].mean(axis=1)

# the phase of the footprint through time and the coherence of each block in time
phi_foot = {}
for name, Zs_ in (('with', Zb), ('without', Zkb)):
    tot = Zs_[:, bi].sum(axis=1)
    phi_foot[name] = np.angle(-tot)
T_win_h = float(WAKE.sum())*dt_h
for name, Zs_ in (('with', Zb), ('without', Zkb)):
    num = np.abs(Zs_[WAKE][:, bi].sum(axis=0))
    den = np.abs(Zs_[WAKE][:, bi]).sum(axis=0)
    blocks[f'coh_{name}'] = np.where(den > 0, num/np.maximum(den, 1e-30), np.nan)
coh_dc = {}
for name in ('with', 'without'):
    out = np.full((centres.size, 4), np.nan)
    v = blocks[f'coh_{name}'].values
    for i in range(centres.size):
        m = (blocks.dc.values >= DC_EDGES[i]) & (blocks.dc.values < DC_EDGES[i + 1]) & np.isfinite(v)
        if m.sum() < MIN_BIN:
            continue
        draws = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        bs = np.median(v[m][draws], axis=1)
        out[i] = [np.median(v[m]), np.percentile(bs, 5), np.percentile(bs, 95), m.sum()]
    coh_dc[name] = out
x_pred = 2*(2*np.pi*centres)*T_win_h/2.0                                  # half the phase turned over the window
coh_pred = np.abs(np.sinc(x_pred/np.pi))                                   # |sin x / x|
coh = {f'median_{n}': float(np.nanmedian(blocks[f'coh_{n}'].values)) for n in ('with', 'without')}
for c in ('free', 'trapped'):
    m = blocks['class'].values == c
    coh[f'with_{c}'] = float(np.nanmedian(blocks.coh_with.values[m])) if m.sum() >= 3 else np.nan
    coh[f'without_{c}'] = float(np.nanmedian(blocks.coh_without.values[m])) if m.sum() >= 3 else np.nan

blocks.to_csv(OUTDIR + 'fig4_blocks.csv', index=False)                   # with the coherence and the drifts

by_class = {}
for c in ('free', 'trapped'):
    m = blocks['class'].values == c
    if m.sum() >= 3:
        zz = blocks.amp.values[m]*np.exp(1j*blocks.phi.values[m])
        draws = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        bs = np.cos(np.angle(zz[draws].sum(axis=1)))
        by_class[c] = (float(np.cos(np.angle(zz.sum()))), float(np.percentile(bs, 5)),
                       float(np.percentile(bs, 95)), int(m.sum()))

# the drift (for Figure 5) and the consistency with Figure 3
stats = {}
okd = np.isfinite(blocks.drift_cph.values) & np.isfinite(blocks.dc.values)
if okd.sum() >= 6:
    ts = theilslopes(blocks.drift_cph.values[okd], blocks.dc.values[okd])
    stats.update(drift_slope=float(ts[0]), drift_lo=float(ts[2]), drift_hi=float(ts[3]),
                 drift_rho=float(spearmanr(blocks.dc.values[okd], blocks.drift_cph.values[okd]).correlation))
    ts = theilslopes(blocks.drift_snap_cph.values[okd], blocks.dc.values[okd])
    stats.update(snap_slope=float(ts[0]), snap_lo=float(ts[2]), snap_hi=float(ts[3]),
                 snap_rho=float(spearmanr(blocks.dc.values[okd], blocks.drift_snap_cph.values[okd]).correlation))
    # agreement of the snapshots with the prediction, circular correlation of the bin means
    okp = np.isfinite(phase_dc[1:, :, 0]) & np.isfinite(pred_dc[1:])
    stats['pred_mean_abs_error'] = float(np.mean(np.abs(np.angle(np.exp(1j*(phase_dc[1:, :, 0] - pred_dc[1:]))))[okp]))
if blocks3 is not None and 'work_sig_it' in blocks:
    okc = np.isfinite(blocks.work_sig_it.values)
    stats['rho_work_Z_vs_fig3'] = float(spearmanr(blocks.work_from_Z.values[okc],
                                                  blocks.work_sig_it.values[okc]).correlation)
    stats['rho_gain_cosphi'] = float(spearmanr(blocks.cosphi.values[okc], blocks.gain.values[okc]).correlation)
ser_Z = -0.5*RHO0*((Z[:, good]*area_s[good]).sum(axis=1)/foot_area).real          # W m-2
if f3 is not None:
    s3 = smooth(f3.ser_sig_it.values)
    stats['corr_series_fig3'] = float(np.corrcoef(ser_Z[WORK], s3[WORK])[0, 1])
    stats['work_from_Z'] = float(ser_Z[WORK].sum()*dt_s)
    stats['work_fig3'] = float(f3.ser_sig_it.values[WORK].sum()*dt_s)
stats['Sr_over_Sl'] = float(np.sum(strain_amp['r'][good]*area_s[good])/np.sum(strain_amp['l'][good]*area_s[good]))

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
kb = np.arange(0, nt, 3)                                                   # every 3 h for the maps
ds = xr.Dataset(
    data_vars={'Zwin_re': (('y', 'x'), Z_win.real), 'Zwin_im': (('y', 'x'), Z_win.imag),
               'Zkwin_re': (('y', 'x'), Zk_win.real), 'Zkwin_im': (('y', 'x'), Zk_win.imag),
               **{f'Zoff{d}_re': (('y', 'x'), snaps[d].real) for d in SNAP_DAYS},
               **{f'Zoff{d}_im': (('y', 'x'), snaps[d].imag) for d in SNAP_DAYS},
               'Sr_amp': (('y', 'x'), strain_amp['r'].astype(np.float32)),
               'drift_cph': (('y', 'x'), drift_map), 'Ew_win': (('y', 'x'), Ew_map),
               'Sl_amp': (('y', 'x'), strain_amp['l'].astype(np.float32)),
               'hov_re': (('time', 'lat_bin'), hov.real), 'hov_im': (('time', 'lat_bin'), hov.imag),
               'hov_n': (('lat_bin',), hov_n),
               'Zb_re': (('time', 'block'), Zb.real), 'Zb_im': (('time', 'block'), Zb.imag),
               'Zkb_re': (('time', 'block'), Zkb.real), 'Zkb_im': (('time', 'block'), Zkb.imag),
               'Eb': (('time', 'block'), Eb.astype(np.float32)),
               'cosine': (('phi_bin', 'cstat'), cosine),
               'R_with': (('time',), R_with), 'R_without': (('time',), R_without),
               'phi_with': (('time',), phi_with), 'phi_without': (('time',), phi_without),
               'rayleigh': (('time',), rayleigh), 'n_with': (('time',), n_with),
               'rose_with': (('rose_bin',), rose['with']), 'rose_without': (('rose_bin',), rose['without']),
               'curve': (('dc_bin', 'cstat'), curve), 'ser_Z': (('time',), ser_Z),
               'phase_dc': (('snap', 'dc_bin', 'cstat'), phase_dc), 'pred_dc': (('snap', 'dc_bin'), pred_dc),
               'phi_foot_with': (('time',), phi_foot['with']), 'phi_foot_without': (('time',), phi_foot['without']),
               'coh_with': (('dc_bin', 'cstat'), coh_dc['with']), 'coh_without': (('dc_bin', 'cstat'), coh_dc['without']),
               'coh_pred': (('dc_bin',), coh_pred),
               'hovdc_re': (('time', 'dc_bin'), hovdc.real), 'hovdc_im': (('time', 'dc_bin'), hovdc.imag),
               'hovdc_n': (('dc_bin',), hovdc_n),
               'lon': (('y', 'x'), lon_s), 'lat': (('y', 'x'), lat_s), 'h': (('y', 'x'), h_s),
               'slope': (('y', 'x'), slope_s), 'foot': (('y', 'x'), foot), 'good': (('y', 'x'), good),
               'dc_FULL': (('y', 'x'), dc_full), 'area': (('y', 'x'), area_s),
               'mask': (('y', 'x'), f2.mask.values)},
    coords=dict(time=t, lat_bin=lat_mid, block=ids, phi_bin=phi_mid, dc_bin=centres, snap=SNAP_DAYS,
                rose_bin=0.5*(rose_edges[1:] + rose_edges[:-1]),
                cstat=['value', 'p05', 'p95', 'n']),
    attrs=dict(windows=str(WINDOWS), work_window=str((T_STORM0, T_WAKE1)), t_off=str(T_OFF)[:16],
               ni_band=str(NI_BAND), m2_band=str(M2_BAND), lp_h=LP_H, smooth_h=SMOOTH_H,
               amp_floor=AMP_FLOOR, amp_med=amp_med, foot_area_m2=foot_area, block_points=BLOCK,
               n_boot=N_BOOT, omega_M2=OMEGA_M2, rho0=RHO0, snap_days=str(SNAP_DAYS),
               cos_fit=str(cos_fit), rho_cos=rho_cos, rho_cos_level=rho_cos_level, lock=str(lock),
               foot_phase=str(foot_phase), by_class=str(by_class), stats=str(stats), coh=str(coh),
               T_win_h=T_win_h))
ds.to_netcdf(OUTDIR + 'fig4_phase.nc')

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
L = ['FIGURE 4 REPORT (full_v5, revision 4)  --  the phase of the wake against the tide',
     '=' * 80,
     f'slab caches, {nt} records, footprint {foot_area/1e6:.0f} km2, {nb} blocks of {BLOCK} points',
     'wake u_FULL - u_TIDE demodulated at omega/2, strain of u_TIDE demodulated at omega, cut-off '
     f'{LP_H:.0f} h', f'Phi = pi + arg Z with Z = sum_z S_r A*^2 dz, wake window {str(T_OFF)[:16]} to {T_WAKE1}',
     '', '1. THE PHASE OF THE FOOTPRINT (wake window)']
L.append(f'   effective phase {foot_phase["phi"]:+.2f} rad, cos {foot_phase["cos"]:+.2f}, share of the area '
         f'with positive cos(Phi) {foot_phase["share_pos"]:.2f}')
for c, v in by_class.items():
    L.append(f'   {c:8s} cos(Phi) {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}], {v[3]} blocks (amplitude-weighted)')
L.append(f'   anticlockwise over clockwise strain amplitude |S_r|/|S_l| over the footprint {stats["Sr_over_Sl"]:.2f}')
if 'corr_series_fig3' in stats:
    L.append(f'   consistency: strain work rebuilt from Z {stats["work_from_Z"]:.1f} J m-2 against Figure 3 '
             f'{stats["work_fig3"]:.1f}, correlation of the series {stats["corr_series_fig3"]:.2f}')
if 'rho_work_Z_vs_fig3' in stats:
    L.append(f'   blocks: work from Z against Figure 3 rho_s {stats["rho_work_Z_vs_fig3"]:.2f}, '
             f'gain against cos(Phi) rho_s {stats["rho_gain_cosphi"]:.2f}')
L.append('')
L.append('2. THE COSINE (tendency of the wake kinetic energy over (rho0/2)|Z|, wake window, dimensionless)')
L.append(f'   fit a cos(Phi) + b: a {cos_fit["a"]:.3f} (the theory gives 1.000 when the strain work alone '
         f'moves the energy), b {cos_fit["b"]:.3f} (the other terms of the budget over the largest strain work)')
L.append(f'   Spearman of the rate against cos(Phi) {rho_cos:.2f} against a time-shift level of {rho_cos_level:.2f}')
L.append(f'{"Phi":>7s} {"median":>9s} {"p05":>9s} {"p95":>9s} {"n":>6s}')
for j in range(PHI_BINS):
    if np.isfinite(cosine[j, 0]):
        L.append(f'{phi_mid[j]:7.2f} {cosine[j, 0]:9.3f} {cosine[j, 1]:9.3f} {cosine[j, 2]:9.3f} '
                 f'{int(cosine[j, 3]):6d}')
L.append('')
L.append('3. THE LOCK (resultant length of exp(i Phi) over the blocks)')
L.append(f'   wake with the tide: storm {lock["R_with_storm"]:.2f}, wake window {lock["R_with_wake"]:.2f}, '
         f'above the Rayleigh level ({lock["rayleigh_wake"]:.2f}) in {lock["hours_above"]} of {lock["hours_wake"]} hours, '
         f'mean phase {lock["mean_phi_wake"]:+.2f} rad')
L.append(f'   kinematic pairing: wake window {lock["R_without_wake"]:.2f}')
L.append(f'   phase of the footprint (amplitude-weighted) in the wake window: with the tide '
         f'{np.angle(np.nanmean(np.exp(1j*phi_foot["with"][WAKE]))):+.2f} rad, kinematic '
         f'{np.angle(np.nanmean(np.exp(1j*phi_foot["without"][WAKE]))):+.2f} rad')
L.append('   coherence of the phase of each block over the wake window (one if held fixed):')
L.append(f'   with the tide median {coh["median_with"]:.2f} (free {coh["with_free"]:.2f}, trapped {coh["with_trapped"]:.2f}), '
         f'kinematic {coh["median_without"]:.2f} (free {coh["without_free"]:.2f}, trapped {coh["without_trapped"]:.2f})')
L.append('   a phase drifting at twice the detuning would give ' + ', '.join(
    f'{c*1e3:+.1f}: {p:.2f}' for c, p in zip(centres, coh_pred) if np.isfinite(coh_dc['with'][list(centres).index(c), 0]))
         + ' (x 1e-3 cph)')
L.append('')
L.append('4. cos(Phi) AGAINST THE DETUNING OF THE FLOW (blocks, amplitude-weighted, 5 to 95%)')
for i, c in enumerate(centres):
    if np.isfinite(curve[i, 0]):
        L.append(f'   {c:+.4f} cph  {curve[i, 0]:+.2f} [{curve[i, 1]:+.2f}, {curve[i, 2]:+.2f}]  n {int(curve[i, 3])}')
L.append('')
L.append('5. THE DETUNING TURNS THE PHASE (Phi by bin of detuning at the snapshots, rad, with the prediction)')
L.append(f'{"dc cph":>9s}' + ''.join(f'{"d" + str(d):>9s}{"pred":>7s}' for d in SNAP_DAYS))
for i, c in enumerate(centres):
    if np.isfinite(phase_dc[:, i, 0]).any():
        L.append(f'{c:+9.4f}' + ''.join(f'{phase_dc[j, i, 0]:9.2f}{pred_dc[j, i]:7.2f}' for j in range(len(SNAP_DAYS))))
if 'pred_mean_abs_error' in stats:
    L.append(f'   mean absolute error of the prediction over the later snapshots {stats["pred_mean_abs_error"]:.2f} rad '
             f'(1.57 for no skill)')
L.append('')
L.append('6. THE DRIFT (for Figure 5, blocks, wake detuning against the detuning of the flow, cph)')
if 'drift_slope' in stats:
    L.append(f'   from the demodulated frequency: Theil and Sen slope {stats["drift_slope"]:.2f} '
             f'[{stats["drift_lo"]:.2f}, {stats["drift_hi"]:.2f}], Spearman {stats["drift_rho"]:.2f}')
    L.append(f'   from the phase at the snapshots: slope {stats["snap_slope"]:.2f} '
             f'[{stats["snap_lo"]:.2f}, {stats["snap_hi"]:.2f}], Spearman {stats["snap_rho"]:.2f}. The theory gives one')
L.append('')
L.append('READING THE RESULT')
L.append('   question 3 (first half) is answered yes if cos(Phi) changes sign between the free and the')
L.append('   trapped blocks as the works of Figure 3 do, the footprint phase of the wake with the tide')
L.append('   sits at the gaining side after the wind stops while the kinematic pairing does not, and the')
L.append('   phase of the blocks is held in time (coherence above what a drift at twice the detuning')
L.append('   allows), so that the detuning selects the phase and the exchange holds it.')
open(OUTDIR + 'fig4_phase_report.txt', 'w').write('\n'.join(L))
print('\n'.join(L))
print('saved', OUTDIR + 'fig4_phase.nc')