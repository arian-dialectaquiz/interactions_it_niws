"""
figv6_06_plot.py  (full_v5/new_figures)
---------------------------------------
Figure 6 of v6. The exchange predicted from the phase at the wind stop.

  (a) the strain work of the footprint after the wind stop measured and from
      the free drift of the phase (the detuning of each block and one
      frequency for the wake)
  (b) the work of each block after the wind stop predicted by the phase on
      the footprint with the measured sign of each block
  (c) the median efficiency by bin of the phase at the wind stop with the
      model curve of each class at its detuning and the curve with no
      detuning
  (d) the timing test folded onto one tidal period (the measured work had
      the tide been shifted against the storm and the model)
  (e) the median |efficiency| by quarter of the decay rate of the amplitude
      with the Lorentzian envelope at the median rate of each quarter

The agreement of the model phase with the measured phase through time stays
in the Supplementary Material. The numbers the panels no longer carry (the
correlations of (a) the rank correlation of (b) and (c) the amplitudes of (d)
and the share inside the envelope of (e)) go to the text. Reads
fig8_out/fig8_model.nc fig8_out/fig8_blocks.csv and fig3_out/fig3_strain.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from v6_common import (ps, IN, OUT, NATURAL_EARTH, LAT_CRIT, CLASS, STAT_KW, RAD2CPH, ev, smooth, line_axes,
                       date_axis, block_ids, extend_for)

ds = xr.open_dataset(IN + 'fig8_out/fig8_model.nc')
f3 = xr.open_dataset(IN + 'fig3_out/fig3_strain.nc')
blocks = pd.read_csv(IN + 'fig8_out/fig8_blocks.csv')
stats = ev(ds.attrs['stats'])
timing = ev(ds.attrs['timing'])
transform = ev(ds.attrs['transform'])
T_TIDE_H = float(ds.attrs['period_h'])
t = pd.to_datetime(ds.tafter.values)
SMOOTH_H = int(ds.attrs['smooth_h'])
MEAS = dict(colour='black', ls='-', lw=1.8, label='measured')
MODEL = dict(colour='firebrick', ls='--', lw=1.6, label='free drift (each block)')
ONE = dict(colour='darkgoldenrod', ls=(0, (1, 1)), lw=1.6, label='free drift (one frequency)')
LEG = dict(fontsize=9.5, frameon=False, labelspacing=0.3, handlelength=2.0, columnspacing=1.0)
rng_plot = np.random.default_rng(3)

# --------------------------------------------------------------------------
# layout in inches. Row 1 holds (a) and the map (b), row 2 holds (c) (d) and
# the narrow (e).
# --------------------------------------------------------------------------
W = 9.5
H_TOP, H1, H_GAP, H2, H_BOT = 0.42, 2.9, 1.70, 2.3, 1.55
H = H_TOP + H1 + H_GAP + H2 + H_BOT
fig = plt.figure(figsize=(W, H))
gs1 = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 0.85], wspace=0.30, left=0.11, right=0.975, top=1 - H_TOP/H,
               bottom=1 - (H_TOP + H1)/H)
gs2 = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.2, 0.72], wspace=0.48, left=0.105, right=0.975,
               top=(H_BOT + H2)/H, bottom=H_BOT/H)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1])
ax_e = fig.add_subplot(gs2[2])

# --------------------------------------------------------------------------
# (a) the strain work of the footprint after the wind stop
# --------------------------------------------------------------------------
fp = stats['footprint']
for var, st in (('foot_meas', MEAS), ('foot_drift', MODEL), ('foot_one', ONE)):
    ax_a.plot(t, smooth(ds[var].values, SMOOTH_H)*1e3, color=st['colour'], linestyle=st['ls'], linewidth=st['lw'],
              zorder=3)
ax_a.axhline(0.0, color='black', linewidth=0.7, zorder=1)
ax_a.set_ylabel('strain work of the\nfootprint ' + r'(mW m$^{-2}$)')
ax_a.set_xlim(t[0], t[-1])
date_axis(ax_a, 2)
line_axes(ax_a)
ps.panel(ax_a, 'a')
ax_a.set_title(f'measured {fp["W_meas"]:.0f} model {fp["W_drift"]:.0f} one frequency {fp["W_one"]:.0f} '
               r'J m$^{-2}$', **STAT_KW)
ax_a.legend(handles=[Line2D([0], [0], color=st['colour'], linestyle=st['ls'], linewidth=st['lw'], label=st['label'])
                     for st in (MEAS, MODEL, ONE)],
            loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, **LEG)

# --------------------------------------------------------------------------
# (b) the predicted work on the footprint with the measured sign of each block
# --------------------------------------------------------------------------
lon3, lat3, h3 = f3.lon.values, f3.lat.values, f3.h.values
foot3, mask3 = f3.foot.values.astype(bool), f3.mask.values == 1
blk3 = block_ids(lon3.shape, int(f3.attrs['block_points']))
pred_map = np.full(lon3.shape, np.nan)
for b, w in zip(blocks.block.values, blocks.W_drift.values):
    pred_map[foot3 & (blk3 == b)] = w/1e3
lim_b = max(round(float(np.nanpercentile(np.abs(np.r_[blocks.W_drift.values, blocks.W_meas.values]), 97)/1e3), 1), 0.1)
cmap_b = mpl.colormaps[ps.CMAP_DIV]
norm_b = mpl.colors.Normalize(vmin=-lim_b, vmax=lim_b)
lon_lim = (float(np.nanmin(lon3[foot3])) - 0.4, float(np.nanmax(lon3[foot3])) + 0.4)
lat_lim = (float(np.nanmin(lat3[foot3])) - 0.3, float(np.nanmax(lat3[foot3])) + 0.3)
ax_b.set_aspect('auto')
ax_b.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], crs=ccrs.PlateCarree())
if NATURAL_EARTH:
    ax_b.coastlines(linewidth=0.6)
    ax_b.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax_b.pcolormesh(lon3, lat3, pred_map, cmap=cmap_b, norm=norm_b, shading='auto', rasterized=True, zorder=2,
                transform=ccrs.PlateCarree())
ax_b.contour(lon3, lat3, np.where(mask3, h3, np.nan), levels=[200.0, 2000.0], colors='dimgray',
             linestyles=['dashed', 'solid'], linewidths=0.5, zorder=3, transform=ccrs.PlateCarree())
ax_b.plot(lon_lim, [LAT_CRIT, LAT_CRIT], color='black', linestyle='dashed', linewidth=1.0, zorder=4,
          transform=ccrs.PlateCarree())
gain_b = blocks.W_meas.values > 0
for fill, mm in ((True, gain_b), (False, ~gain_b)):
    ax_b.plot(blocks.lon.values[mm], blocks.lat.values[mm], linestyle='none', marker='o', markersize=4.0,
              markerfacecolor='black' if fill else 'white', markeredgecolor='white' if fill else 'black',
              markeredgewidth=0.8, zorder=5, transform=ccrs.PlateCarree())
gl = ax_b.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.6,
                    linestyle='--', zorder=6)
gl.top_labels = gl.right_labels = False
gl.xlines = gl.ylines = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {'size': 10, 'color': 'dimgray'}
ps.panel(ax_b, 'b')
sd = stats['drift']
ax_b.set_title(f'slope {sd["slope"]:.2f} ({sd["lo"]:.2f} to {sd["hi"]:.2f})' + '\n'
               + f'sign agreement {sd["sign_agreement"]:.2f}', **STAT_KW)
cax_b = ax_b.inset_axes([0.0, -0.36/H1 - 0.035, 1.0, 0.035])
cb_b = mpl.colorbar.ColorbarBase(cax_b, cmap=cmap_b, norm=norm_b, orientation='horizontal',
                                 extend=extend_for(pred_map, -lim_b, lim_b))
cb_b.set_label(r'work predicted by the phase (kJ m$^{-2}$)', fontsize=9.5, labelpad=2)
cax_b.tick_params(labelsize=9, length=2, pad=1)
cax_b.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f'{v:g}'))
cb_b.outline.set_linewidth(0.5)
ax_b.legend(handles=[Line2D([0], [0], linestyle='none', marker='o', markersize=4.5, markerfacecolor='black',
                            markeredgecolor='white', label='measured gain'),
                     Line2D([0], [0], linestyle='none', marker='o', markersize=4.5, markerfacecolor='white',
                            markeredgecolor='black', label='measured loss'),
                     Line2D([0], [0], color='black', linestyle='dashed', linewidth=1.0, label=r'$28.8^{\circ}$S'),
                     Line2D([0], [0], color='dimgray', linestyle='dashed', linewidth=0.6, label='200 m'),
                     Line2D([0], [0], color='dimgray', linestyle='solid', linewidth=0.6, label='2000 m')],
            loc='upper center', bbox_to_anchor=(0.5, -(0.36 + 0.10 + 0.40)/H1), ncol=3,
            **dict(LEG, handlelength=1.6))

# --------------------------------------------------------------------------
# (c) the efficiency by bin of the phase at the wind stop
# --------------------------------------------------------------------------
PG = ds.phi_grid.values
PHI_BIN = 30.0
pedges = np.arange(-180.0, 180.0 + 1e-6, PHI_BIN)
pmid = 0.5*(pedges[1:] + pedges[:-1])
for a, b, pm in zip(pedges[:-1], pedges[1:], pmid):
    m = (blocks.phi0_deg.values >= a) & (blocks.phi0_deg.values < b)
    v = blocks.eff_meas.values[m]
    if v.size == 0:
        continue
    med = np.median(v)
    if v.size >= 3:
        bs = np.median(v[rng_plot.integers(0, v.size, size=(1000, v.size))], axis=1)
        lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
        ax_c.bar(pm, hi - lo, bottom=lo, width=0.9*PHI_BIN, color='gray', alpha=0.25, edgecolor='none', zorder=1)
    ax_c.bar(pm, med, width=0.9*PHI_BIN, color='khaki' if med > 0 else 'none', edgecolor='dimgray',
             hatch=None if med > 0 else '////', linewidth=0.6, zorder=2)
    ax_c.text(pm, -1.0, f'{v.size}', ha='center', va='bottom', fontsize=8.5, color='dimgray', zorder=5)
ax_c.plot(PG, np.cos(np.radians(PG)), color='black', linestyle='dotted', linewidth=1.0, zorder=3)
for c in CLASS:
    ec = ds[f'eff_curve_{c}'].values
    if np.isfinite(ec).any():
        ax_c.plot(PG, ec, color=CLASS[c]['colour'], linestyle='-', linewidth=1.8, zorder=4)
ax_c.axhline(0.0, color='black', linewidth=0.7, zorder=1)
ax_c.axvline(0.0, color='black', linewidth=0.6, zorder=1)
ax_c.set_xlim(-180, 180)
ax_c.set_ylim(-1.05, 1.05)
ax_c.set_xticks([-180, -90, 0, 90, 180])
ax_c.set_xticklabels([r'$-180^{\circ}$', r'$-90^{\circ}$', '0', r'$90^{\circ}$', r'$180^{\circ}$'])
ax_c.set_xlabel(r'$\Phi_0$ at the wind stop')
ax_c.set_ylabel('efficiency\n(work over the largest work)', fontsize=11)
line_axes(ax_c)
ps.panel(ax_c, 'c')
psi = {c: -transform[c]['psi_deg'] for c in CLASS if np.isfinite(transform[c]['psi_deg'])}
best = [r'{:+.0f}$^{{\circ}}$ {}'.format(psi[c], CLASS[c]['label'].split()[0]) for c in CLASS if c in psi]
ax_c.set_title(r'largest gain at $\Phi_0$ = ' + (best[0] if best else '') + '\n'
               + ('and ' + ' and '.join(best[1:]) + ' resonance' if len(best) > 1 else 'resonance'), **STAT_KW)
ax_c.legend(handles=[Patch(facecolor='khaki', edgecolor='dimgray', label='gain'),
                     Patch(facecolor='none', edgecolor='dimgray', hatch='////', label='loss'),
                     Patch(facecolor='gray', alpha=0.25, edgecolor='none', label='5 to 95%')]
            + [Line2D([0], [0], color=CLASS[c]['colour'], linewidth=1.8, label=CLASS[c]['label']) for c in CLASS]
            + [Line2D([0], [0], color='black', linestyle='dotted', linewidth=1.0, label='no detuning')],
            loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, **LEG)

# --------------------------------------------------------------------------
# (d) the timing test folded onto one tidal period
# --------------------------------------------------------------------------
fh, nm = ds.fold_h.values, ds.null_meas.values/1e3
fb, fm = ds.fold_bin.values, ds.fold_mean.values/1e3
shm, nmod = ds.shift_model.values, ds.null_mod.values/1e3
ax_d.fill_between(shm, 0.0, nmod, where=nmod > 0, facecolor='khaki', edgecolor='none', interpolate=True, zorder=1)
ax_d.fill_between(shm, 0.0, nmod, where=nmod < 0, facecolor='none', edgecolor='dimgray', hatch='////', linewidth=0.0,
                  interpolate=True, zorder=1)
ax_d.plot(shm, nmod, color=MODEL['colour'], linestyle=MODEL['ls'], linewidth=1.6, zorder=3)
ax_d.plot(fh, nm, linestyle='none', marker='o', markersize=3.5, markerfacecolor='white', markeredgecolor='black',
          markeredgewidth=0.8, zorder=4)
okf = np.isfinite(fm)
ax_d.plot(fb[okf], fm[okf], color='black', linestyle='-', linewidth=1.8, zorder=5)
ax_d.axhline(0.0, color='black', linewidth=0.7, zorder=2)
ax_d.axvline(0.0, color='black', linewidth=1.0, zorder=2)
ax_d.set_xlim(-0.5*T_TIDE_H, 0.5*T_TIDE_H)
ax_d.set_xticks([-6, -3, 0, 3, 6])
ax_d.set_xlabel('shift of the tide\nagainst the storm (h)')
ax_d.set_ylabel('work after the wind\n' + r'stop (kJ m$^{-2}$)', fontsize=11)
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.set_title(f'correlation over the shifts {timing["corr"]:.2f}', **STAT_KW)
ax_d.legend(handles=[Line2D([0], [0], linestyle='none', marker='o', markersize=3.5, markerfacecolor='white',
                            markeredgecolor='black', label='measured'),
                     Line2D([0], [0], color='black', linewidth=1.8, label='folded mean'),
                     Line2D([0], [0], color=MODEL['colour'], linestyle=MODEL['ls'], linewidth=1.6, label='model'),
                     Line2D([0], [0], color='black', linewidth=1.0, label='actual timing'),
                     Patch(facecolor='khaki', edgecolor='none', label='gain'),
                     Patch(facecolor='none', edgecolor='dimgray', hatch='////', label='loss')],
            loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, **LEG)

# --------------------------------------------------------------------------
# (e) the efficiency by quarter of the decay rate (vertical bars at the
# median rate of each quarter)
# --------------------------------------------------------------------------
okr = np.isfinite(blocks.r_per_day.values)
r_all = blocks.r_per_day.values[okr]
e_all = np.abs(blocks.eff_meas.values[okr])
d_all = blocks.drift_cph.values[okr]/RAD2CPH
qe = np.quantile(r_all, [0, 0.25, 0.5, 0.75, 1.0])
xq = np.arange(4).astype(float)
labels_e = []
for j in range(4):
    m = (r_all >= qe[j]) & ((r_all <= qe[j + 1]) if j == 3 else (r_all < qe[j + 1]))
    v = e_all[m]
    labels_e.append(f'{np.median(r_all[m]):.2f}' if m.any() else '')
    if v.size == 0:
        continue
    med = np.median(v)
    if v.size >= 3:
        bs = np.median(v[rng_plot.integers(0, v.size, size=(1000, v.size))], axis=1)
        lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
        ax_e.bar(xq[j], hi - lo, bottom=lo, width=0.62, color='gray', alpha=0.25, edgecolor='none', zorder=1)
    ax_e.bar(xq[j], med, width=0.62, color='black', edgecolor='none', zorder=3)
    r_med, d_med = np.median(r_all[m])/86400.0, np.median(np.abs(d_all[m]))
    ax_e.plot([xq[j]], [r_med/np.sqrt(r_med**2 + d_med**2)], linestyle='none', marker='_', markersize=17,
              markeredgewidth=2.2, color='firebrick', zorder=4)
ax_e.set_xticks(xq)
ax_e.set_xticklabels(labels_e, fontsize=9.5)
ax_e.set_xlim(-0.6, 3.6)
ax_e.set_ylim(0.0, 1.0)
ax_e.set_xlabel('decay rate of each\nquarter ' + r'(d$^{-1}$)')
ax_e.set_ylabel(r'$|$efficiency$|$')
line_axes(ax_e)
ax_e.grid(False, axis='x')
ps.panel(ax_e, 'e')
ax_e.set_title(rf'$\rho_s$ = {stats["rho_abs_eff_rate"]:.2f} (level {stats["rho_abs_eff_rate_level"]:.2f})',
               **STAT_KW)
ax_e.legend(handles=[Patch(facecolor='black', label='median'),
                     Patch(facecolor='gray', alpha=0.25, edgecolor='none', label='5 to 95%'),
                     Line2D([0], [0], linestyle='none', marker='_', markersize=14, markeredgewidth=2.2,
                            color='firebrick', label=r'$r/\sqrt{r^2+\Delta_w^2}$')],
            loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=1, **dict(LEG, handlelength=1.4))

fig.savefig(OUT + 'figv6_6_model.png', dpi=300)
print('saved', OUT + 'figv6_6_model.png')