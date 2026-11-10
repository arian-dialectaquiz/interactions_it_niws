"""
figv6_04_plot.py  (full_v5/new_figures)
---------------------------------------
Figure 4 of v6. The phase of the wake against the tide and the frequency of the wake.

  (a) phase efficiency cos(Phi) of the wake window over the bight (25 km
      smoothing) with the contour of exact resonance
  (b) cos(Phi) of the wake window by bin of the detuning of the flow under
      the wake with the 5 to 95% range of the blocks
  (c) the wake frequency minus each candidate (exact resonance the planetary
      frequency the effective frequency and the wake median) and the flow and
      wake detunings by class as medians with the 5 to 95% of the blocks and
      the locking half-width
  (d) gain of the wake and tidal strain work by bin of the flow detuning
  (e) the controls of the strain work and of the gain ranked by rank
      correlation over the blocks with the shift level of each test

Reads fig4_out/fig4_phase.nc fig5_out/fig5_resonance.nc fig5_out/fig5_blocks.csv
of v5 and the region maps.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from v6_common import (ps, IN, OUT, REGION, CLASS, GAIN_FACE, STAT_KW, MAP_SIGMA_R, MAP_ASPECT, black_on_white, ev,
                       smooth2d, geo_axes, map_lines, map_handles, map_legend, inset_colourbar, line_axes,
                       hatch_above, cell_widths)

f4 = xr.open_dataset(IN + 'fig4_out/fig4_phase.nc')
f5 = xr.open_dataset(IN + 'fig5_out/fig5_resonance.nc')
blocks = pd.read_csv(IN + 'fig5_out/fig5_blocks.csv')
rg = xr.open_dataset(REGION)
AMP_FLOOR = float(f4.attrs['amp_floor'])
kunze = ev(f5.attrs['kunze'])
COS_TRIM = (0.07, 0.96)
cmap_cos = mpl.colors.ListedColormap(mpl.colormaps['gnuplot2'](np.linspace(COS_TRIM[0], COS_TRIM[1], 256)),
                                     name='gnuplot2_trim')
norm_cos = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
COS_LEVELS = np.linspace(-1.0, 1.0, 41)

# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
W = 9.5
L1, R1, RAT1, WS1 = 0.070, 0.975, [1.0, 1.1], 0.30
H_TOP, H_GAP1, H2, H_GAP2, H3, H_BOT = 0.08, 0.78, 2.0, 1.12, 2.7, 1.28
H1 = MAP_ASPECT*cell_widths(W, L1, R1, RAT1, WS1)[0]      # the map keeps the proportions of Figure 1
H = H_TOP + H1 + H_GAP1 + H2 + H_GAP2 + H3 + H_BOT
y1 = 1 - (H_TOP + H1)/H
y2 = y1 - (H_GAP1 + H2)/H
fig = plt.figure(figsize=(W, H))
gs1 = GridSpec(1, 2, figure=fig, width_ratios=RAT1, wspace=WS1, left=L1, right=R1, top=1 - H_TOP/H, bottom=y1)
gs2 = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.38, left=0.140, right=0.975,
               top=y1 - H_GAP1/H, bottom=y2)
gs3 = GridSpec(1, 1, figure=fig, left=0.215, right=0.975, top=(H_BOT + H3)/H, bottom=H_BOT/H)
ax_a = fig.add_subplot(gs1[0], projection=ccrs.PlateCarree())
gs1b = gs1[1].subgridspec(3, 1, height_ratios=[0.50, H1 - 1.55, 1.05], hspace=0.0)   # a shorter panel beside the map
ax_b = fig.add_subplot(gs1b[1])
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1])
ax_e = fig.add_subplot(gs3[0])

# --------------------------------------------------------------------------
# (a) the phase efficiency over the wake window
# --------------------------------------------------------------------------
lon_r, lat_r, h_r = rg.lon.values, rg.lat.values, rg.h.values
ocean_r = rg.mask.values == 1
slope_r = ocean_r & (h_r > 200.0) & (h_r < 2000.0)
Zre, Zim = rg.Zwin_re.values, rg.Zwin_im.values
has = ocean_r & np.isfinite(Zre) & ((Zre != 0) | (Zim != 0))
Zs = smooth2d(np.where(has, Zre, np.nan), MAP_SIGMA_R) + 1j*smooth2d(np.where(has, Zim, np.nan), MAP_SIGMA_R)
phi_s, amp_s = np.angle(-Zs), np.abs(Zs)
floor_a = AMP_FLOOR*np.nanmedian(amp_s[slope_r & np.isfinite(amp_s)])
shown = has & np.isfinite(amp_s)
ax_a.pcolormesh(lon_r, lat_r, np.where(shown, 0.0, np.nan), cmap=mpl.colors.ListedColormap(['gainsboro']),
                shading='auto', zorder=0, rasterized=True, transform=ccrs.PlateCarree())
ax_a.contourf(lon_r, lat_r, np.where(shown & (amp_s > floor_a), np.cos(phi_s), np.nan), levels=COS_LEVELS,
              cmap=cmap_cos, norm=norm_cos, zorder=1, transform=ccrs.PlateCarree())
map_lines(ax_a, lon_r, lat_r, ocean_r, h_r, dc=rg.dc_FULL_wake.values, crit=False)
ax_a.plot([-52.0, -39.2], [-28.8, -28.8], color='black', linestyle='dashed', linewidth=1.0,
          path_effects=black_on_white, zorder=6, transform=ccrs.PlateCarree())
geo_axes(ax_a, 'a')
hand = map_handles()
hand[1] = Line2D([0], [0], color='black', linestyle='dashed', linewidth=1.0, path_effects=black_on_white,
                 label=r'$28.8^{\circ}$S')
map_legend(ax_a, hand[:2] + [Patch(facecolor='gainsboro', edgecolor='none', label='below the amplitude floor')]
           + hand[2:])
inset_colourbar(ax_a, cmap_cos, norm_cos, r'phase efficiency $\cos\Phi$',
                ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])

# --------------------------------------------------------------------------
# (b) cos(Phi) by bin of the detuning of the flow
# --------------------------------------------------------------------------
x = f4.dc_bin.values*1e3
cv = f4.curve.values
ok = np.isfinite(cv[:, 0])
x_lo, x_hi = (x[ok].min() - 0.6, x[ok].max() + 0.6) if ok.any() else (x.min(), x.max())
h_above = hatch_above(ax_b, x_hi)
ax_b.fill_between(x[ok], cv[ok, 1], cv[ok, 2], facecolor='silver', edgecolor='none', alpha=0.6, zorder=1)
ax_b.plot(x[ok], cv[ok, 0], color='black', linestyle='solid', linewidth=1.8, marker='o', markersize=4.5,
          markerfacecolor='white', zorder=4)
ax_b.axhline(0.0, color='black', linewidth=0.7, zorder=2)
ax_b.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_b.set_xlim(x_lo, x_hi)
ax_b.set_ylim(-1.05, 1.05)
ax_b.set_xlabel(r'flow detuning under the wake ($10^{-3}$ cph)')
ax_b.set_ylabel(r'$\cos\Phi$ of the wake window')
line_axes(ax_b)
ps.panel(ax_b, 'b')
by_class = ev(f4.attrs['by_class'])
cls_txt = '\n'.join(r'$\cos\Phi$ {} {:.2f}'.format(CLASS[c]['label'], np.atleast_1d(by_class[c])[0])
                     for c in CLASS if c in by_class)
ax_b.set_title(cls_txt, **STAT_KW)
ax_b.legend(handles=[Line2D([0], [0], color='black', linewidth=1.8, marker='o', markersize=4.5,
                            markerfacecolor='white', label='blocks by bin'),
                     Patch(facecolor='silver', edgecolor='none', alpha=0.6, label='5 to 95% of blocks'), h_above],
            loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, columnspacing=1.6)

# --------------------------------------------------------------------------
# (c) the frequency of the wake against its candidates and the detunings by class
# --------------------------------------------------------------------------


def spread_bar(ax, y, values, colour, height=0.62):
    v = values[np.isfinite(values)]
    if v.size < 3:
        return np.nan, np.nan
    lo, hi, med = np.percentile(v, 5), np.percentile(v, 95), np.median(v)
    ax.barh(y, hi - lo, left=lo, height=height, color=colour, alpha=0.18, edgecolor='none', zorder=1)
    ax.barh(y, med, height=height, color=colour, edgecolor='none', zorder=3)
    return lo, hi


drift = blocks.drift_cph.values*1e3
cls_v = blocks['class'].values
ROWS_C = [(r'$\omega_w - \omega/2$', drift, 'black'),
          (r'$\omega_w - |f|$', blocks.wake_minus_f.values*1e3, 'black'),
          (r'$\omega_w - |f| - \zeta/2$', blocks.blue_shift.values*1e3, 'black'),
          (r'$\omega_w - \bar{\omega}_w$', drift - np.median(drift), 'black'),
          None,
          (r'$\Delta_c$', np.where(cls_v == 'free', blocks.dc.values*1e3, np.nan), CLASS['free']['colour']),
          (r'$\Delta_c$', np.where(cls_v == 'trapped', blocks.dc.values*1e3, np.nan), CLASS['trapped']['colour']),
          (r'$\Delta_w$', np.where(cls_v == 'free', drift, np.nan), CLASS['free']['colour']),
          (r'$\Delta_w$', np.where(cls_v == 'trapped', drift, np.nan), CLASS['trapped']['colour'])]
ypos, labels_c = [], []
y = 0.0
for row in ROWS_C[::-1]:
    if row is None:
        y += 0.7
        continue
    ypos.append(y)
    labels_c.append(row[0])
    y += 1.0
ypos = np.array(ypos)
half_S = float(np.nanmedian(blocks.half_S.values))*1e3
ax_c.axvspan(-half_S, half_S, facecolor='silver', alpha=0.35, edgecolor='none', zorder=0)
lo_all, hi_all = [], []
for yy, (name, v, colour) in zip(ypos, [r for r in ROWS_C if r is not None][::-1]):
    lo, hi = spread_bar(ax_c, yy, v, colour)
    lo_all.append(lo), hi_all.append(hi)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_c.set_yticks(ypos)
ax_c.set_yticklabels(labels_c, fontsize=11.5)
ax_c.set_xlabel(r'$10^{-3}$ cph')
pad = 0.08*(np.nanmax(hi_all) - np.nanmin(lo_all))
ax_c.set_xlim(np.nanmin(lo_all) - pad, np.nanmax(hi_all) + pad)
ax_c.set_ylim(-0.6, ypos.max() + 0.6)
line_axes(ax_c)
ax_c.grid(False, axis='y')
ps.panel(ax_c, 'c', x=0.955)
if kunze:
    ax_c.set_title(f'Kunze slope {kunze["slope"]:.2f} ({kunze["lo"]:.2f} to {kunze["hi"]:.2f})' + '\n'
                   + f'blue shift {100*kunze["blue_shift_rel"]:.1f}% of f', **STAT_KW)
ax_c.legend(handles=[Patch(facecolor='black', label='median'),
                     Patch(facecolor='black', alpha=0.18, edgecolor='none', label='5 to 95%'),
                     Patch(facecolor=CLASS['free']['colour'], label='below resonance'),
                     Patch(facecolor=CLASS['trapped']['colour'], label='above resonance'),
                     Patch(facecolor='silver', alpha=0.35, edgecolor='none', label=r'$|S_r|/2$')],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, columnspacing=1.0,
            handlelength=1.4)

# --------------------------------------------------------------------------
# (d) gain and strain work by flow detuning
# --------------------------------------------------------------------------
x5 = f5.dc_bin.values*1e3
gb, wb = f5.gain_bins.values, f5.work_bins.values
okg, okw = np.isfinite(gb[:, 0]), np.isfinite(wb[:, 0])
x_lo, x_hi = x5[okg | okw].min() - 0.6, x5[okg | okw].max() + 0.6
h_above_d = hatch_above(ax_d, x_hi)
for okk, arr, colour, face in ((okg, gb, 'darkgoldenrod', GAIN_FACE), (okw, wb, 'black', 'silver')):
    ax_d.fill_between(x5[okk], arr[okk, 1]/1e3, arr[okk, 2]/1e3, facecolor=face, edgecolor='none', alpha=0.6,
                      zorder=1)
    ax_d.plot(x5[okk], arr[okk, 0]/1e3, color=colour, linestyle='-', linewidth=1.8, marker='o', markersize=4,
              markerfacecolor='white', zorder=4)
ax_d.axhline(0.0, color='black', linewidth=0.7, zorder=2)
ax_d.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_d.set_xlim(x_lo, x_hi)
ax_d.set_xlabel(r'flow detuning under the wake ($10^{-3}$ cph)')
ax_d.set_ylabel(r'kJ m$^{-2}$ over the wake window')
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.legend(handles=[Line2D([0], [0], color='darkgoldenrod', linewidth=1.8, marker='o', markersize=4,
                            markerfacecolor='white', label='gain of the wake'),
                     Line2D([0], [0], color='black', linewidth=1.8, marker='o', markersize=4,
                            markerfacecolor='white', label='tidal strain work'),
                     Patch(facecolor='gray', edgecolor='none', alpha=0.4, label='5 to 95% of blocks'), h_above_d],
            loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9.5, frameon=False, columnspacing=1.0,
            handlelength=1.8)

# --------------------------------------------------------------------------
# (e) the controls ranked
# --------------------------------------------------------------------------
ctl = f5.controls.values
cands = [str(c) for c in f5.candidate.values]
order = np.argsort(-np.abs(np.nan_to_num(ctl[0, :, 0])))
ypos = np.arange(len(cands))[::-1].astype(float)
tick_halo = [pe.Stroke(linewidth=3.2, foreground='white'), pe.Normal()]
for i, (name, colour, off) in enumerate((('strain work', 'black', 0.18), ('gain', 'darkgoldenrod', -0.18))):
    lev = ctl[i, order, 2]
    ax_e.barh(ypos + off, 2*lev, left=-lev, height=0.36, color='silver', alpha=0.45, edgecolor='none', zorder=1)
    ax_e.barh(ypos + off, ctl[i, order, 0], height=0.36, color=colour, edgecolor='none', zorder=3)
    okp = np.isfinite(ctl[i, order, 1])
    ax_e.plot(ctl[i, order, 1][okp], (ypos + off)[okp], linestyle='none', marker='|', markersize=9,
              markeredgewidth=1.4, color='black', path_effects=tick_halo, zorder=4)
ax_e.axvline(0.0, color='black', linewidth=0.7, zorder=2)
ax_e.set_yticks(ypos)
ax_e.set_yticklabels([cands[j] for j in order], fontsize=10.5)
x_lo = min(np.nanmin(ctl[:, :, 0]), -np.nanmax(ctl[:, :, 2]), np.nanmin(ctl[:, :, 1])) - 0.05
x_hi = max(np.nanmax(ctl[:, :, 0]), np.nanmax(ctl[:, :, 2]), np.nanmax(ctl[:, :, 1])) + 0.05
ax_e.set_xlim(x_lo, x_hi)
ax_e.set_xticks([v for v in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0) if x_lo <= v <= x_hi])
ax_e.set_xlabel(r'rank correlation $\rho_s$ over the blocks')
line_axes(ax_e)
ax_e.grid(False, axis='y')
ps.panel(ax_e, 'e', x=0.955)
ax_e.legend(handles=[Patch(facecolor='black', label='with the tidal strain work'),
                     Patch(facecolor='darkgoldenrod', label='with the gain of the wake'),
                     Line2D([0], [0], linestyle='none', marker='|', markersize=9, markeredgewidth=1.4, color='black',
                            path_effects=tick_halo, label='partial on the wake kinetic energy'),
                     Patch(facecolor='silver', alpha=0.5, edgecolor='none', label='shift level of each test')],
            loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, columnspacing=1.4)

fig.savefig(OUT + 'figv6_4_phase.png', dpi=300)
print('saved', OUT + 'figv6_4_phase.png')