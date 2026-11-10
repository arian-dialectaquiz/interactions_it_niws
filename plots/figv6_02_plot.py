"""
figv6_02_plot.py  (full_v5/new_figures)
---------------------------------------
Figure 2 of v6. The two wakes and what the wind put into each.

  (a) near-inertial kinetic energy of the wake with the tide over the wake
      window and (b) of the wake without it on one logarithmic scale
  (c) with over without the tide by latitude along the slope with its
      5 to 95% block-bootstrap band
  (d) the footprint through time with the near-inertial wind work accumulated
      on each wake
  (e) the near-inertial wind work and the change of the wake kinetic energy
      over the work window by class of the flow for the two wakes (block means
      with 5 to 95% block-bootstrap intervals)

Reads fig2_out/fig2_wake.nc and fig6_out/fig6_budget.nc of v5 and the region
maps. Nothing is computed from the model.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from v6_common import (ps, IN, OUT, REGION, LAT_CRIT, LAT_LIM, WAKE, CLASS, TERM, GAIN_FACE, LOSS_HATCH, STAT_KW,
                       IN_LEG, MAP_ASPECT, ev, extend_for, nice, smooth, geo_axes, map_lines, map_handles,
                       map_legend, inset_colourbar, line_axes, shade_windows, date_axis, cell_widths)

ds = xr.open_dataset(IN + 'fig2_out/fig2_wake.nc')
f6 = xr.open_dataset(IN + 'fig6_out/fig6_budget.nc')
rg = xr.open_dataset(REGION)
BAND = ds.attrs['main_band']
WINDOWS = ev(ds.attrs['windows'])
T_OFF = pd.Timestamp(ds.attrs['t_off'])
SMOOTH_H = int(ds.attrs['smooth_h'])
budget = ev(f6.attrs['budget'])
t = pd.to_datetime(ds.time.values)
PAIR = {'tide': WAKE['with'], 'notide': WAKE['without']}

# --------------------------------------------------------------------------
# layout in inches. The maps keep the proportions of Figure 1 and the keys of
# (d) and (e) sit above their panels, in the gap below the maps.
# --------------------------------------------------------------------------
W = 9.5
L1, R1, RAT1, WS1 = 0.075, 0.940, [1.0, 1.0, 0.44], 0.16
L2, R2, RAT2, WS2 = 0.075, 0.975, [1.45, 1.1], 0.38
H_TOP, H_GAP, H2, H_BOT = 0.34, 1.62, 2.0, 0.62
H1 = MAP_ASPECT*cell_widths(W, L1, R1, RAT1, WS1)[0]
H = H_TOP + H1 + H_GAP + H2 + H_BOT
fig = plt.figure(figsize=(W, H))
gs1 = GridSpec(1, 3, figure=fig, width_ratios=RAT1, wspace=WS1, left=L1, right=R1, top=1 - H_TOP/H,
               bottom=1 - (H_TOP + H1)/H)
gs2 = GridSpec(1, 2, figure=fig, width_ratios=RAT2, wspace=WS2, left=L2, right=R2, top=(H_BOT + H2)/H,
               bottom=H_BOT/H)
ax_a = fig.add_subplot(gs1[0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs1[1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs1[2])
ax_d = fig.add_subplot(gs2[0])
ax_e = fig.add_subplot(gs2[1])

# --------------------------------------------------------------------------
# (a) and (b) the two wakes on one colour scale
# --------------------------------------------------------------------------
slope = ds.slope.values.astype(bool)
Ew = {p: ds[f'Ew{BAND}_{p}_wake'].values for p in PAIR}
both = np.concatenate([Ew[p][slope & (Ew[p] > 0)] for p in PAIR])     # the slope sets the colour range
vmin, vmax = nice(np.percentile(both, 8)), nice(np.percentile(both, 97))
norm = LogNorm(vmin=vmin, vmax=vmax)
levels = np.logspace(np.log10(vmin), np.log10(vmax), 25)
cmap = mpl.colormaps[ps.CMAP_ENERGY]
lon_r, lat_r, h_r = rg.lon.values, rg.lat.values, rg.h.values
ocean_r = rg.mask.values == 1
RG = {'tide': rg.Ew_tide_wake.values, 'notide': rg.Ew_notide_wake.values}
ext = extend_for(np.concatenate([RG[p][ocean_r & np.isfinite(RG[p]) & (RG[p] > 0)] for p in PAIR]), vmin, vmax)
for ax, p, run, letter, left in ((ax_a, 'tide', 'FULL', 'a', True), (ax_b, 'notide', 'WIND', 'b', False)):
    F = np.where(ocean_r & (RG[p] > 0), RG[p], np.nan)
    ax.contourf(lon_r, lat_r, F, levels=levels, cmap=cmap, norm=norm, extend=ext, zorder=0,
                transform=ccrs.PlateCarree())
    map_lines(ax, lon_r, lat_r, ocean_r, h_r, dc=rg[f'dc_{run}_wake'].values)
    geo_axes(ax, letter, left_labels=left)
    ax.set_title(PAIR[p]['label'], fontsize=11, fontweight='bold', loc='center', pad=5)
    map_legend(ax, map_handles())
inset_colourbar(ax_a, cmap, norm, r'wake kinetic energy (J m$^{-2}$)', extend=ext)

# --------------------------------------------------------------------------
# (c) with over without the tide by latitude (the key sits above the panel, level with the map titles)
# --------------------------------------------------------------------------
ylat, lr = ds.lat_bin.values, ds.lat_ratio.values
ok = np.isfinite(lr[:, 0])
ax_c.fill_betweenx(ylat[ok], lr[ok, 1], lr[ok, 2], facecolor='silver', edgecolor='none', alpha=0.6, zorder=1)
ax_c.fill_betweenx(ylat[ok], 1.0, lr[ok, 0], where=lr[ok, 0] >= 1.0, facecolor=GAIN_FACE, edgecolor='none',
                   interpolate=True, zorder=2)
ax_c.fill_betweenx(ylat[ok], 1.0, lr[ok, 0], where=lr[ok, 0] < 1.0, facecolor='none', edgecolor='dimgray',
                   hatch=LOSS_HATCH, linewidth=0.0, interpolate=True, zorder=2)
ax_c.plot(lr[ok, 0], ylat[ok], color='black', linestyle='solid', linewidth=1.7, zorder=4)
ax_c.axvline(1.0, color='black', linestyle='solid', linewidth=0.8, zorder=3)
ax_c.axhline(LAT_CRIT, color='black', linestyle='dashed', linewidth=0.9, zorder=3)
ax_c.set_xscale('log', base=2)
fin = lr[ok][:, 1:]
fin = fin[np.isfinite(fin) & (fin > 0)]
lim = max(0.6, min(2.0, 1.05*np.max(np.abs(np.log2(fin))))) if fin.size else 1.0
ax_c.set_xlim(2**-lim, 2**lim)
ticks = [v for v in (0.25, 0.5, 1.0, 2.0, 4.0) if 2**-lim <= v <= 2**lim]
ax_c.set_xticks(ticks)
ax_c.set_xticklabels([f'{v:g}' for v in ticks])
ax_c.minorticks_off()
ax_c.set_ylim(LAT_LIM)
ax_c.set_yticks([-31.5, -30.0, -28.5, -27.0, -25.5, -24.0, -22.5])
ax_c.yaxis.tick_right()
ax_c.yaxis.set_major_formatter(LATITUDE_FORMATTER)
ax_c.tick_params(axis='y', labelsize=10, labelcolor='dimgray')
ax_c.set_xlabel('wake kinetic energy\nwith over without the tide', fontsize=10.5)
ax_c.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Patch(facecolor='silver', edgecolor='none', alpha=0.6, label='5 to 95%'),
                     Patch(facecolor=GAIN_FACE, edgecolor='none', label='gain'),
                     Patch(facecolor='none', edgecolor='dimgray', hatch=LOSS_HATCH, label='loss')],
            loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=9, frameon=False, handlelength=1.3,
            handletextpad=0.5, columnspacing=0.8)

# --------------------------------------------------------------------------
# (d) the footprint through time with the wind work accumulated on each wake
# --------------------------------------------------------------------------
ax_d.set_xlim(t[0], t[-1])
shade_windows(ax_d, WINDOWS, inside=True)
ser = {p: smooth(ds[f'ser_Ew{BAND}_{p}'].sel(sector='South').values, SMOOTH_H) for p in PAIR}
cum = {p: ds[f'cumwork_{p}'].values for p in PAIR}
ax_d.fill_between(t, ser['notide'], ser['tide'], where=ser['tide'] >= ser['notide'], facecolor=GAIN_FACE,
                  edgecolor='none', interpolate=True, zorder=1)
ax_d.fill_between(t, ser['notide'], ser['tide'], where=ser['tide'] < ser['notide'], facecolor='none',
                  edgecolor='dimgray', hatch=LOSS_HATCH, linewidth=0.0, interpolate=True, zorder=1)
for p, st in PAIR.items():
    ax_d.plot(t, ser[p], color=st['colour'], linestyle=st['ls'], linewidth=1.8, zorder=4)
    grown = cum[p] > 0.05*np.nanmax(cum[p])
    ax_d.plot(t, np.where(grown, cum[p], np.nan), color=st['colour'], linestyle='dotted', linewidth=1.3, zorder=3)
ax_d.axvline(T_OFF, color='dimgray', linestyle='solid', linewidth=0.9, zorder=2)
top = max(np.nanmax(np.r_[ser['tide'], ser['notide']]), np.nanmax(np.r_[cum['tide'], cum['notide']]))
ax_d.set_yscale('log')
ax_d.set_ylim(10**(np.log10(top) - 2.3), 10**(np.log10(top) + 0.35))     # room above the peak for the window names
ax_d.set_ylabel(r'footprint mean (J m$^{-2}$)')
date_axis(ax_d, 4)
line_axes(ax_d)
ps.panel(ax_d, 'd', x=0.006, y=0.96)
ax_d.legend(handles=[Line2D([0], [0], color=PAIR['tide']['colour'], linestyle=PAIR['tide']['ls'], linewidth=1.8,
                            label='wake with the tide'),
                     Line2D([0], [0], color=PAIR['notide']['colour'], linestyle=PAIR['notide']['ls'], linewidth=1.8,
                            label='wake without the tide'),
                     Patch(facecolor=GAIN_FACE, edgecolor='none', label='gain with the tide'),
                     Line2D([0], [0], color=PAIR['tide']['colour'], linestyle='dotted', linewidth=1.3,
                            label='wind work with the tide'),
                     Line2D([0], [0], color=PAIR['notide']['colour'], linestyle='dotted', linewidth=1.3,
                            label='wind work without the tide'),
                     Line2D([0], [0], color='dimgray', linestyle='solid', linewidth=0.9, label='wind stops')],
            loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9.5, frameon=False, columnspacing=1.2,
            handlelength=2.2, labelspacing=0.3)

# --------------------------------------------------------------------------
# (e) wind work and change of the wake kinetic energy by class for the two wakes
# --------------------------------------------------------------------------
terms6 = [str(x) for x in f6.term.values]
cls6 = [str(x) for x in f6.cls.values]
ROWS = [('near-inertial\nwind work', 'wind work'), ('change of\nkinetic energy', 'tendency')]
ypos = np.arange(len(ROWS))[::-1].astype(float)
OFF = {('free', 'with'): 0.30, ('free', 'without'): 0.10, ('trapped', 'with'): -0.10, ('trapped', 'without'): -0.30}
for iw, w in enumerate(('with', 'without')):
    ct = f6.cls_terms.values[iw]                             # (cls, term, stat)
    for c in CLASS:
        ic = cls6.index(c)
        for j, (_, k) in enumerate(ROWS):
            it = terms6.index(k)
            est, lo, hi = ct[ic, it, 0]/1e3, ct[ic, it, 1]/1e3, ct[ic, it, 2]/1e3
            y = ypos[j] + OFF[(c, w)]
            if np.isfinite(lo):
                ax_e.barh(y, hi - lo, left=lo, height=0.18, color=CLASS[c]['colour'], alpha=0.22, edgecolor='none',
                          zorder=1)
            if np.isfinite(est):
                ax_e.barh(y, est, height=0.18, color=CLASS[c]['colour'] if w == 'with' else 'white',
                          edgecolor=CLASS[c]['colour'], hatch=None if w == 'with' else '////', linewidth=0.8,
                          zorder=3)
ax_e.axvline(0.0, color='black', linewidth=0.7, zorder=2)
ax_e.set_yticks(ypos)
ax_e.set_yticklabels([r for r, _ in ROWS], fontsize=10.5)
ax_e.set_xlabel(r'kJ m$^{-2}$ over the work window')
xl = max(abs(v) for v in ax_e.get_xlim())
ax_e.set_xlim(-xl, xl)
ax_e.set_ylim(-0.6, ypos.max() + 0.7)
line_axes(ax_e)
ax_e.grid(False, axis='y')
ps.panel(ax_e, 'e')
ax_e.set_title('footprint wind work with over without the tide '
               f'{budget["with"]["wind work"]/budget["without"]["wind work"]:.2f}', **STAT_KW)
ax_e.legend(handles=[Patch(facecolor=CLASS['free']['colour'], edgecolor=CLASS['free']['colour'],
                           label=CLASS['free']['label']),
                     Patch(facecolor=CLASS['trapped']['colour'], edgecolor=CLASS['trapped']['colour'],
                           label=CLASS['trapped']['label']),
                     Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='5 to 95% of blocks'),
                     Patch(facecolor='dimgray', edgecolor='dimgray', label='with the tide'),
                     Patch(facecolor='white', edgecolor='dimgray', hatch='////', label='without the tide')],
            loc='lower center', bbox_to_anchor=(0.45, 1.13), ncol=2, fontsize=9.5, frameon=False, columnspacing=1.2,
            handlelength=1.8, labelspacing=0.3)

fig.savefig(OUT + 'figv6_2_wakes.png', dpi=300)
print('saved', OUT + 'figv6_2_wakes.png')