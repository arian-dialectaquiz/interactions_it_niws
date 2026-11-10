"""
figS4_plot.py  (full_v5/new_figures)  Supplementary Figure S4
--------------------------------------------------------------
The work of the internal tide along the slope and through the column and the
semidiurnal band through time (the panels of Figures 3 and 6 of v5 that leave
the main text).

  (a) the work of the internal tide and the gain of the wake along the slope
      by latitude (half-degree bins) with 5 to 95% block-bootstrap bands
  (b) the works by depth over the work window in the blocks below resonance
      and (c) above resonance on an axis linear within 1 uW m-3 and
      logarithmic beyond
  (d) the semidiurnal kinetic energy over the footprint through time in FULL
      in the internal tide alone and in the storm-made motion of each pair

Reads fig3_out/fig3_strain.nc fig6_out/fig6_budget.nc and fig2_out/fig2_wake.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from v6_common import (ps, IN, OUT, LAT_LIM, LAT_CRIT, CLASS, TERM, GAIN_FACE, STAT_KW, ev, nice, smooth, line_axes,
                       shade_windows, date_axis, block_ids, block_mean_band)

ds = xr.open_dataset(IN + 'fig3_out/fig3_strain.nc')
f6 = xr.open_dataset(IN + 'fig6_out/fig6_budget.nc')
f2 = xr.open_dataset(IN + 'fig2_out/fig2_wake.nc')
stats = ev(ds.attrs['stats'])
HAVE_W = bool(int(ds.attrs['have_w']))
rng = np.random.default_rng(7)
lat_s, area_s = ds.lat.values, ds.area.values
slope_s, foot = ds.slope.values.astype(bool), ds.foot.values.astype(bool)
blk_s = block_ids(lat_s.shape, int(ds.attrs['block_points']))
SYMLOG_THRESH = 1.0
LAT_BIN = 0.5
sec = list(f2.sector.values).index('South')
t6 = pd.to_datetime(f6.time.values)
WINDOWS = ev(f6.attrs['windows'])
T_OFF = pd.Timestamp(f6.attrs['t_off'])
T_W0, T_W1 = (pd.Timestamp(x) for x in ev(f6.attrs['work_window']))
closure = ev(f6.attrs['closure'])
SMOOTH_H = int(f6.attrs['smooth_h'])

fig = plt.figure(figsize=(9.5, 10.4))
gs1 = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.55, 0.55], wspace=0.26, left=0.085, right=0.975, top=0.935,
               bottom=0.640)
gs2 = GridSpec(1, 1, figure=fig, left=0.085, right=0.975, top=0.370, bottom=0.175)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1])
ax_c = fig.add_subplot(gs1[2], sharey=ax_b)
ax_d = fig.add_subplot(gs2[0])

# (a) the work of the internal tide and the gain along the slope
work_it_map = ds.Pint_sig_it.values + ds.Pint_del_it.values + (ds.Pint_z_it.values if HAVE_W else 0.0)
gain_map = ds.gain.values
edges = np.arange(np.floor(np.nanmin(lat_s[slope_s])), np.ceil(np.nanmax(lat_s[slope_s])) + 1e-6, LAT_BIN)
mid = 0.5*(edges[1:] + edges[:-1])
prof_w, prof_g = np.full((mid.size, 3), np.nan), np.full((mid.size, 3), np.nan)
for i in range(mid.size):
    m = slope_s & (lat_s >= edges[i]) & (lat_s < edges[i + 1])
    prof_w[i] = block_mean_band(work_it_map, m, area_s, blk_s, rng)
    prof_g[i] = block_mean_band(gain_map, m, area_s, blk_s, rng)
ax_a2 = ax_a.twiny()
foot_lat = (float(np.nanmin(lat_s[foot])), float(np.nanmax(lat_s[foot])))
ax_a.axhspan(foot_lat[0], foot_lat[1], facecolor='lightgray', alpha=0.35, edgecolor='none', zorder=0)
okw, okg = np.isfinite(prof_w[:, 0]), np.isfinite(prof_g[:, 0])
ax_a.fill_betweenx(mid[okw], prof_w[okw, 1]/1e3, prof_w[okw, 2]/1e3, facecolor='silver', edgecolor='none', alpha=0.45,
                   zorder=1)
ax_a.plot(prof_w[okw, 0]/1e3, mid[okw], color='black', linestyle='-', linewidth=2.0, zorder=4)
ax_a2.fill_betweenx(mid[okg], prof_g[okg, 1]/1e3, prof_g[okg, 2]/1e3, facecolor=GAIN_FACE, edgecolor='none', alpha=0.35,
                    zorder=1)
ax_a2.plot(prof_g[okg, 0]/1e3, mid[okg], color=TERM['dEdt']['colour'], linestyle=(0, (5, 2)), linewidth=2.0, zorder=3)
ax_a.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_a.axhline(LAT_CRIT, color='black', linestyle='dashed', linewidth=0.9, zorder=2)
xw = 1.1*np.nanmax(np.abs(prof_w[:, 1:]))/1e3
xg = 1.1*np.nanmax(np.abs(prof_g[:, 1:]))/1e3
ax_a.set_xlim(-xw, xw)
ax_a2.set_xlim(-xg, xg)
ax_a.set_ylim(LAT_LIM)
ax_a.set_yticks([-31.5, -30.0, -28.5, -27.0, -25.5, -24.0, -22.5])
ax_a.yaxis.set_major_formatter(LATITUDE_FORMATTER)
ax_a.tick_params(axis='y', labelsize=10.5, labelcolor='dimgray')
ax_a.set_xlabel(r'work of the internal tide over the window (kJ m$^{-2}$)')
ax_a2.set_xlabel(r'gain of the wake (kJ m$^{-2}$)', color=TERM['dEdt']['colour'])
ax_a2.tick_params(axis='x', colors=TERM['dEdt']['colour'], labelsize=10.5)
ax_a2.spines['top'].set_color(TERM['dEdt']['colour'])
ax_a.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
ax_a.spines[['right']].set_visible(False)
ax_a2.spines[['right']].set_visible(False)
ps.panel(ax_a, 'a')
if 'rho' in stats:
    ax_a.text(0.98, 0.03, rf'blocks $\rho_s$ = {stats["rho"]:.2f} (level {stats["rho_level"]:.2f})' + '\n'
              + rf'partial $\rho_s$ = {stats["rho_partial"]:.2f}', transform=ax_a.transAxes, ha='right', va='bottom',
              fontsize=10, color='dimgray', linespacing=1.3)
ax_a.legend(handles=[Line2D([0], [0], color='black', linewidth=2.0, label='work of the internal tide'),
                     Line2D([0], [0], color=TERM['dEdt']['colour'], linestyle=(0, (5, 2)), linewidth=2.0,
                            label='gain of the wake'),
                     Patch(facecolor='gray', alpha=0.4, edgecolor='none', label='5 to 95% of blocks'),
                     Patch(facecolor='lightgray', alpha=0.35, edgecolor='none', label='footprint')],
            loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, columnspacing=1.4)

# (b) and (c) the works by depth
zb = ds.zbin.values
prof = ds.prof.values*1e6
terms = list(ds.term.values)
qs = ['sig_it', 'del_it', 'z_it', 'sm'] if HAVE_W else ['sig_it', 'del_it', 'sm']
for ax, c, letter in ((ax_b, 'free', 'b'), (ax_c, 'trapped', 'c')):
    j = list(ds.cls.values).index(c)
    for q in qs:
        y = prof[terms.index(q), j]
        ok = np.isfinite(y)
        ax.plot(y[ok], zb[ok], color=TERM[q]['colour'], linestyle=TERM[q]['ls'], linewidth=TERM[q]['lw'])
    ax.axvline(0.0, color='black', linewidth=0.7)
    ax.set_ylim(800, 0)
    ax.set_xscale('symlog', linthresh=SYMLOG_THRESH, linscale=0.8)
    ax.set_xlabel(r'$\mu$W m$^{-3}$' + '\n' + f'[log beyond {SYMLOG_THRESH:g}]', fontsize=10.5)
    ax.set_title(CLASS[c]['label'], fontsize=11, color=CLASS[c]['colour'], loc='center', pad=5)
    line_axes(ax)
    ps.panel(ax, letter)
ax_b.set_ylabel('depth (m)')
plt.setp(ax_c.get_yticklabels(), visible=False)
xl = nice(max(np.nanmax(np.abs(prof[terms.index(q), j])) for q in qs if q in terms for j in range(prof.shape[1])))*1.5
ticks = [v for v in (-100, -10, -1, 0, 1, 10, 100) if abs(v) <= xl]
for a in (ax_b, ax_c):
    a.set_xlim(-xl, xl)
    a.set_xticks(ticks)
    a.set_xticklabels([f'{v:g}' for v in ticks])
    a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax_b.legend(handles=[Line2D([0], [0], color=TERM[q]['colour'], linestyle=TERM[q]['ls'], linewidth=TERM[q]['lw'],
                            label=TERM[q]['label']) for q in qs],
            loc='upper center', bbox_to_anchor=(1.15, -0.26), ncol=1, frameon=False, columnspacing=1.2)

# (d) the semidiurnal band through time
ax_d.set_xlim(T_W0 - pd.Timedelta(hours=12), T_W1)
ax_d.axvline(T_OFF, color='black', linewidth=0.8, zorder=2)
date_axis(ax_d, 2)
shade_windows(ax_d, WINDOWS, label=False)
ax_d.plot(t6, smooth(f2.ser_M2_FULL.values[:, sec], SMOOTH_H), color=ps.RUN_COLOUR['FULL'], linestyle=ps.RUN_STYLE['FULL'],
          linewidth=1.8, zorder=5)
ax_d.plot(t6, smooth(f6.ser_M2_tide.values, SMOOTH_H), color=ps.RUN_COLOUR['TIDE'], linestyle=ps.RUN_STYLE['TIDE'],
          linewidth=1.6, zorder=4)
ax_d.plot(t6, smooth(f6.ser_M2_sm.values, SMOOTH_H), color='deeppink', linestyle=(0, (6, 2, 1, 2)), linewidth=1.6,
          zorder=3)
ax_d.plot(t6, smooth(f6.ser_M2_ot.values, SMOOTH_H), color=ps.RUN_COLOUR['WIND'], linestyle=ps.RUN_STYLE['WIND'],
          linewidth=1.4, zorder=3)
ax_d.set_ylabel(r'semidiurnal kinetic energy (J m$^{-2}$)')
ax_d.set_ylim(bottom=0.0)
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.set_title(f'wake window FULL over TIDE {closure["FULL"]/closure["TIDE"]:.2f}' + '\n'
               + f'cross term {closure["cross"]:.0f} J m$^{{-2}}$', **STAT_KW)
ax_d.legend(handles=[Line2D([0], [0], color=ps.RUN_COLOUR['FULL'], linestyle=ps.RUN_STYLE['FULL'], linewidth=1.8,
                            label='FULL'),
                     Line2D([0], [0], color=ps.RUN_COLOUR['TIDE'], linestyle=ps.RUN_STYLE['TIDE'], linewidth=1.6,
                            label='internal tide (TIDE)'),
                     Line2D([0], [0], color='deeppink', linestyle=(0, (6, 2, 1, 2)), linewidth=1.6,
                            label='storm-made (FULL minus TIDE)'),
                     Line2D([0], [0], color=ps.RUN_COLOUR['WIND'], linestyle=ps.RUN_STYLE['WIND'], linewidth=1.4,
                            label='storm-made (WIND minus BACK)')],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, columnspacing=1.6, handlelength=2.6)

fig.savefig(OUT + 'figS4_works_band.png', dpi=300)
print('saved', OUT + 'figS4_works_band.png')
