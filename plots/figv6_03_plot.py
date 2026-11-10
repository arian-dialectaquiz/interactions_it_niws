"""
figv6_03_plot.py  (full_v5/new_figures)
---------------------------------------
Figure 3 of v6. The work of the tidal strain on the wake and the loss of the tide.

  (a) tidal strain work accumulated over the work window (25 km smoothing)
      with the gain of the wake contoured and the contour of exact resonance
  (b) the budget of the gain over the footprint through time (the large terms
      above and the three works of the tide below at their own scale)
  (c) cumulative tidal strain work on the wake with the tide and on the wake
      without it (the kinematic pairing) inside the range the same tide gives
      when shifted to every phase
  (d) the tide side through time (the work on the wake and the loss of the
      tide accumulated) with the standing deficit of the semidiurnal band and
      its null
  (e) the works by group (block means with 5 to 95% block-bootstrap
      intervals) with the bilinearity slope

Reads fig3_out/fig3_strain.nc fig6_out/fig6_budget.nc fig2_out/fig2_wake.nc
of v5 and the region maps.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import cartopy.crs as ccrs
from v6_common import (ps, IN, OUT, REGION, LAT_LIM, LAT_CRIT, WAKE, TERM, GAIN_FACE, STAT_KW, MAP_SIGMA_R,
                       MAP_SIGMA_S, MAP_ASPECT, ev, extend_for, nice, smooth, smooth2d, geo_axes, map_lines,
                       map_handles, map_legend, inset_colourbar, line_axes, shade_windows, date_axis, block_ids,
                       block_mean_band, cell_widths)

ds = xr.open_dataset(IN + 'fig3_out/fig3_strain.nc')
f6 = xr.open_dataset(IN + 'fig6_out/fig6_budget.nc')
f2 = xr.open_dataset(IN + 'fig2_out/fig2_wake.nc')
rg = xr.open_dataset(REGION)
stats = ev(ds.attrs['stats'])
tide_side = ev(f6.attrs['tide_side'])
WINDOWS = ev(ds.attrs['windows'])
T_W0, T_W1 = (pd.Timestamp(x) for x in ev(ds.attrs['work_window']))
T_OFF = pd.Timestamp(ds.attrs['t_off'])
SMOOTH_H = int(ds.attrs['smooth_h'])
HAVE_W = bool(int(ds.attrs['have_w']))
t = pd.to_datetime(ds.time.values)
rng = np.random.default_rng(7)
SECTOR_NAMES = [str(x) for x in f2.sector.values]
sector_code = f2.sector_code.values
foot = ds.foot.values.astype(bool)
slope_s, area_s, dc_s = ds.slope.values.astype(bool), ds.area.values, ds.dc_FULL.values
blk_s = block_ids(ds.lon.shape, int(ds.attrs['block_points']))

# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
W = 9.5
L1, R1, RAT1, WS1 = 0.070, 0.975, [1.0, 1.25], 0.30
L2, R2, RAT2, WS2 = 0.08, 0.975, [1.0, 1.0, 0.95], 0.46
H_TOP, H_GAP, H2, H_BOT = 0.32, 1.28, 1.9, 1.38
H1 = MAP_ASPECT*cell_widths(W, L1, R1, RAT1, WS1)[0]      # the map keeps the proportions of Figure 1
H = H_TOP + H1 + H_GAP + H2 + H_BOT
fig = plt.figure(figsize=(W, H))
gs1 = GridSpec(1, 2, figure=fig, width_ratios=RAT1, wspace=WS1, left=L1, right=R1, top=1 - H_TOP/H,
               bottom=1 - (H_TOP + H1)/H)
gs2 = GridSpec(1, 3, figure=fig, width_ratios=RAT2, wspace=WS2, left=L2, right=R2, top=(H_BOT + H2)/H,
               bottom=H_BOT/H)
ax_a = fig.add_subplot(gs1[0], projection=ccrs.PlateCarree())
gs1b = gs1[1].subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.08)
ax_b = fig.add_subplot(gs1b[0])
ax_b2 = fig.add_subplot(gs1b[1], sharex=ax_b)
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1])
ax_e = fig.add_subplot(gs2[2])

# --------------------------------------------------------------------------
# (a) the tidal strain work and the gain
# --------------------------------------------------------------------------
P_slope = smooth2d(ds.Pint_sig_it.values, MAP_SIGMA_S)/1e3                  # kJ m-2
vals = P_slope[foot & np.isfinite(P_slope)]
lim = nice(np.percentile(np.abs(vals), 97)) if vals.size else 1.0
norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)
levels = np.linspace(-lim, lim, 25)
cmap = mpl.colormaps[ps.CMAP_DIV]
lon_r, lat_r, h_r = rg.lon.values, rg.lat.values, rg.h.values
ocean_r = rg.mask.values == 1
P = smooth2d(np.where(ocean_r, rg.Pint_sig_it.values, np.nan), MAP_SIGMA_R)/1e3
ext_a = extend_for(P[ocean_r], -lim, lim)
ax_a.contourf(lon_r, lat_r, P, levels=levels, cmap=cmap, norm=norm, extend=ext_a, zorder=0,
              transform=ccrs.PlateCarree())
G = smooth2d(np.where(ocean_r, rg.Ew_tide_wake.values - rg.Ew_notide_wake.values, np.nan), MAP_SIGMA_R)/1e3
G_slope = smooth2d(ds.gain.values, MAP_SIGMA_S)/1e3
glev = nice(np.nanpercentile(np.abs(G_slope[foot]), 75)) if foot.any() else 1.0
ax_a.contour(lon_r, lat_r, G, levels=[glev, 3*glev], colors='black', linestyles='solid', linewidths=[0.7, 1.1],
             zorder=4, transform=ccrs.PlateCarree())
ax_a.contour(lon_r, lat_r, G, levels=[-glev], colors='black', linestyles='dashed', linewidths=0.7, zorder=4,
             transform=ccrs.PlateCarree())
map_lines(ax_a, lon_r, lat_r, ocean_r, h_r, dc=rg.dc_FULL_wake.values)
geo_axes(ax_a, 'a')
map_legend(ax_a, [Line2D([0], [0], color='black', linestyle='solid', linewidth=0.9,
                         label=f'gain {glev:g} and {3*glev:g} kJ m$^{{-2}}$'),
                  Line2D([0], [0], color='black', linestyle='dashed', linewidth=0.7, label=f'loss {glev:g}')]
           + map_handles())
inset_colourbar(ax_a, cmap, norm, r'tidal strain work (kJ m$^{-2}$)', extend=ext_a,
                ticks=[-lim, -lim/2, 0.0, lim/2, lim])

# --------------------------------------------------------------------------
# (b) the budget of the gain through time
# --------------------------------------------------------------------------
ax_b.set_xlim(t[0] + pd.Timedelta(hours=24), t[-1])
shade_windows(ax_b, WINDOWS)
for w, (a0, a1) in WINDOWS.items():
    ax_b2.axvspan(pd.Timestamp(a0), pd.Timestamp(a1), facecolor='none', edgecolor='silver',
                  hatch=ps.WINDOW_HATCH[{'pre': 'T1', 'storm': 'T2', 'wake': 'T3'}[w]], linewidth=0.0, zorder=0)
S = {k: smooth(ds[f'ser_{k}'].values, SMOOTH_H) for k in ('it', 'sig_it', 'del_it', 'z_it', 'sm', 'dPi')}
S['dEdt'] = ds.ser_dEdt.values
for q in ('dEdt', 'dPi', 'sm', 'it'):
    ax_b.plot(t, 1e3*S[q], color=TERM[q]['colour'], linestyle=TERM[q]['ls'], linewidth=TERM[q]['lw'], zorder=3)
low = ('sig_it', 'del_it', 'z_it') if HAVE_W else ('sig_it', 'del_it')
for q in low:
    ax_b2.plot(t, 1e3*S[q], color=TERM[q]['colour'], linestyle=TERM[q]['ls'], linewidth=TERM[q]['lw'], zorder=3)
for ax in (ax_b, ax_b2):
    ax.axhline(0.0, color='black', linewidth=0.7, zorder=2)
    line_axes(ax)
ax_b.set_ylabel('budget of the gain\n' + r'(mW m$^{-2}$)', fontsize=11)
ax_b2.set_ylabel('works of the tide\n' + r'(mW m$^{-2}$)', fontsize=11)
fig.align_ylabels([ax_b, ax_b2])
plt.setp(ax_b.get_xticklabels(), visible=False)
date_axis(ax_b2, 3)
ps.panel(ax_b, 'b', x=0.012)
fig.legend(handles=[Line2D([0], [0], color=TERM[q]['colour'], linestyle=TERM[q]['ls'], linewidth=TERM[q]['lw'],
                           label=TERM[q]['label']) for q in ('dEdt', 'dPi', 'sm', 'it') + low],
           loc='upper center', bbox_to_anchor=(0.5, 1 - (H_TOP + H1 + 0.30)/H), ncol=4, fontsize=9.5, frameon=False,
           columnspacing=1.0, handlelength=2.2, labelspacing=0.3)

# --------------------------------------------------------------------------
# (c) cumulative strain work within the range of every tidal phase
# --------------------------------------------------------------------------
tw = (t >= T_W0) & (t < T_W1)
dt_s = float((t[1] - t[0]).total_seconds())
i0 = int(np.argmin(np.abs(ds['shift'].values)))
ax_c.set_xlim(T_W0, T_W1 + pd.Timedelta(hours=18))
shade_windows(ax_c, WINDOWS, label=False)
BAND = {'with': dict(facecolor='silver', edgecolor='none', hatch=None, alpha=0.7),
        'without': dict(facecolor='none', edgecolor=WAKE['without']['colour'], hatch='////', alpha=0.6)}
hc = []
for k_, var in (('with', 'null_with_t'), ('without', 'null_without_t')):
    st = WAKE[k_]
    C = np.cumsum(np.where(tw[None, :], ds[var].values, 0.0), axis=1)*dt_s
    lo, hi = C[:, tw].min(axis=0), C[:, tw].max(axis=0)
    band = ax_c.fill_between(t[tw], lo, hi, linewidth=0.0, zorder=1, **BAND[k_])
    ax_c.plot(t[tw], C[i0, tw], color=st['colour'], linestyle=st['ls'], linewidth=1.8, zorder=4)
    hc.append((band, Line2D([0], [0], color=st['colour'], linestyle=st['ls'], linewidth=1.8)))
gain_end = ev(ds.attrs['budget'])['gain over the window']
ax_c.plot([T_W1], [gain_end], linestyle='none', marker='<', markersize=8, color=TERM['dEdt']['colour'],
          markeredgecolor='black', markeredgewidth=0.6, zorder=6, clip_on=False)
ax_c.axhline(0.0, color='black', linewidth=0.7, zorder=2)
ax_c.set_ylabel(r'cumulative tidal strain work (J m$^{-2}$)')
date_axis(ax_c, 4)
line_axes(ax_c)
ps.panel(ax_c, 'c')
null = ev(ds.attrs['null'])
ax_c.set_title(f'actual timing at rank {null["with_rank"]:.2f}' + '\n'
               + f'kinematic pairing at rank {null["without_rank"]:.2f}', **STAT_KW)
ax_c.legend(handles=hc + [Line2D([0], [0], linestyle='none', marker='<', markersize=7, color=TERM['dEdt']['colour'],
                                 markeredgecolor='black', markeredgewidth=0.6)],
            labels=[WAKE['with']['label'], 'kinematic pairing', 'gain over the window'],
            handler_map={tuple: HandlerTuple(ndivide=1)}, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1,
            frameon=False, columnspacing=1.6, handlelength=2.6)

# --------------------------------------------------------------------------
# (d) the tide side through time
# --------------------------------------------------------------------------
t6 = pd.to_datetime(f6.time.values)
W6 = (t6 >= T_W0) & (t6 < T_W1)
ax_d.set_xlim(T_W0, T_W1 + pd.Timedelta(hours=18))
shade_windows(ax_d, WINDOWS, label=False)
ax_d.plot(t6[W6], f6.P_cum.values[W6]/1e3, color='black', linestyle='-', linewidth=1.6, zorder=4)
ax_d.plot(t6[W6], f6.T_cum.values[W6]/1e3, color=ps.RUN_COLOUR['TIDE'], linestyle='-', linewidth=1.8, zorder=4)
ax_d.plot(t6, -f6.X_sm.values/1e3, color=ps.RUN_COLOUR['TIDE'], linestyle='--', linewidth=1.4, zorder=3)
ax_d.plot(t6, -f6.X_null_s.values/1e3, color='gray', linestyle=':', linewidth=1.4, zorder=3)
ax_d.axvline(T_OFF, color='black', linewidth=0.8, zorder=2)
ax_d.axhline(0.0, color='black', linewidth=0.7, zorder=1)
ax_d.set_ylabel(r'kJ m$^{-2}$')
date_axis(ax_d, 4)
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.set_title(f'loss over work {tide_side["minus_T_over_P"]:.2f}' + '\n'
               + f'deficit over loss rate {tide_side["tau_days"]:.1f} d', **STAT_KW)
ax_d.legend(handles=[Line2D([0], [0], color='black', linewidth=1.6, label='work on the wake'),
                     Line2D([0], [0], color=ps.RUN_COLOUR['TIDE'], linewidth=1.8, label='loss of the tide'),
                     Line2D([0], [0], color=ps.RUN_COLOUR['TIDE'], linestyle='--', linewidth=1.4,
                            label='deficit of the semidiurnal band'),
                     Line2D([0], [0], color='gray', linestyle=':', linewidth=1.4, label='null cross term')],
            loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, columnspacing=1.2,
            handlelength=2.6)

# --------------------------------------------------------------------------
# (e) the works by group
# --------------------------------------------------------------------------
work_it_map = ds.Pint_sig_it.values + ds.Pint_del_it.values + (ds.Pint_z_it.values if HAVE_W else 0.0)
sm_map = ds.Pint_sig_sm.values + ds.Pint_del_sm.values + (ds.Pint_z_sm.values if HAVE_W else 0.0)
GROUPS = [('below\nresonance', foot & (dc_s < 0)), ('above\nresonance', foot & (dc_s > 0)),
          ('central\nslope', slope_s & (sector_code == SECTOR_NAMES.index('Central'))),
          ('northern\nslope', slope_s & (sector_code == SECTOR_NAMES.index('North')))]
WORKS = [('internal tide', work_it_map, 'black'),
         ('storm-made motion', sm_map, TERM['sm']['colour']),
         ('kinematic pairing', ds.Pint_sig_kin.values, WAKE['without']['colour'])]
ypos = np.arange(len(GROUPS))[::-1].astype(float)
offs = np.linspace(0.26, -0.26, len(WORKS))
for (wname, wmap, colour), off in zip(WORKS, offs):
    for j, (gname, gmask) in enumerate(GROUPS):
        est, lo, hi = block_mean_band(wmap, gmask, area_s, blk_s, rng)
        if np.isfinite(lo):
            ax_e.barh(ypos[j] + off, (hi - lo)/1e3, left=lo/1e3, height=0.24, color=colour, alpha=0.22,
                      edgecolor='none', zorder=1)
        if np.isfinite(est):
            ax_e.barh(ypos[j] + off, est/1e3, height=0.24, color=colour, edgecolor='none', zorder=3)
ax_e.axvline(0.0, color='black', linewidth=0.7, zorder=2)
ax_e.set_yticks(ypos)
ax_e.set_yticklabels([g for g, _ in GROUPS], fontsize=10.5)
ax_e.set_xlabel(r'work over the window (kJ m$^{-2}$)')
xl = max(abs(v) for v in ax_e.get_xlim())
ax_e.set_xlim(-xl, xl)
line_axes(ax_e)
ax_e.grid(False, axis='y')
ps.panel(ax_e, 'e', x=0.955)
if 'bil_slope' in stats:
    ax_e.set_title(f'bilinearity slope {stats["bil_slope"]:.2f}' + '\n' + f'({stats["bil_lo"]:.2f} to {stats["bil_hi"]:.2f})',
                   **STAT_KW)
ax_e.legend(handles=[Patch(facecolor=colour, label=wname) for wname, _, colour in WORKS]
            + [Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='5 to 95% of blocks')],
            loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, columnspacing=1.4)

fig.savefig(OUT + 'figv6_3_strain.png', dpi=300)
print('saved', OUT + 'figv6_3_strain.png')