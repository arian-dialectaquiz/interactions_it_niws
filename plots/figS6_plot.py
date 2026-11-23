"""
figS6_plot.py  (full_v5/new_figures)  Supplementary Figure S6
--------------------------------------------------------------
The detuning of the wake and of the flow over the bight (the maps of
Figure 5 of v5 that leave the main text).

  (a) detuning of the wake measured from the drift of its phase over the wake
      window (25 km smoothing) and (b) detuning of the flow under the wake on
      the same scale
  (c) the two detunings by latitude along the slope with the blue shift
      between them shaded

Reads fig5_out/fig5_resonance.nc of v5 and the region maps.
"""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cmocean
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from v6_common import (ps, IN, OUT, REGION, LAT_LIM, LAT_CRIT, MAP_SIGMA_R, MAP_ASPECT, IN_LEG, extend_for, smooth2d,
                       geo_axes, map_lines, map_handles, map_legend, inset_colourbar, cell_widths)

rg = xr.open_dataset(REGION)
DC_LIM = 3.0
cmap_dc, norm_dc = cmocean.cm.balance, mpl.colors.Normalize(vmin=-DC_LIM, vmax=DC_LIM)
lon, lat, h = rg.lon.values, rg.lat.values, rg.h.values
ocean = rg.mask.values == 1
slope = ocean & (h > 200.0) & (h < 2000.0)

W = 9.5
L, R, RAT, WS = 0.075, 0.950, [1.0, 1.0, 0.44], 0.16
H_TOP, H_BOT = 0.32, 0.72
H1 = MAP_ASPECT*cell_widths(W, L, R, RAT, WS)[0]          # the maps keep the proportions of Figure 1
H = H_TOP + H1 + H_BOT
fig = plt.figure(figsize=(W, H))
gs = GridSpec(1, 3, figure=fig, width_ratios=RAT, wspace=WS, left=L, right=R, top=1 - H_TOP/H, bottom=H_BOT/H)
ax_a = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[2])

# (a) and (b) the two detunings on one scale
fields = {'a': (smooth2d(np.where(ocean, rg.drift_cph.values, np.nan), MAP_SIGMA_R)*1e3, 'wake'),
          'b': (np.where(ocean, rg.dc_FULL_wake.values, np.nan)*1e3, 'flow')}
levels_dc = np.linspace(-DC_LIM, DC_LIM, 25)
ext_ab = extend_for(np.concatenate([fields[k][0][np.isfinite(fields[k][0])] for k in fields]), -DC_LIM, DC_LIM)
for ax, letter, left in ((ax_a, 'a', True), (ax_b, 'b', False)):
    F, name = fields[letter]
    ax.contourf(lon, lat, F, levels=levels_dc, cmap=cmap_dc, norm=norm_dc, extend=ext_ab, zorder=1,
                transform=ccrs.PlateCarree())
    map_lines(ax, lon, lat, ocean, h, dc=rg.dc_FULL_wake.values)
    geo_axes(ax, letter, left_labels=left)
    ax.set_title(f'detuning of the {name}', fontsize=10.5, fontweight='bold', loc='center', pad=5)
    map_legend(ax, map_handles())
inset_colourbar(ax_a, cmap_dc, norm_dc, r'detuning from $\omega/2$ ($10^{-3}$ cph)', extend=ext_ab)

# (c) the two detunings by latitude along the slope
LAT_BIN_W = 0.25
edges = np.arange(LAT_LIM[0], LAT_LIM[1] + 1e-6, LAT_BIN_W)
mid = 0.5*(edges[1:] + edges[:-1])
prof = {}
for name, F in (('wake', rg.drift_cph.values*1e3), ('flow', rg.dc_FULL_wake.values*1e3)):
    prof[name] = np.full(mid.size, np.nan)
    for i in range(mid.size):
        m = slope & (lat >= edges[i]) & (lat < edges[i + 1]) & np.isfinite(F)
        if m.sum() >= 3:
            prof[name][i] = np.nanmedian(F[m])
ax_c.fill_betweenx(mid, prof['wake'], prof['flow'], facecolor='silver', edgecolor='none', alpha=0.6, zorder=1)
ax_c.plot(prof['flow'], mid, color='dimgray', linestyle='dashed', linewidth=1.4, zorder=3)
ax_c.plot(prof['wake'], mid, color='black', linestyle='solid', linewidth=1.7, zorder=4)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_c.axhline(LAT_CRIT, color='black', linestyle='dashed', linewidth=0.9, zorder=2)
ax_c.set_xlim(-DC_LIM - 0.5, DC_LIM + 1.5)
ax_c.set_ylim(LAT_LIM)
ax_c.set_yticks([-31.5, -30.0, -28.5, -27.0, -25.5, -24.0, -22.5])
ax_c.yaxis.tick_right()
ax_c.yaxis.set_major_formatter(LATITUDE_FORMATTER)
ax_c.tick_params(axis='y', labelsize=10, labelcolor='dimgray')
ax_c.tick_params(axis='x', labelsize=10)
ax_c.set_xlabel(r'detuning on the slope' + '\n' + r'($10^{-3}$ cph)', fontsize=10)
ax_c.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Line2D([0], [0], color='black', linewidth=1.7, label='wake'),
                     Line2D([0], [0], color='dimgray', linestyle='dashed', linewidth=1.4, label='flow'),
                     Patch(facecolor='silver', alpha=0.6, edgecolor='none', label='blue shift')],
            loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=8.5, frameon=False, handlelength=1.3,
            handletextpad=0.5, columnspacing=0.8)

fig.savefig(OUT + 'figS6_detuning.png', dpi=300)
print('saved', OUT + 'figS6_detuning.png')