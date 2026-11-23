"""
figS8_plot.py  (full_v5/new_figures)  Supplementary Figure S8
--------------------------------------------------------------
The strain of the balanced flow and the second-order correction it makes to
the effective inertial frequency.

  (a) strain of the balanced flow over |f| on the slope band (FULL in the wake
      window, the flow that defines the classes) with the contour of exact
      resonance
  (b) area-weighted cumulative distributions over the footprint of the
      vorticity the strain and the divergence of that flow, each over |f|
  (c) the correction to the effective frequency by bin of the first-order
      detuning of the flow, median with its 5 to 95% block bootstrap, and the
      estimate -|sigma|^2 / (8|f|)

Reads ../figS8_out/figS8_strain.nc written by ../figS8_compute.py.

  cd ~/dd_waves/paper_3/full_v5/new_figures
  python figS8_plot.py
"""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from v6_common import (ps, IN, OUT, STAT_KW, extend_for, nice, geo_axes, map_lines, map_handles, map_legend,
                       inset_colourbar, line_axes, hatch_above)

ds = xr.open_dataset(IN + 'figS8_out/figS8_strain.nc')
RUN, WIN = ds.attrs['main_run'], ds.attrs['main_window']
lon, lat, h = ds.lon.values, ds.lat.values, ds.h.values
ocean = ds.mask.values == 1
slope, foot, area = ds.slope.values.astype(bool), ds.foot.values.astype(bool), ds.area.values
sof = ds[f'sof_{RUN}_{WIN}'].values
zof = ds[f'zof_{RUN}_{WIN}'].values
dof = ds[f'dof_{RUN}_{WIN}'].values
dc1 = ds[f'dc1_{RUN}_{WIN}'].values
sof_raw = ds['sof_raw'].values

fig = plt.figure(figsize=(9.5, 6.3))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.92], height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.42,
              left=0.070, right=0.975, top=0.955, bottom=0.095)
ax_a = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# (a) strain of the balanced flow over |f| on the slope band
F = np.where(slope, sof, np.nan)
vmax = nice(np.nanpercentile(F, 98))
cmap_s, norm_s = plt.get_cmap('YlGnBu'), mpl.colors.Normalize(vmin=0.0, vmax=vmax)
ext_a = extend_for(F, 0.0, vmax)
ax_a.pcolormesh(lon, lat, F, cmap=cmap_s, norm=norm_s, shading='auto', zorder=1, transform=ccrs.PlateCarree())
map_lines(ax_a, lon, lat, ocean, h, dc=np.where(slope, dc1, np.nan))
geo_axes(ax_a, 'a')
ax_a.text(0.975, 0.845, 'strain of the balanced flow', transform=ax_a.transAxes, ha='right', va='top', fontsize=11,
          fontweight='bold', zorder=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2.5))
map_legend(ax_a, map_handles())
inset_colourbar(ax_a, cmap_s, norm_s, r'strain over $|f|$', extend=ext_a)


# (b) cumulative distributions over the footprint
def cdf(x, wgt):
    g = np.isfinite(x) & np.isfinite(wgt)
    o = np.argsort(x[g])
    return x[g][o], np.cumsum(wgt[g][o])/np.sum(wgt[g])


LINES = {'vorticity': dict(x=np.abs(zof), colour='black', ls='solid', lw=1.9, label=r'vorticity $|\zeta|/|f|$'),
         'strain': dict(x=sof, colour='teal', ls='dashed', lw=1.9, label=r'strain $|\sigma|/|f|$'),
         'divergence': dict(x=np.abs(dof), colour='dimgray', ls='dotted', lw=1.7, label=r'divergence $|\delta|/|f|$'),
         'strain raw': dict(x=sof_raw, colour='darkgoldenrod', ls='dashdot', lw=1.2,
                            label='strain without the Gaussian')}
x_hi = 0.0
for name, s in LINES.items():
    xs, cs = cdf(s['x'][foot], area[foot])
    ax_b.plot(xs, cs, color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], zorder=3)
    x_hi = max(x_hi, float(np.interp(0.98, cs, xs)))
ax_b.set_xlim(0.0, nice(x_hi))
ax_b.set_ylim(0.0, 1.0)
ax_b.set_xlabel(r'gradient of the balanced flow over $|f|$')
ax_b.set_ylabel('share of the footprint area')
line_axes(ax_b)
ps.panel(ax_b, 'b')
ax_b.legend(handles=[Line2D([0], [0], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], label=s['label'])
                     for s in LINES.values()],
            loc='lower right', fontsize=9, frameon=True, framealpha=0.92, edgecolor='none', borderpad=0.3,
            labelspacing=0.25, handlelength=2.4)
ax_b.set_title(f'footprint median  strain {ds.attrs["median_sof_foot"]:.2f}  vorticity '
               f'{ds.attrs["median_zof_foot"]:.2f}', **STAT_KW)

# (c) the correction by bin of the first-order detuning
xb = ds.dc_bin.values*1e3
c = ds.curve_corr.values*1e3
ok = np.isfinite(c[:, 0])
band = ok & np.isfinite(c[:, 1])
ax_c.fill_between(xb[band], c[band, 1], c[band, 2], facecolor='silver', edgecolor='none', alpha=0.7, zorder=2)
ax_c.plot(xb[ok], c[ok, 0], color='black', linestyle='solid', linewidth=1.9, marker='o', markersize=3.5, zorder=4)
ax_c.plot(xb[ok], ds.curve_corr_est.values[ok]*1e3, color=ps.REFERENCE, linestyle='dashed', linewidth=1.4, zorder=5)
ax_c.axhline(0.0, color='black', linewidth=0.8, zorder=1)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=1)
pad = 0.5*(xb[1] - xb[0])
ax_c.set_xlim(xb[ok].min() - pad, xb[ok].max() + pad)
lo = float(np.nanmin(np.where(band[:, None], c, c[:, :1])))
ax_c.set_ylim(2.1*lo if lo < 0 else -0.01, -0.12*lo if lo < 0 else 0.01)       # room for the key under the band
h_above = hatch_above(ax_c, ax_c.get_xlim()[1])
ax_c.set_xlabel(r'first-order detuning of the flow ($10^{-3}$ cph)')
ax_c.set_ylabel(r'correction ($10^{-3}$ cph)')
line_axes(ax_c)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Line2D([0], [0], color='black', linewidth=1.9, marker='o', markersize=3.5,
                            label='exact minus first-order frequency'),
                     Patch(facecolor='silver', alpha=0.7, edgecolor='none', label='5 to 95% block bootstrap'),
                     Line2D([0], [0], color=ps.REFERENCE, linestyle='dashed', linewidth=1.4,
                            label=r'$-|\sigma|^2/(8|f|)$'), h_above],
            loc='lower left', fontsize=9, frameon=True, framealpha=0.92, edgecolor='none', borderpad=0.3,
            labelspacing=0.25, handlelength=2.4)
ax_c.set_title(f'class changes on {100*ds.attrs["share_flip_foot"]:.1f}% of the footprint area  '
               f'{ds.attrs["n_flip_blocks"]} of {ds.attrs["n_blocks"]} blocks', **STAT_KW)

fig.savefig(OUT + 'figS8_strain.png', dpi=300)
print('saved', OUT + 'figS8_strain.png')