"""
figS9_plot.py  (full_v5/new_figures)  Supplementary Figure S9
--------------------------------------------------------------
Is the 3 km sampling of the 1 km model enough for the works of the paper
(revision 3 of figS9_compute.py).

  (a) the works of the internal tide on the wake accumulated over the work
      window on one footprint tile against the spacing of the sampling (lines)
      and with the fields averaged over 3 by 3 points before the sampling (open
      markers), the spacing of the paper marked, with the same work from the
      tide run of the paper at 3 km (star)
  (b) ratios to the value at 1 km against the spacing: the sum of the works
      with the tide run and with FULL alone, the root mean square strain of the
      internal tide and the wake kinetic energy
  (c) the strain work of the tile through time at 1 km at the spacing of the
      paper and at 6 km, with the strain work from the tide run of the paper

Reads ../figS9_out/figS9_sampling.nc written by ../figS9_compute.py.

  cd ~/dd_waves/paper_3/full_v5/new_figures
  python figS9_plot.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from v6_common import ps, IN, OUT, TERM, STAT_KW, ev, smooth, line_axes, shade_windows, date_axis

ds = xr.open_dataset(IN + 'figS9_out/figS9_sampling.nc')
tests, variants, terms, sources = (list(ds[c].values) for c in ('test', 'variant', 'term', 'source'))
PAPER = ds.attrs['paper_variant']
WINDOWS = ev(ds.attrs['windows'])
t = pd.to_datetime(ds.time.values)
dx = ds.spacing_km.values
cP, cF = tests.index('paper'), tests.index('full')
J = ds.J.values                                                              # (test, variant, term)
Ja = ds.anchor_J.values                                                      # (source, term)
lines = [i for i, v in enumerate(variants) if v.startswith('s')]             # the samplings with no averaging
i1, ip, ia = variants.index('s1'), variants.index(PAPER), variants.index('avg3')
jP, jS = terms.index('P'), terms.index('sig')
k_it = sources.index('it')
STYLE = {'P': TERM['it'], 'sig': TERM['sig_it'], 'del': TERM['del_it'], 'z': TERM['z_it'],
         'T': dict(colour='royalblue', ls=(0, (5, 1)), lw=1.5, label='tide side of the triad')}
MARK = {'sig': 'o', 'del': 's', 'z': '^', 'T': 'D', 'P': 'o'}

fig = plt.figure(figsize=(9.5, 8.0))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.80], wspace=0.30, hspace=0.84,
              left=0.090, right=0.975, top=0.955, bottom=0.115)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# (a) the accumulated works of the internal tide on the wake against the spacing
for q in ('P', 'sig', 'del', 'z', 'T'):
    j, s = terms.index(q), STYLE[q]
    ax_a.plot(dx[lines], J[cP, lines, j], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], marker=MARK[q],
              markersize=4.5 if q != 'P' else 6.5, zorder=4 if q != 'P' else 3)
    ax_a.plot([dx[ia]], [J[cP, ia, j]], linestyle='none', marker=MARK[q], markersize=8.5, markerfacecolor='none',
              markeredgecolor=s['colour'], markeredgewidth=1.4, zorder=5)
ax_a.plot([dx[ip]], [Ja[k_it, jP]], linestyle='none', marker='*', markersize=14, markerfacecolor='firebrick',
          markeredgecolor='black', markeredgewidth=0.6, zorder=6)
ax_a.axvline(dx[ip], color='dimgray', linestyle='dotted', linewidth=1.2, zorder=1)
ax_a.axhline(0.0, color='black', linewidth=0.8, zorder=1)
ax_a.set_xticks(dx[lines])
ax_a.set_xlim(0.5, dx[lines].max() + 0.5)
ax_a.set_xlabel('spacing of the sampling (km)')
ax_a.set_ylabel(r'work over the work window (J m$^{-2}$)')
line_axes(ax_a)
ps.panel(ax_a, 'a')
hand = [Line2D([0], [0], color=STYLE[q]['colour'], linestyle=STYLE[q]['ls'], linewidth=STYLE[q]['lw'], marker=MARK[q],
               markersize=4.5, label=STYLE[q]['label']) for q in ('P', 'sig', 'del', 'z', 'T')]
hand += [Line2D([0], [0], linestyle='none', marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=1.4, label='mean of 3 by 3 points'),
         Line2D([0], [0], color='dimgray', linestyle='dotted', linewidth=1.2, label='spacing of the paper'),
         Line2D([0], [0], linestyle='none', marker='*', markersize=11, markerfacecolor='firebrick',
                markeredgecolor='black', markeredgewidth=0.6, label='tide run of the paper (3 km)')]
ax_a.legend(handles=hand, loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=8.5, frameon=False,
            labelspacing=0.25, columnspacing=1.2, handlelength=2.6)
ax_a.set_title(f'{dx[ip]:.0f} km over 1 km  strain work {J[cP, ip, jS]/J[cP, i1, jS]:.2f}  sum of the works '
               f'{J[cP, ip, jP]/J[cP, i1, jP]:.2f}', **STAT_KW)

# (b) ratios to the value at 1 km
RAT = {'P1': dict(y=J[cP, :, jP], colour='black', ls='solid', marker='o', lw=2.2,
                  label='work of the internal tide on the wake'),
       'P2': dict(y=J[cF, :, jP], colour='darkgoldenrod', ls='dashdot', marker='^', lw=1.6,
                  label='the same with FULL alone'),
       'strain': dict(y=ds.strain_rms.values[cP], colour='teal', ls='dashed', marker='s', lw=1.6,
                      label='rms strain of the internal tide'),
       'ke': dict(y=ds.ke.values[cP], colour='dimgray', ls='dotted', marker='D', lw=1.6, label='wake kinetic energy')}
r_all = []
for name, s in RAT.items():
    r = s['y']/s['y'][i1]
    r_all += list(r[lines]) + [r[ia]]
    ax_b.plot(dx[lines], r[lines], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], marker=s['marker'],
              markersize=4.5, zorder=4)
    ax_b.plot([dx[ia]], [r[ia]], linestyle='none', marker=s['marker'], markersize=8.5, markerfacecolor='none',
              markeredgecolor=s['colour'], markeredgewidth=1.4, zorder=5)
ax_b.axvline(dx[ip], color='dimgray', linestyle='dotted', linewidth=1.2, zorder=1)
ax_b.axhline(1.0, color='black', linewidth=0.8, zorder=1)
ax_b.set_xticks(dx[lines])
ax_b.set_xlim(0.5, dx[lines].max() + 0.5)
ax_b.set_ylim(min(0.0, 1.1*float(np.nanmin(r_all))), 1.12*max(1.0, float(np.nanmax(r_all))))
ax_b.set_xlabel('spacing of the sampling (km)')
ax_b.set_ylabel('relative to 1 km')
line_axes(ax_b)
ps.panel(ax_b, 'b')
ax_b.legend(handles=[Line2D([0], [0], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], marker=s['marker'],
                            markersize=4.5, label=s['label']) for s in RAT.values()],
            loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=1, fontsize=8.5, frameon=False, labelspacing=0.25,
            handlelength=2.6)
ax_b.set_title(f'{dx[ip]:.0f} km over 1 km with FULL alone {J[cF, ip, jP]/J[cF, i1, jP]:.2f}', **STAT_KW)

# (c) the strain work of the tile through time
n_s = int(ds.attrs['smooth_h'])
SER = {'s1': dict(colour='black', ls='solid', lw=2.0, label='1 km'),
       PAPER: dict(colour='firebrick', ls='dashed', lw=1.6, label=f'{dx[ip]:.0f} km (the paper)'),
       'avg3': dict(colour='teal', ls='dashdot', lw=1.4, label='mean of 3 by 3 points'),
       's6': dict(colour='dimgray', ls='dotted', lw=1.5, label='6 km')}
for v, s in SER.items():
    if v not in variants:
        continue
    ax_c.plot(t, smooth(ds.series.values[cP, variants.index(v), jS], n_s)*1e3, color=s['colour'], linestyle=s['ls'],
              linewidth=s['lw'], zorder=4 if v == 's1' else 3)
ax_c.plot(t, smooth(ds.anchor_series.values[k_it, jS], n_s)*1e3, color='royalblue', linestyle=(0, (1, 1)),
          linewidth=1.3, zorder=2)
ax_c.axhline(0.0, color='black', linewidth=0.8, zorder=1)
t0, t1 = pd.Timestamp(ev(ds.attrs['work_window'])[0]) - pd.Timedelta(days=2), pd.Timestamp(ev(ds.attrs['work_window'])[1])
ax_c.set_xlim(t0, t1)
shade_windows(ax_c, WINDOWS, inside=True)
date_axis(ax_c)
ax_c.set_ylabel(r'tidal strain work (mW m$^{-2}$)')
line_axes(ax_c)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Line2D([0], [0], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], label=s['label'])
                     for v, s in SER.items() if v in variants]
            + [Line2D([0], [0], color='royalblue', linestyle=(0, (1, 1)), linewidth=1.3,
                      label='tide run of the paper (3 km)')],
            loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=9, frameon=False, columnspacing=1.2,
            handlelength=2.4)
ax_c.set_title(f'correlation of the {dx[ip]:.0f} km series with the 1 km series '
               f'{float(ds.corr_with_1km.values[cP, ip, jS]):.2f}', **STAT_KW)

fig.savefig(OUT + 'figS9_sampling.png', dpi=300)
print('saved', OUT + 'figS9_sampling.png')