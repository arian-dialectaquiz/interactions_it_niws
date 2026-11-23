"""
figS10_plot.py  (full_v5/new_figures)
-------------------------------------
The barotropic tide and the internal tide in the work on the wake.

  (a) work on the wake accumulated over the footprint through time for the
      tide (Figure 3) the internal tide and the barotropic tide
  (b) works over the work window by term for the two parts (bars)
  (c) strain work of each part by depth below and above resonance
  (d) strain work after the wind subsides for the tide shifted in time, each
      part, with the actual tide marked and its place among the shifts

Reads ../figS10_out/figS10_tide_parts.nc written by ../figS10_compute.py.

  cd ~/dd_waves/paper_3/full_v5/new_figures
  python figS10_plot.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from v6_common import ps, IN, OUT, CLASS, STAT_KW, ev, line_axes, shade_windows, date_axis

ds = xr.open_dataset(IN + 'figS10_out/figS10_tide_parts.nc')
WINDOWS = ev(ds.attrs['windows'])
T_OFF = pd.Timestamp(ds.attrs['t_off'])
t = pd.to_datetime(ds.time.values)
dt_s = float((t[1] - t[0]).total_seconds())
tab = {tuple(k.split('|')): v for k, v in ev(ds.attrs['table']).items()}        # (term part window) -> J m-2
place = {tuple(k.split('|')): v for k, v in ev(ds.attrs['place']).items()}      # (part window) -> actual place half range
work = (t >= pd.Timestamp(ev(ds.attrs['work_window'])[0])) & (t < pd.Timestamp(ev(ds.attrs['work_window'])[1]))

PART = {'all': dict(colour='black', ls='solid', lw=2.3, label='tide (Figure 3)'),
        'bc': dict(colour='teal', ls='dashed', lw=2.0, label='internal tide (deviation from the depth mean)'),
        'bt': dict(colour='darkgoldenrod', ls='dotted', lw=2.0, label='barotropic tide (depth mean)')}
TERMS = ['sig', 'del', 'z', 'P', 'T']
TERM_LABEL = {'sig': 'strain\nwork', 'del': 'divergence\nwork', 'z': 'vertical\nshear work', 'P': 'work on\nthe wake',
              'T': 'tide side'}

fig = plt.figure(figsize=(9.5, 8.6))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62,
              left=0.085, right=0.975, top=0.930, bottom=0.135)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# (a) the work on the wake accumulated through the work window
for p in ('bt', 'bc', 'all'):
    s = PART[p]
    y = np.cumsum(np.where(work, ds[f'ser_P_{p}'].values, 0.0))*dt_s
    ax_a.plot(t[work], y[work], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], zorder=4 if p == 'all' else 3)
ax_a.axhline(0.0, color='black', linewidth=0.8, zorder=1)
ax_a.axvline(T_OFF, color='dimgray', linewidth=1.0, zorder=2)
ax_a.set_xlim(t[work][0], t[work][-1])
ax_a.set_ylabel(r'work on the wake (J m$^{-2}$)')
date_axis(ax_a, interval=3)
line_axes(ax_a)
shade_windows(ax_a, WINDOWS)
ps.panel(ax_a, 'a')
ax_a.set_title(f'share of the internal tide {ds.attrs["share_bc_P"]:.2f}', **{**STAT_KW, 'pad': 20})

# (b) the works over the work window by term
x = np.arange(len(TERMS))
wid = 0.38
for i, p in enumerate(('bt', 'bc')):
    s = PART[p]
    vals = [tab[(q, p, 'work')] for q in TERMS]
    ax_b.bar(x + (i - 0.5)*wid, vals, width=wid, facecolor=s['colour'] if p == 'bc' else 'none', edgecolor=s['colour'],
             hatch='' if p == 'bc' else '////', linewidth=1.2, zorder=3)
for j, q in enumerate(TERMS):
    ax_b.plot([j - wid, j + wid], [tab[(q, 'all', 'work')]]*2, color='black', linewidth=2.0, zorder=5)
ax_b.axhline(0.0, color='black', linewidth=0.8, zorder=1)
ax_b.set_xticks(x)
ax_b.set_xticklabels([TERM_LABEL[q] for q in TERMS], fontsize=9.5)
ax_b.set_ylabel(r'over the work window (J m$^{-2}$)')
line_axes(ax_b)
ps.panel(ax_b, 'b')
ax_b.set_title(f'share of the internal tide\nstrain work {ds.attrs["share_bc_sig"]:.2f}  '
               f'tide side {ds.attrs["share_bc_T"]:.2f}', **STAT_KW)

# (c) the strain work of each part by depth and class
zb = ds.zbin.values
keep = zb <= 700.0
for c, cs in CLASS.items():
    for p in ('bt', 'bc'):
        ax_c.plot(ds[f'prof_sig_{p}_{c}'].values[keep]*1e6, zb[keep], color=cs['colour'], linestyle=PART[p]['ls'],
                  linewidth=1.9, marker=cs['marker'], markersize=3.5, zorder=3)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=1)
ax_c.set_ylim(700.0, 0.0)
ax_c.set_xlabel(r'strain work (10$^{-6}$ W m$^{-3}$)')
ax_c.set_ylabel('depth (m)')
line_axes(ax_c)
ps.panel(ax_c, 'c')

# (d) the shifted tides after the wind subsides
sh = ds['shift'].values
for p in ('bt', 'bc', 'all'):
    s = PART[p]
    y = ds[f'shifted_{p}_after'].values
    ax_d.plot(sh, y, color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], zorder=4 if p == 'all' else 3)
    ax_d.plot(0.0, y[sh == 0][0], marker='o', markersize=6, markerfacecolor='white', markeredgecolor=s['colour'],
              markeredgewidth=1.6, linestyle='none', zorder=6)
ax_d.axhline(0.0, color='black', linewidth=0.8, zorder=1)
ax_d.axvline(0.0, color='dimgray', linewidth=1.0, zorder=2)
ax_d.set_xlim(sh.min(), sh.max())
ax_d.set_xticks(np.arange(-48, 49, 24))
ax_d.set_xlabel('shift of the tide against the wake (h)')
ax_d.set_ylabel('strain work after the\n' + r'wind subsides (J m$^{-2}$)')
line_axes(ax_d)
ps.panel(ax_d, 'd')
n = int(ds.attrs['n_shifts'])
ax_d.set_title(f'place of the actual tide among {n} shifts\ninternal {place[("bc", "after")][1]}  '
               f'barotropic {place[("bt", "after")][1]}  tide {place[("all", "after")][1]}', **STAT_KW)

# keys under the panels
fig.legend(handles=[Line2D([0], [0], color=s['colour'], linestyle=s['ls'], linewidth=s['lw'], label=s['label'])
                    for s in (PART['all'], PART['bc'], PART['bt'])]
           + [Line2D([0], [0], color='dimgray', linewidth=1.0, label='the wind subsides (a) and no shift (d)')],
           loc='lower center', bbox_to_anchor=(0.5, 0.512), ncol=2, fontsize=9.5, frameon=False, handlelength=2.6,
           columnspacing=1.6)
fig.legend(handles=[Patch(facecolor=PART['bc']['colour'], edgecolor=PART['bc']['colour'], label='internal tide (b)'),
                    Patch(facecolor='none', edgecolor=PART['bt']['colour'], hatch='////', label='barotropic tide (b)'),
                    Line2D([0], [0], color='black', linewidth=2.0, label='tide (b)'),
                    Line2D([0], [0], color=CLASS['free']['colour'], marker=CLASS['free']['marker'], markersize=4,
                           linewidth=1.9, label='below resonance (c)'),
                    Line2D([0], [0], color=CLASS['trapped']['colour'], marker=CLASS['trapped']['marker'], markersize=4,
                           linewidth=1.9, label='above resonance (c)'),
                    Line2D([0], [0], marker='o', markersize=6, markerfacecolor='white', markeredgecolor='black',
                           markeredgewidth=1.6, linestyle='none', label='actual tide (d)')],
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5, frameon=False, handlelength=2.6,
           columnspacing=1.6)

fig.savefig(OUT + 'figS10_tide_parts.png', dpi=300)
print('saved', OUT + 'figS10_tide_parts.png')