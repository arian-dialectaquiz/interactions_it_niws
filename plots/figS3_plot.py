"""
figS3_plot.py  (full_v5/new_figures)  Supplementary Figure S3
--------------------------------------------------------------
The two wakes compared against the detuning of the flow and in time and the
kinetic energy budget of each (the panels of Figures 2 and 6 of v5 that leave
the main text).

  (a) wake kinetic energy in the wake window by bin of the detuning of the
      flow under each wake with 5 to 95% block-bootstrap bands
  (b) with over without the tide through time by group (footprint and its
      two classes the central slope and the northern slope) with 5 to 95%
      bands
  (c) budget of the wake with the tide over the footprint accumulated from
      the start of the storm and (d) of the wake without it

Reads fig2_out/fig2_wake.nc and fig6_out/fig6_budget.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from v6_common import ps, IN, OUT, WAKE, STAT_KW, ev, line_axes, shade_windows, date_axis, hatch_above

ds = xr.open_dataset(IN + 'fig2_out/fig2_wake.nc')
f6 = xr.open_dataset(IN + 'fig6_out/fig6_budget.nc')
WINDOWS = ev(ds.attrs['windows'])
T_OFF = pd.Timestamp(ds.attrs['t_off'])
t = pd.to_datetime(ds.time.values)
PAIR = {'tide': WAKE['with'], 'notide': WAKE['without']}
GROUP_STYLE = {'footprint': dict(colour='black', ls='-', lw=1.8, label='footprint'),
               'footprint free': dict(colour='steelblue', ls='--', lw=1.5, label='footprint below resonance'),
               'footprint trapped': dict(colour='indianred', ls='-.', lw=1.5, label='footprint above resonance'),
               'central': dict(colour='darkgoldenrod', ls=':', lw=1.5, label='central slope'),
               'north': dict(colour='slategray', ls='-', lw=1.1, label='northern slope')}
t6 = pd.to_datetime(f6.time.values)
T_W0, T_W1 = (pd.Timestamp(x) for x in ev(f6.attrs['work_window']))
budget = ev(f6.attrs['budget'])
dt_h = float((t6[1] - t6[0])/pd.Timedelta(hours=1))
WORK = (t6 >= T_W0) & (t6 < T_W1)
TERM6 = {'tendency': dict(colour='darkgoldenrod', ls='-', lw=2.4, label='change of the wake kinetic energy'),
         'wind work': dict(colour='dimgray', ls='--', lw=1.6, label='wind work'),
         'internal tide': dict(colour='black', ls='-', lw=1.6, label='work of the internal tide'),
         'storm-made motion': dict(colour='deeppink', ls=(0, (6, 2, 1, 2)), lw=1.6, label='work of the storm-made motion'),
         'residual': dict(colour='mediumpurple', ls=(0, (3, 1, 1, 1)), lw=1.6, label='residual')}
TERM_WITHOUT = {'internal tide': 'kinematic pairing', 'storm-made motion': 'strain work of WIND minus BACK'}
TERMS6 = [str(x) for x in f6.term.values]

fig = plt.figure(figsize=(9.5, 9.8))
gs1 = GridSpec(1, 2, figure=fig, width_ratios=[0.85, 1.0], wspace=0.34, left=0.085, right=0.975, top=0.950,
               bottom=0.660)
gs2 = GridSpec(1, 2, figure=fig, wspace=0.12, left=0.085, right=0.975, top=0.405, bottom=0.205)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1])
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1], sharey=ax_c)

# (a) wake kinetic energy against the detuning of the flow under each wake
x = ds.dc_bin.values*1e3
cur = {p: ds[f'curveE_{p}'].values for p in PAIR}
have = np.isfinite(cur['tide'][:, 0]) | np.isfinite(cur['notide'][:, 0])
x_lo, x_hi = (x[have].min() - 0.6, x[have].max() + 0.6) if have.any() else (x.min(), x.max())
h_above = hatch_above(ax_a, x_hi)
for p, st in PAIR.items():
    c, ok = cur[p], np.isfinite(cur[p][:, 0])
    ax_a.fill_between(x[ok], c[ok, 1], c[ok, 2], facecolor=st['colour'], edgecolor='none', alpha=0.18, zorder=1)
    ax_a.plot(x[ok], c[ok, 0], color=st['colour'], linestyle=st['ls'], marker=st['marker'], markersize=5,
              markerfacecolor='white', markeredgewidth=1.2, linewidth=1.7, zorder=3)
ax_a.axvline(0.0, color='black', linestyle='solid', linewidth=0.8, zorder=2)
ax_a.set_xlim(x_lo, x_hi)
ax_a.set_yscale('log')
fin = np.r_[cur['tide'][:, 1:], cur['notide'][:, 1:]]
fin = fin[np.isfinite(fin) & (fin > 0)]
ax_a.set_ylim(10**np.floor(np.log10(fin.min())), 10**np.ceil(np.log10(fin.max())))
ax_a.set_xlabel(r'detuning of the flow under each wake ($10^{-3}$ cph)')
ax_a.set_ylabel(r'wake kinetic energy (J m$^{-2}$)')
line_axes(ax_a)
ps.panel(ax_a, 'a')
ax_a.legend(handles=[Line2D([0], [0], color=PAIR[p]['colour'], linestyle=PAIR[p]['ls'], marker=PAIR[p]['marker'],
                            markersize=5, markerfacecolor='white', markeredgewidth=1.2, label=PAIR[p]['label'])
                     for p in PAIR]
            + [Patch(facecolor='gray', edgecolor='none', alpha=0.3, label='5 to 95% of blocks'), h_above],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False, columnspacing=1.4)

# (b) with over without the tide through time by group
tb = pd.to_datetime(ds.time_band.values)
t0 = pd.Timestamp(WINDOWS['storm'][0])
rt, rb = ds.ratio_t.values, ds.ratio_band.values
groups = [str(g) for g in ds.group.values]
ax_b.set_xlim(t0, t[-1])
shade_windows(ax_b, WINDOWS)
ax_b.axhline(1.0, color='black', linestyle='solid', linewidth=0.8, zorder=2)
ax_b.axvline(T_OFF, color='dimgray', linestyle='solid', linewidth=0.9, zorder=2)
hb = []
for gname, st in GROUP_STYLE.items():
    if gname not in groups:
        continue
    gi = groups.index(gname)
    ok_t, ok_b = (t >= t0), (tb >= t0)
    if not np.isfinite(rt[gi][ok_t]).any():
        continue
    ax_b.fill_between(tb[ok_b], rb[gi, ok_b, 0], rb[gi, ok_b, 1], facecolor=st['colour'], edgecolor='none', alpha=0.15,
                      zorder=1)
    ax_b.plot(t[ok_t], rt[gi][ok_t], color=st['colour'], linestyle=st['ls'], linewidth=st['lw'], zorder=4)
    hb.append(Line2D([0], [0], color=st['colour'], linestyle=st['ls'], linewidth=st['lw'], label=st['label']))
ax_b.set_yscale('log', base=2)
fin = rb[:, tb >= t0, :]
fin = fin[np.isfinite(fin) & (fin > 0)]
lim = max(0.5, min(2.0, 1.05*np.max(np.abs(np.log2(fin))))) if fin.size else 1.0
ax_b.set_ylim(2**-lim, 2**lim)
ticks = [v for v in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0) if 2**-lim <= v <= 2**lim]
ax_b.set_yticks(ticks)
ax_b.set_yticklabels([f'{v:g}' for v in ticks])
ax_b.minorticks_off()
date_axis(ax_b, 3)
ax_b.set_ylabel('wake kinetic energy\nwith over without the tide')
line_axes(ax_b)
ps.panel(ax_b, 'b', x=0.012)
ax_b.legend(handles=hb + [Patch(facecolor='gray', edgecolor='none', alpha=0.3, label='5 to 95% of blocks')],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, columnspacing=1.0,
            handlelength=2.2)


# (c) and (d) the budget of each wake
def cum(x):
    c = np.zeros(x.size)
    kw = np.where(WORK)[0]
    c[kw] = np.cumsum(x[kw])*dt_h*3600.0
    c[kw[-1] + 1:] = c[kw[-1]]
    return c


for ax, w, lab in ((ax_c, 'with', 'c'), (ax_d, 'without', 'd')):
    ax.set_xlim(T_W0 - pd.Timedelta(hours=12), T_W1)
    ax.axvline(T_OFF, color='black', linewidth=0.8, zorder=2)
    date_axis(ax, 3)
    shade_windows(ax, WINDOWS, label=False)
    for k in TERMS6:
        v = f6[f'term_{w}_{k.replace(" ", "_").replace("-", "_")}'].values
        st = TERM6[k]
        ax.plot(t6[WORK], cum(v)[WORK]/1e3, color=st['colour'], linestyle=st['ls'], linewidth=st['lw'],
                zorder=4 if k == 'tendency' else 3)
    ax.axhline(0.0, color='black', linewidth=0.7, zorder=1)
    line_axes(ax)
    ps.panel(ax, lab)
    ax.legend(handles=[Line2D([0], [0], color=TERM6[k]['colour'], linestyle=TERM6[k]['ls'], linewidth=TERM6[k]['lw'],
                              label=TERM6[k]['label'] if w == 'with' else TERM_WITHOUT.get(k, TERM6[k]['label']))
                       for k in TERMS6],
              loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False, columnspacing=1.2,
              handlelength=2.6)
ax_c.set_ylabel(r'accumulated from 24 March (kJ m$^{-2}$)')
plt.setp(ax_d.get_yticklabels(), visible=False)
ax_c.set_title('wake with the tide' + '\n'
               + f'tide over wind work {budget["with"]["internal tide"]/budget["with"]["wind work"]:.2f}', **STAT_KW)
ax_d.set_title('wake without the tide' + '\n'
               + f'pairing over tide {budget["without"]["internal tide"]/budget["with"]["internal tide"]:.2f}', **STAT_KW)

fig.savefig(OUT + 'figS3_wakes_budget.png', dpi=300)
print('saved', OUT + 'figS3_wakes_budget.png')
