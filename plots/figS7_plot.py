"""
figS7_plot.py  (full_v5/new_figures)  Supplementary Figure S7
--------------------------------------------------------------
The budget terms by class the decay of the wake by depth and the agreement of
the model phase with the measured phase (the panels of Figures 6 7 and 8 of
v5 that leave the main text).

  (a) the terms of the budget by class for the wake with the tide and (b)
      for the wake without it (block means with 5 to 95% block-bootstrap
      intervals)
  (c) the decay rate by depth over four days after the peak of each class by
      class and pair with 5 to 95% block-bootstrap bands for the wake with
      the tide (growth to the left of zero)
  (d) the agreement of the model phase with the measured phase of the blocks
      through the wake window (amplitude-weighted resultant of the phase
      error) for the free drift at the detuning of each block for one
      frequency for the wake and for a fixed phase against the level random
      phases give

Reads fig6_out/fig6_budget.nc fig7_out/fig7_fate.nc and fig8_out/fig8_model.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from v6_common import ps, IN, OUT, CLASS, STAT_KW, ev, line_axes, date_axis

f6 = xr.open_dataset(IN + 'fig6_out/fig6_budget.nc')
f7 = xr.open_dataset(IN + 'fig7_out/fig7_fate.nc')
f8 = xr.open_dataset(IN + 'fig8_out/fig8_model.nc')
TERMS6 = [str(x) for x in f6.term.values]
SHORT = {'tendency': 'change of\nkinetic energy', 'wind work': 'wind work', 'internal tide': 'internal tide',
         'storm-made motion': 'storm-made\nmotion', 'residual': 'residual'}
SHORT_WITHOUT = dict(SHORT, **{'internal tide': 'kinematic\npairing', 'storm-made motion': 'strain work of\nWIND minus BACK'})
horizon = ev(f8.attrs['horizon'])
t8 = pd.to_datetime(f8.tafter.values)
Z_SHOW = 500.0
zmid = f7.zbin.values
Z_EDGES = np.array(ev(f7.attrs['z_edges']))
show = Z_EDGES[:-1] < Z_SHOW
PAIR = {'with': dict(ls='-', lw=1.8, label='with the tide'),
        'without': dict(ls=ps.RUN_STYLE['WIND'], lw=1.5, label='without the tide')}
MODEL = dict(colour='firebrick', ls='--', lw=1.6, label='free drift (detuning of each block)')
ONE = dict(colour='darkgoldenrod', ls=(0, (1, 1)), lw=1.6, label='free drift (one frequency for the wake)')
FIXED = dict(colour='gray', ls='-', lw=1.2, label='fixed phase')

fig = plt.figure(figsize=(9.5, 9.8))
gs1 = GridSpec(1, 2, figure=fig, wspace=0.70, left=0.165, right=0.975, top=0.940, bottom=0.665)
gs2 = GridSpec(1, 2, figure=fig, width_ratios=[0.8, 1.2], wspace=0.34, left=0.085, right=0.975, top=0.480,
               bottom=0.225)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1])
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1])

# (a) and (b) the terms by class for the two wakes
ypos = np.arange(len(TERMS6))[::-1].astype(float)
for ax, iw, lab, title in ((ax_a, 0, 'a', 'wake with the tide'), (ax_b, 1, 'b', 'wake without the tide')):
    ct = f6.cls_terms.values[iw]
    for ic, c in enumerate(CLASS):
        off = 0.19 if c == 'free' else -0.19
        ok = np.isfinite(ct[ic, :, 0])
        ax.barh(ypos[ok] + off, (ct[ic, :, 2] - ct[ic, :, 1])[ok]/1e3, left=ct[ic, :, 1][ok]/1e3, height=0.36,
                color=CLASS[c]['colour'], alpha=0.25, edgecolor='none', zorder=1)
        ax.barh(ypos[ok] + off, ct[ic, :, 0][ok]/1e3, height=0.36, color=CLASS[c]['colour'], edgecolor='none', zorder=3)
    ax.axvline(0.0, color='black', linewidth=0.7, zorder=2)
    ax.set_yticks(ypos)
    ax.set_xlabel(r'kJ m$^{-2}$ over the work window')
    ax.set_title(title, fontsize=11, loc='center', pad=5)
    line_axes(ax)
    ax.grid(False, axis='y')
    ps.panel(ax, lab, x=0.955)
ax_a.set_yticklabels([SHORT[k] for k in TERMS6], fontsize=10.5)
ax_b.set_yticklabels([SHORT_WITHOUT[k] for k in TERMS6], fontsize=10.5)
xl = max(abs(v) for ax in (ax_a, ax_b) for v in ax.get_xlim())
ax_a.set_xlim(-xl, xl)
ax_b.set_xlim(-xl, xl)
ax_a.legend(handles=[Patch(facecolor=CLASS[c]['colour'], label=CLASS[c]['label']) for c in CLASS]
            + [Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='5 to 95% of the block mean')],
            loc='upper center', bbox_to_anchor=(1.35, -0.20), ncol=3, frameon=False, columnspacing=1.4)

# (c) the decay rate by depth after the peak
for c in CLASS:
    for p in PAIR:
        r = f7[f'rate_{p}_{c}'].values[show]
        ok = np.isfinite(r[:, 0])
        if p == 'with':
            ax_c.fill_betweenx(zmid[show][ok], r[ok, 1], r[ok, 2], facecolor=CLASS[c]['colour'], alpha=0.18,
                               edgecolor='none', zorder=1)
        ax_c.plot(r[ok, 0], zmid[show][ok], color=CLASS[c]['colour'], linestyle=PAIR[p]['ls'], linewidth=PAIR[p]['lw'],
                  zorder=3)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_c.set_ylim(Z_SHOW, 0.0)
ax_c.set_xlabel('rate after the peak (d$^{-1}$)\n[growth below 0 decay above]')
ax_c.set_ylabel('depth (m)')
line_axes(ax_c)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Line2D([0], [0], color=CLASS[c]['colour'], linewidth=1.8, label=CLASS[c]['label']) for c in CLASS]
            + [Line2D([0], [0], color='gray', linestyle=PAIR[p]['ls'], linewidth=PAIR[p]['lw'], label=PAIR[p]['label'])
               for p in PAIR]
            + [Patch(facecolor='gray', alpha=0.25, edgecolor='none', label='5 to 95% of blocks (with the tide)')],
            loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, columnspacing=1.2, handlelength=2.6)

# (d) the agreement of the model phase with the measured phase
for k, st in (('frozen', FIXED), ('one', ONE), ('drift', MODEL)):
    ax_d.plot(t8, f8[f'agree_{k}'].values, color=st['colour'], linestyle=st['ls'], linewidth=st['lw'], zorder=3)
ax_d.plot(t8, f8.agree_level.values, color='black', linestyle='dotted', linewidth=0.9, zorder=2)
ax_d.axhline(float(f8.attrs['horizon_level']), color='dimgray', linewidth=0.6, zorder=1)
ax_d.set_ylim(0.0, 1.02)
ax_d.set_xlim(t8[0], t8[-1])
ax_d.set_ylabel('agreement of the model phase\nwith the measured phase')
date_axis(ax_d, 2)
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.set_title(f'agreement halved after {horizon["drift"]:.1f} d (free drift)' + '\n'
               + f'and {horizon["frozen"]:.1f} d (fixed phase)', **STAT_KW)
ax_d.legend(handles=[Line2D([0], [0], color=st['colour'], linestyle=st['ls'], linewidth=st['lw'], label=st['label'])
                     for st in (MODEL, ONE, FIXED)]
            + [Line2D([0], [0], color='black', linestyle='dotted', linewidth=0.9, label='random phases')],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False, columnspacing=1.4, handlelength=2.6)

fig.savefig(OUT + 'figS7_class_decay_phase.png', dpi=300)
print('saved', OUT + 'figS7_class_decay_phase.png')
