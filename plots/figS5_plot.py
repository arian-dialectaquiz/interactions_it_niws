"""
figS5_plot.py  (full_v5/new_figures)  Supplementary Figure S5
--------------------------------------------------------------
The phase of the wake against the tide in time (the panels of Figure 4 of v5
that leave the main text).

  (a) cos(Phi) over the footprint blocks by bin of the detuning of the flow
      and time (amplitude-weighted)
  (b) cos(Phi) of the footprint through time for the wake with the tide and
      for the kinematic pairing
  (c) the coherence of the phase of each block over the wake window by bin of
      the detuning of the flow for the two pairings with the value a phase
      drifting at twice the detuning would give
  (d) the distribution of Phi over the footprint weighted by the largest work
      each column could do for the two pairings with their mean directions

Reads fig4_out/fig4_phase.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from v6_common import ps, IN, OUT, WAKE, black_on_white, ev, line_axes, shade_windows, date_axis, hatch_above

ds = xr.open_dataset(IN + 'fig4_out/fig4_phase.nc')
t = pd.to_datetime(ds.time.values)
WINDOWS = ev(ds.attrs['windows'])
T_OFF = pd.Timestamp(ds.attrs['t_off'])
T_W0, T_W1 = (pd.Timestamp(x) for x in ev(ds.attrs['work_window']))
AMP_FLOOR = float(ds.attrs['amp_floor'])
PAIRING = {'with': dict(WAKE['with']), 'without': dict(colour=WAKE['without']['colour'], ls=WAKE['without']['ls'],
                                                        label='kinematic pairing')}
cmap_cos = mpl.colors.ListedColormap(mpl.colormaps['gnuplot2'](np.linspace(0.07, 0.96, 256)), name='gnuplot2_trim')
norm_cos = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

fig = plt.figure(figsize=(9.5, 9.4))
gs1 = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.62, left=0.085, right=0.925, top=0.935, bottom=0.640)
gs2 = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 0.8], wspace=0.30, left=0.085, right=0.955, top=0.445, bottom=0.165)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1])
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1], projection='polar')

# (a) cos(Phi) against the detuning of the flow and time
z = ds.hovdc_re.values + 1j*ds.hovdc_im.values
phi_h, amp_h = np.angle(-z), np.abs(z)
tw = (t >= T_W0) & (t < T_W1)
have = ds.hovdc_n.values >= 1
floor_h = AMP_FLOOR*np.nanmedian(amp_h[tw][:, have]) if have.any() else 0.0
H = np.where((amp_h > floor_h) & have[None, :], np.cos(phi_h), np.nan)
xdc = ds.dc_bin.values*1e3
ok_b = np.isfinite(H).any(axis=0)
base = np.broadcast_to(np.where(ok_b, 0.0, np.nan)[:, None], (xdc.size, t.size))
ax_a.pcolormesh(t, xdc, base, cmap=mpl.colors.ListedColormap(['gainsboro']), shading='nearest', zorder=0, rasterized=True)
ax_a.pcolormesh(t, xdc, H.T, cmap=cmap_cos, norm=norm_cos, shading='nearest', zorder=1, rasterized=True)
ax_a.axvline(T_OFF, color='black', linestyle='solid', linewidth=0.9, path_effects=black_on_white, zorder=3)
ax_a.axhline(0.0, color='black', linestyle='solid', linewidth=0.9, path_effects=black_on_white, zorder=3)
ax_a.set_xlim(T_W0, T_W1)
ax_a.set_ylim(xdc[ok_b].min() - 0.5, xdc[ok_b].max() + 0.5)
ax_a.invert_yaxis()
ax_a.set_ylabel('detuning of the flow\n' + r'($10^{-3}$ cph) [below to above resonance]')
date_axis(ax_a, 3)
for w in ('storm', 'wake'):
    a0, a1 = (pd.Timestamp(x) for x in WINDOWS[w])
    ax_a.text(a0 + (a1 - a0)/2, 1.02, {'storm': 'storm', 'wake': 'wake'}[w], transform=ax_a.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=10.5, color='dimgray')
ps.panel(ax_a, 'a')
cax_a = ax_a.inset_axes([1.03, 0.0, 0.03, 1.0])
cb_a = mpl.colorbar.ColorbarBase(cax_a, cmap=cmap_cos, norm=norm_cos, orientation='vertical')
cb_a.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cb_a.set_label(r'$\cos\Phi$ [loss to gain]', fontsize=10.5)
cax_a.tick_params(labelsize=9.5, length=2, pad=1)
cb_a.outline.set_linewidth(0.5)
ax_a.legend(handles=[Line2D([0], [0], color='black', linewidth=0.9, path_effects=black_on_white,
                            label='wind stops and exact resonance'),
                     Patch(facecolor='gainsboro', edgecolor='none', label='below the amplitude floor')],
            loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, columnspacing=2.0)

# (b) cos(Phi) of the footprint through time
ax_b.set_xlim(T_W0, T_W1)
shade_windows(ax_b, WINDOWS)
cw, ck = np.cos(ds.phi_foot_with.values), np.cos(ds.phi_foot_without.values)
ax_b.fill_between(t, 0.0, cw, where=cw > 0, facecolor='khaki', edgecolor='none', interpolate=True, zorder=1)
ax_b.fill_between(t, 0.0, cw, where=cw < 0, facecolor='none', edgecolor='dimgray', hatch='////', linewidth=0.0,
                  interpolate=True, zorder=1)
ax_b.plot(t, ck, color=PAIRING['without']['colour'], linestyle=PAIRING['without']['ls'], linewidth=1.4, zorder=3)
ax_b.plot(t, cw, color=PAIRING['with']['colour'], linestyle=PAIRING['with']['ls'], linewidth=1.8, zorder=4)
ax_b.axhline(0.0, color='black', linewidth=0.7, zorder=2)
ax_b.set_ylim(-1.05, 1.05)
ax_b.set_ylabel(r'$\cos\Phi$ of the footprint')
date_axis(ax_b, 3)
line_axes(ax_b)
ps.panel(ax_b, 'b')
ax_b.legend(handles=[Line2D([0], [0], color=PAIRING[k]['colour'], linestyle=PAIRING[k]['ls'], linewidth=1.8,
                            label=PAIRING[k]['label']) for k in PAIRING]
            + [Patch(facecolor='khaki', edgecolor='none', label='gain'),
               Patch(facecolor='none', edgecolor='dimgray', hatch='////', label='loss')],
            loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, columnspacing=1.6)

# (c) the coherence of the phase of each block
x_lo, x_hi = (xdc[ok_b].min() - 0.6, xdc[ok_b].max() + 0.6)
h_above = hatch_above(ax_c, x_hi)
for k, var in (('without', 'coh_without'), ('with', 'coh_with')):
    st = PAIRING[k]
    v = ds[var].values
    ok = np.isfinite(v[:, 0])
    ax_c.fill_between(xdc[ok], v[ok, 1], v[ok, 2], facecolor=st['colour'], edgecolor='none', alpha=0.18, zorder=1)
    ax_c.plot(xdc[ok], v[ok, 0], color=st['colour'], linestyle=st['ls'], linewidth=1.8, marker='o', markersize=4.5,
              markerfacecolor='white', markeredgewidth=1.2, zorder=4)
pr = ds.coh_pred.values
ax_c.plot(xdc[ok_b], pr[ok_b], color='black', linestyle='dotted', linewidth=1.2, zorder=3)
ax_c.axvline(0.0, color='black', linewidth=0.8, zorder=2)
ax_c.set_xlim(x_lo, x_hi)
ax_c.set_ylim(0.0, 1.02)
ax_c.set_xlabel(r'detuning of the flow under the wake ($10^{-3}$ cph)')
ax_c.set_ylabel('coherence of the phase\nover the wake window')
line_axes(ax_c)
ps.panel(ax_c, 'c')
ax_c.legend(handles=[Line2D([0], [0], color=PAIRING[k]['colour'], linestyle=PAIRING[k]['ls'], linewidth=1.8, marker='o',
                            markersize=4.5, markerfacecolor='white', markeredgewidth=1.2, label=PAIRING[k]['label'])
                     for k in PAIRING]
            + [Line2D([0], [0], color='black', linestyle='dotted', linewidth=1.2, label='drift at twice the detuning'),
               Patch(facecolor='gray', edgecolor='none', alpha=0.3, label='5 to 95% of blocks'), h_above],
            loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, columnspacing=1.2)

# (d) the rose of the phase over the footprint
rb = ds.rose_bin.values
wdt = rb[1] - rb[0]
for k, var, z in (('with', 'rose_with', 3), ('without', 'rose_without', 4)):
    r = ds[var].values
    st = PAIRING[k]
    if k == 'with':
        ax_d.bar(rb, r, width=wdt, bottom=0.0, facecolor=st['colour'], edgecolor='none', alpha=0.35, zorder=z)
    else:
        ax_d.bar(rb, r, width=wdt, bottom=0.0, facecolor='none', edgecolor=st['colour'], linewidth=1.2, linestyle=st['ls'],
                 zorder=z)
    zz = (r*np.exp(1j*rb)).sum()
    ax_d.annotate('', xy=(np.angle(zz), 1.1*r.max()*abs(zz)/r.sum()*len(r)/2), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='-|>', color=st['colour'], linewidth=1.6), zorder=6)
ax_d.set_theta_zero_location('E')
ax_d.set_theta_direction(1)
ax_d.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax_d.set_xticklabels([r'0$^{\circ}$ gain', r'90$^{\circ}$', r'180$^{\circ}$ loss', r'$-90^{\circ}$'], fontsize=10.5)
ax_d.set_yticklabels([])
ax_d.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax_d.text(-0.10, 1.02, '(d)', transform=ax_d.transAxes, fontsize=13, fontweight='bold', ha='left', va='bottom')
ax_d.legend(handles=[Patch(facecolor=PAIRING['with']['colour'], alpha=0.35, label=PAIRING['with']['label']),
                     Patch(facecolor='none', edgecolor=PAIRING['without']['colour'], linewidth=1.2,
                           linestyle=PAIRING['without']['ls'], label=PAIRING['without']['label'])],
            loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)

fig.savefig(OUT + 'figS5_phase.png', dpi=300)
print('saved', OUT + 'figS5_phase.png')
