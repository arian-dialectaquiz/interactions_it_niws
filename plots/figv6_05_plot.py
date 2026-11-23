"""
figv6_05_plot.py  (full_v5/new_figures)
---------------------------------------
Figure 5 of v6. Where the energy of the wake goes.

  (a) near-inertial kinetic energy density of the wake with the tide by depth
      and time over the footprint blocks below resonance and (b) above
      resonance with the centroid depth of both wakes
  (c) the downward fraction of the wake energy through the wake window from
      the rotary vertical wavenumber spectra by class and pair
  (d) rotary vertical wavenumber spectra of the wake with the tide over the
      wake window (downward and upward parts) by class
  (e) the ratio with over without the tide of the wake kinetic energy by depth
      in the wake window by class with block-bootstrap bands

The decay rate by depth stays in the Supplementary Material. Reads
fig7_out/fig7_fate.nc of v5.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from v6_common import ps, IN, OUT, CLASS, STAT_KW, IN_LEG, ev, extend_for, line_axes, date_axis

ds = xr.open_dataset(IN + 'fig7_out/fig7_fate.nc')
t = pd.to_datetime(ds.time.values)
t_spec = pd.to_datetime(ds.tspec.values)
WINDOWS = ev(ds.attrs['windows'])
T_OFF = pd.Timestamp(ds.attrs['t_off'])
T_W0, T_W1 = (pd.Timestamp(x) for x in ev(ds.attrs['work_window']))
T_K0, T_K1 = (pd.Timestamp(x) for x in WINDOWS['wake'])
down_frac = ev(ds.attrs['down_frac'])
shear_ratio = ev(ds.attrs['shear_ratio'])
Z_SHOW = 500.0
zmid = ds.zbin.values
Z_EDGES = np.array(ev(ds.attrs['z_edges']))
kappa = ds.kappa.values
CLASS_LONG = {'free': r'blocks below resonance ($\Delta_c < 0$)', 'trapped': r'blocks above resonance ($\Delta_c > 0$)'}
PAIR = {'with': dict(ls='-', lw=1.8, label='with the tide'),
        'without': dict(ls=ps.RUN_STYLE['WIND'], lw=1.5, label='without the tide')}
CENT = {'with': dict(colour='black', ls='-', lw=1.8, label='centroid with the tide'),
        'without': dict(colour=ps.RUN_COLOUR['WIND'], ls=ps.RUN_STYLE['WIND'], lw=1.6,
                        label='centroid without the tide')}
cmap_e = mpl.colormaps[ps.CMAP_ENERGY]


def nice5(v):
    e = 10**np.floor(np.log10(v))
    return e*min((1, 2, 5, 10), key=lambda m: abs(m*e - v))


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
W = 9.5
H_TOP, H1, H_GAP, H2, H_BOT = 0.36, 2.5, 1.22, 2.1, 0.78
H = H_TOP + H1 + H_GAP + H2 + H_BOT
fig = plt.figure(figsize=(W, H))
gs1 = GridSpec(1, 2, figure=fig, wspace=0.10, left=0.095, right=0.905, top=1 - H_TOP/H, bottom=1 - (H_TOP + H1)/H)
gs2 = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.9], wspace=0.46, left=0.095, right=0.975,
               top=(H_BOT + H2)/H, bottom=H_BOT/H)
ax_a = fig.add_subplot(gs1[0])
ax_b = fig.add_subplot(gs1[1], sharey=ax_a)
ax_c = fig.add_subplot(gs2[0])
ax_d = fig.add_subplot(gs2[1])
ax_e = fig.add_subplot(gs2[2])

# --------------------------------------------------------------------------
# (a) and (b) the wake with the tide by depth and time by class
# --------------------------------------------------------------------------
show = Z_EDGES[:-1] < Z_SHOW
zedge_show = Z_EDGES[:show.sum() + 1]
tsel = (t >= T_W0) & (t < T_W1)
D = {c: ds[f'dens_with_{c}'].values[tsel][:, show] for c in CLASS}
vals = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in D.values()])
vmin, vmax = nice5(np.percentile(vals, 5)), nice5(np.percentile(vals, 99))
norm_e = LogNorm(vmin=vmin, vmax=vmax)
for ax, c, lab in ((ax_a, 'free', 'a'), (ax_b, 'trapped', 'b')):
    tt = mdates.date2num(t[tsel])
    t_edges = np.r_[tt - 0.5*(tt[1] - tt[0]), tt[-1] + 0.5*(tt[1] - tt[0])]
    ax.pcolormesh(t_edges, zedge_show, D[c].T, cmap=cmap_e, norm=norm_e, shading='flat', rasterized=True, zorder=1)
    for p in CENT:
        ax.plot(t[tsel], ds[f'centroid_{p}_{c}'].values[tsel], color=CENT[p]['colour'], linestyle=CENT[p]['ls'],
                linewidth=CENT[p]['lw'], zorder=4)
    ax.axvline(T_OFF, color='black', linewidth=0.8, zorder=3)
    ax.set_xlim(T_W0, T_W1)
    ax.set_ylim(Z_SHOW, 0.0)
    date_axis(ax, 3)
    ax.set_title(CLASS_LONG[c], fontsize=11, loc='center', pad=5,
                 color=CLASS[c]['colour'])
    ps.panel(ax, lab)
ax_a.set_ylabel('depth (m)')
plt.setp(ax_b.get_yticklabels(), visible=False)
cax = ax_b.inset_axes([1.03, 0.0, 0.03, 1.0])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap_e, norm=norm_e, orientation='vertical',
                               extend=extend_for(vals, vmin, vmax))
cb.set_label(r'wake kinetic energy with the tide (J m$^{-3}$)', fontsize=10.5)
cax.tick_params(labelsize=9.5, length=2, pad=1)
cb.outline.set_linewidth(0.5)
ax_a.legend(handles=[Line2D([0], [0], color=CENT[p]['colour'], linestyle=CENT[p]['ls'], linewidth=CENT[p]['lw'],
                            label=CENT[p]['label']) for p in CENT]
            + [Line2D([0], [0], color='black', linewidth=0.8, label='wind stops')],
            loc='upper center', bbox_to_anchor=(1.05, -0.16), ncol=3, frameon=False, columnspacing=1.6,
            handlelength=2.6)

# --------------------------------------------------------------------------
# (c) the downward fraction through the wake window
# --------------------------------------------------------------------------
for c in CLASS:
    for p in PAIR:
        ax_c.plot(t_spec, ds[f'down_{p}_{c}'].values, color=CLASS[c]['colour'], linestyle=PAIR[p]['ls'],
                  linewidth=PAIR[p]['lw'], zorder=3)
ax_c.axhline(0.5, color='black', linewidth=0.7, zorder=2)
ax_c.set_xlim(T_K0, T_K1)
ax_c.set_ylim(0.0, 1.0)
ax_c.set_ylabel('downward fraction\nof the wake energy')
date_axis(ax_c, 3)
line_axes(ax_c)
ps.panel(ax_c, 'c')
ax_c.set_title('with the tide {:.2f} below and {:.2f} above'.format(
    down_frac[('with', 'free')], down_frac[('with', 'trapped')]) + '\n'
    + 'without the tide {:.2f} and {:.2f}'.format(down_frac[('without', 'free')], down_frac[('without', 'trapped')]),
    **STAT_KW)
ax_c.legend(handles=[Line2D([0], [0], color=CLASS[c]['colour'], linewidth=1.8, label=CLASS[c]['label']) for c in CLASS]
            + [Line2D([0], [0], color='gray', linestyle=PAIR[p]['ls'], linewidth=PAIR[p]['lw'], label=PAIR[p]['label'])
               for p in PAIR],
            loc='lower right', ncol=1, **dict(IN_LEG, handlelength=2.2))

# --------------------------------------------------------------------------
# (d) the rotary vertical wavenumber spectra of the wake with the tide
# --------------------------------------------------------------------------
pos = kappa > 0
kpos = kappa[pos]
i_up = np.array([int(np.argmin(np.abs(kappa + k))) for k in kpos])
lam = 2*np.pi/kpos
order = np.argsort(lam)
for c in CLASS:
    S = ds[f'spec_with_{c}'].values
    down, up = S[pos][order], S[i_up][order]
    if not np.isfinite(down).any() or np.nansum(down) <= 0:
        continue
    ax_d.plot(lam[order], down, color=CLASS[c]['colour'], linestyle='-', linewidth=1.8, zorder=3)
    ax_d.plot(lam[order], up, color=CLASS[c]['colour'], linestyle='--', linewidth=1.4, zorder=3)
ax_d.set_xscale('log')
ax_d.set_yscale('log')
ax_d.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax_d.set_xlabel('vertical wavelength (m)')
ax_d.set_ylabel(r'rotary spectrum (m$^{2}$ s$^{-2}$)')
line_axes(ax_d)
ps.panel(ax_d, 'd')
ax_d.set_title('downward over upward\n{:.1f} below and {:.1f} above'.format(
    down_frac[('with', 'free')]/max(1 - down_frac[('with', 'free')], 1e-9),
    down_frac[('with', 'trapped')]/max(1 - down_frac[('with', 'trapped')], 1e-9)), **STAT_KW)
ax_d.legend(handles=[Line2D([0], [0], color='gray', linestyle='-', linewidth=1.8, label='downward'),
                     Line2D([0], [0], color='gray', linestyle='--', linewidth=1.4, label='upward')],
            loc='upper left', bbox_to_anchor=(0.0, 0.86), ncol=1, **dict(IN_LEG, handlelength=2.2))

# --------------------------------------------------------------------------
# (e) the ratio with over without the tide by depth
# --------------------------------------------------------------------------
for c in CLASS:
    r = ds[f'ratio_{c}'].values[show]
    ok = np.isfinite(r[:, 0]) & (r[:, 0] > 0)
    ax_e.fill_betweenx(zmid[show][ok], r[ok, 1], r[ok, 2], facecolor=CLASS[c]['colour'], alpha=0.18,
                       edgecolor='none', zorder=1)
    ax_e.plot(r[ok, 0], zmid[show][ok], color=CLASS[c]['colour'], linestyle='-', linewidth=1.8, zorder=3)
ax_e.axvline(1.0, color='black', linewidth=0.8, zorder=2)
ax_e.set_ylim(Z_SHOW, 0.0)
ax_e.set_xlabel('wake kinetic energy\nwith over without the tide')
ax_e.set_ylabel('depth (m)')
line_axes(ax_e)
ps.panel(ax_e, 'e')
ax_e.set_title('shear variance with over without' + '\n' + '{:.2f} below and {:.2f} above'.format(
    shear_ratio['free'], shear_ratio['trapped']), **STAT_KW)
ax_e.legend(handles=[Patch(facecolor='gray', alpha=0.25, edgecolor='none', label='5 to 95%\nof blocks')],
            loc='lower right', ncol=1, **dict(IN_LEG, handlelength=1.4))

fig.savefig(OUT + 'figv6_5_fate.png', dpi=300)
print('saved', OUT + 'figv6_5_fate.png')