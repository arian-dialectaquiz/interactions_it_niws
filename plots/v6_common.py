"""
v6_common.py  (full_v5/new_figures)
-----------------------------------
Paths and the few helpers every v6 figure shares. The v6 figures are drawn
from the outputs already computed for Figures 1 to 8 of v5 (the netCDF files
and block tables under ../figN_out and ../region_out), so nothing is run again
from the model. Each plot script stays flat with its own GridSpec and copies
the panel code it needs; this module only holds what would otherwise be
repeated verbatim (paths, map frame, window shading, the smoothing of a map
and the block-bootstrap mean of a map).

Figure text carries no commas and no figurative names. Classes are named by
the sign of the flow detuning only (below resonance and above resonance).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sys.path.insert(0, '..')
import paper_style as ps  # noqa: E402

ps.apply()
# fonts sized for a figure 9.5 inches wide printed at the text width (about 8 pt on the page)
mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 11.5, 'legend.fontsize': 10,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11,
                     'hatch.color': 'dimgray', 'hatch.linewidth': 0.5})
FIG_W = 9.5                                  # width of every main figure (inches)
TXT_KW = dict(fontsize=10.5, color='dimgray')  # short notes inside or beside a panel
ps.CMAP_ENERGY = 'YlGnBu'                    # kinetic energies, light where small, readable with colour vision deficiency


def panel(ax, letter, x=0.02, y=0.98, colour='black'):
    """The panel letter inside the upper left corner on a white patch (larger than paper_style)."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
            color=colour, zorder=20, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))


ps.panel = panel

IN = '../'                                   # the v5 outputs
REGION = IN + 'region_out/region_maps.nc'
OUT = './'
NATURAL_EARTH = True

LAT_CRIT = -28.8
LON_LIM, LAT_LIM = (-52.0, -39.2), (-33.0, -21.0)
CBAR_BOX = [0.115, 0.930, 0.42, 0.030]       # along the top of the continent in every map
MAP_ASPECT = 1.17                            # height over width of the bight maps, as in Figure 1
MAP_SIGMA_R = 2                              # grid points of 6 km on the region maps (25 km)
MAP_SIGMA_S = 4                              # grid points of 3 km on the slope fields (25 km)
RAD2CPH = 3600.0/(2*np.pi)
DEG = 180.0/np.pi

# one style per quantity through the paper
WAKE = {'with': dict(colour=ps.RUN_COLOUR['FULL'], ls=ps.RUN_STYLE['FULL'], marker=ps.RUN_MARKER['FULL'],
                     label='wake with the tide'),
        'without': dict(colour=ps.RUN_COLOUR['WIND'], ls=ps.RUN_STYLE['WIND'], marker=ps.RUN_MARKER['WIND'],
                        label='wake without the tide')}
CLASS = {'free': dict(colour='steelblue', marker='o', label='below resonance'),
         'trapped': dict(colour='indianred', marker='^', label='above resonance')}
TERM = {'it': dict(colour='black', ls='-', lw=2.4, label='work of the internal tide'),
        'sig_it': dict(colour='black', ls='-', lw=1.8, label='tidal strain work'),
        'del_it': dict(colour='teal', ls='--', lw=1.3, label='tidal divergence work'),
        'z_it': dict(colour='sienna', ls='-.', lw=1.3, label='tidal vertical shear work'),
        'sm': dict(colour='deeppink', ls=(0, (3, 1, 1, 1)), lw=1.9, label='storm-made semidiurnal work'),
        'dPi': dict(colour='dimgray', ls=':', lw=1.5, label='wind work difference'),
        'dEdt': dict(colour='darkgoldenrod', ls='-', lw=2.4, label='tendency of the gain'),
        'wind': dict(colour='dimgray', ls='--', lw=1.6, label='wind work'),
        'residual': dict(colour='mediumpurple', ls=(0, (3, 1, 1, 1)), lw=1.6, label='residual')}
GAIN_FACE = 'khaki'
LOSS_HATCH = '////'
STAT_KW = dict(fontsize=9.5, color='dimgray', loc='right', pad=4, linespacing=1.25)
IN_LEG = dict(fontsize=9, frameon=True, framealpha=0.92, facecolor='white', edgecolor='none', borderpad=0.35,
              labelspacing=0.3, handlelength=1.8, handletextpad=0.5, borderaxespad=0.4)   # legends inside a panel
WIN_HATCH = {'pre': ps.WINDOW_HATCH['T1'], 'storm': ps.WINDOW_HATCH['T2'], 'wake': ps.WINDOW_HATCH['T3']}
WIN_LABEL = {'pre': 'before', 'storm': 'storm', 'wake': 'wake'}
white_on_black = [pe.Stroke(linewidth=2.0, foreground='black'), pe.Normal()]
black_on_white = [pe.Stroke(linewidth=2.4, foreground='white'), pe.Normal()]


def ev(text):
    return eval(text, {'nan': np.nan, 'inf': np.inf, 'np': np})


def extend_for(data, vmin, vmax):
    """The side towards which the data pass the colour limits and no other."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    lo, hi = bool(data.min() < vmin), bool(data.max() > vmax)
    return {(True, True): 'both', (True, False): 'min', (False, True): 'max'}.get((lo, hi), 'neither')


def nice(v):
    """The nearest of 1 2 and 5 times a power of ten."""
    e = 10**np.floor(np.log10(v))
    return e*min((1, 2, 5, 10), key=lambda m: abs(np.log10(v/(m*e))))


def smooth2d(a, s):
    g = np.isfinite(a)
    num = gaussian_filter(np.where(g, a, 0.0), s, mode='nearest')
    den = gaussian_filter(g.astype(float), s, mode='nearest')
    return np.where(g & (den > 0.3), num/np.maximum(den, 1e-6), np.nan)


def smooth(x, n):
    return np.convolve(np.pad(x, n//2, mode='edge'), np.ones(n)/n, mode='valid')[:x.size]


def geo_axes(ax, label, left_labels=True):
    ax.set_aspect('auto')
    ax.set_extent([LON_LIM[0], LON_LIM[1], LAT_LIM[0], LAT_LIM[1]], crs=ccrs.PlateCarree())
    if NATURAL_EARTH:
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.6,
                      linestyle='--', zorder=11)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.xlines = gl.ylines = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {'size': 10, 'color': 'dimgray'}
    ps.panel(ax, label)


def map_lines(ax, lon, lat, ocean, h, dc=None, crit=True):
    """Isobaths (200 m dashed and 2000 m solid) the contour of exact resonance and the planetary critical latitude."""
    import matplotlib.pyplot as plt
    ax.contour(lon, lat, np.where(ocean, h, np.nan), levels=[200.0, 2000.0], colors='dimgray',
               linestyles=['dashed', 'solid'], linewidths=0.5, zorder=3, transform=ccrs.PlateCarree())
    if dc is not None:
        cs = ax.contour(lon, lat, np.where(ocean, dc, np.nan), levels=[0.0], colors='white', linewidths=0.9,
                        zorder=5, transform=ccrs.PlateCarree())
        plt.setp(cs.collections if hasattr(cs, 'collections') else [cs], path_effects=white_on_black)
    if crit:
        ax.plot(LON_LIM, [LAT_CRIT, LAT_CRIT], color='black', linestyle='dashed', linewidth=1.0, zorder=6,
                transform=ccrs.PlateCarree())


def map_handles(dc=True):
    from matplotlib.lines import Line2D
    hand = []
    if dc:
        hand.append(Line2D([0], [0], color='white', linestyle='solid', linewidth=1.2, path_effects=white_on_black,
                           label=r'$\Delta_c = 0$'))
    hand += [Line2D([0], [0], color='black', linestyle='dashed', linewidth=1.0, label=r'$28.8^{\circ}$S'),
             Line2D([0], [0], color='dimgray', linestyle='dashed', linewidth=0.6, label='200 m'),
             Line2D([0], [0], color='dimgray', linestyle='solid', linewidth=0.6, label='2000 m')]
    return hand


def map_legend(ax, handles):
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.985, 0.015), ncol=2, fontsize=9,
              frameon=True, framealpha=0.92, facecolor='white', edgecolor='none', borderpad=0.4,
              labelspacing=0.3, columnspacing=0.9, handlelength=1.8)


def inset_colourbar(ax, cmap, norm, title, extend='neither', ticks=None):
    cax = ax.inset_axes(CBAR_BOX)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(title, fontsize=9.5, labelpad=2)
    cax.tick_params(labelsize=9, length=2, pad=1)
    cax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f'{v:g}'))
    cb.outline.set_linewidth(0.5)
    return cb


def cell_widths(fig_w, left, right, ratios, wspace):
    """Widths in inches of the columns of a GridSpec (the same arithmetic as matplotlib)."""
    n = len(ratios)
    cell = (right - left)*fig_w/(n + wspace*(n - 1))
    return [r*cell*n/sum(ratios) for r in ratios]


def line_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)


def shade_windows(ax, windows, label_y=1.02, inside=False, label=True):
    """Hatch the windows (before blank storm lines wake dots) and name them above the axis or inside it."""
    x0, x1 = (pd.Timestamp(mdates.num2date(v)).tz_localize(None) for v in ax.get_xlim())
    for w, (a0, a1) in windows.items():
        ax.axvspan(pd.Timestamp(a0), pd.Timestamp(a1), facecolor='none', edgecolor='silver', hatch=WIN_HATCH[w],
                   linewidth=0.0, zorder=0)
        mid_w = pd.Timestamp(a0) + (pd.Timestamp(a1) - pd.Timestamp(a0))/2
        if label and x0 <= mid_w <= x1:
            if inside:
                ax.text(mid_w, 0.97, WIN_LABEL[w], transform=ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=10.5, color='dimgray', zorder=6,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
            else:
                ax.text(mid_w, label_y, WIN_LABEL[w], transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=10.5, color='dimgray')


def date_axis(ax, interval=2):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(ps.english_date))


ABOVE_HATCH = '\\\\\\\\'


def hatch_above(ax, x_hi):
    """Hatch the side of a detuning axis where the flow is above resonance and return the legend handle for it."""
    from matplotlib.patches import Patch
    ax.axvspan(0.0, x_hi, facecolor='none', edgecolor='silver', hatch=ABOVE_HATCH, linewidth=0.0, zorder=0)
    return Patch(facecolor='none', edgecolor='silver', hatch=ABOVE_HATCH, label='flow above resonance')


def block_ids(shape, block):
    return (np.arange(shape[0])[:, None]//block)*1000 + (np.arange(shape[1])[None, :]//block)


def block_mean_band(values, mask, area, blk, rng, n=1000):
    """Area-weighted mean of a map over mask with a 5 to 95% block bootstrap."""
    m = mask & np.isfinite(values)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    v, a, b = values[m], area[m], blk[m]
    ids = np.unique(b)
    est = float(np.sum(v*a)/np.sum(a))
    if ids.size < 2:
        return est, np.nan, np.nan
    num = np.array([np.sum(v[b == i]*a[b == i]) for i in ids])
    den = np.array([np.sum(a[b == i]) for i in ids])
    draw = rng.integers(0, ids.size, size=(n, ids.size))
    bs = num[draw].sum(axis=1)/den[draw].sum(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))