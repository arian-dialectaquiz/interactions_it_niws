"""
fig_01_plot.py  (full_v5, revision 4)
-------------------------------------
Figure 1. The region, the model grids, the storm, the tide and the places on
the slope where near-inertial waves at half the tidal frequency can propagate.

The maps return to the look of the earlier Figure 1 (cartopy, cmocean, the
colourbar on land, the station markers of the series), and the figure carries
four panels that read as one story.

  (a) the 1 km domain, its bathymetry, isobaths, sectors, landmarks, moorings
      and the best track, with the nesting of the 1 km grid in the 3 km grid
      in an inset placed in the empty corner of the frame
  (b) the detuning of the joint experiment in the storm window
  (c) the share of the slope with negative detuning by latitude in the three
      windows, on the latitude axis of the maps
  (d) the semidiurnal envelope, the wind stress over the southern slope and
      the three windows

The maps keep their geographic proportions, the colourbars sit on the
continent at the top left with their title below, the legends sit offshore in
the empty corner or on the continent under the colourbar, the moorings are
lettered on the map, and a colourbar extends only towards the side the data
actually pass. The isobaths are the 200 and 2000 m of the slope band used in
every later figure. The rotary spectrum goes to Figure 2 and the agreement
between experiments stays in the report.

Reads fig1_out/fig1_v5.nc, fig1_track.csv and fig1_stations.csv written by
fig_01_compute.py, and the two grid files below for the maps.
All colours are named colours, lines differ by style as well as by colour, and
a colourbar extends only where the data pass its limits.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import sys
sys.path.insert(0, '..')
import paper_style as ps

ps.apply()
from v6_common import panel as _panel    # the larger panel letters of v6 (v6_common also sets the v6 rcParams)
ps.panel = _panel
# fonts of the 11 inch opening figure scaled to print at about 8 pt on the page
mpl.rcParams.update({'font.size': 12.5, 'axes.labelsize': 12.5, 'legend.fontsize': 10.5,
                     'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
                     'hatch.color': 'dimgray', 'hatch.linewidth': 0.5})

OUTDIR = '../fig1_out/'
CHILD = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc'
PARENT = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/3km/inputs/deproas_spongebob_grd_cropped_3km.nc'
NATURAL_EARTH = True        # coastline and land from cartopy, as in the earlier figure

ds = xr.open_dataset(OUTDIR + 'fig1_v5.nc')
track = pd.read_csv(OUTDIR + 'fig1_track.csv', parse_dates=['time'])
sta = pd.read_csv(OUTDIR + 'fig1_stations.csv')
child, parent = xr.open_dataset(CHILD), xr.open_dataset(PARENT)

RAD2CPH = 3600.0/(2*np.pi)
LAT_CRIT = float(ds.attrs['lat_crit'])
EPOCHS = eval(ds.attrs['epochs'])
REGION_ETA = eval(ds.attrs['region_eta'])
LANDFALL = pd.Timestamp(ds.attrs['landfall'])
T_OFF = pd.Timestamp(ds.attrs['t_wind_off']) if ds.attrs['t_wind_off'] else None
MAP_EPOCH = 'T2'
LANDMARKS = {'CF': (-23.018, -41.970), 'SSI': (-23.831, -45.409), 'CSM': (-28.600, -48.809)}
LON_LIM, LAT_LIM = (-52.0, -39.2), (-33.0, -21.0)
INSET_LIM = (-58.0, -30.0, -41.0, -13.0)     # lon0 lon1 lat0 lat1 of the nesting inset
DC_LIM = 8.0                 # 1e-3 cph
CBAR_BOX = [0.115, 0.935, 0.45, 0.028]

MOOR_STYLE = {'A': dict(marker='*', color='darkgoldenrod', ms=12),
              'B': dict(marker='v', color='mediumorchid', ms=8),
              'C': dict(marker='D', color='teal', ms=7),
              'D': dict(marker='s', color='deepskyblue', ms=7),
              'E': dict(marker='^', color='black', ms=8),
              'F': dict(marker='X', color='firebrick', ms=9)}
ISOBATHS = [(200, 'dimgray', 'dashed', '200 m'),
            (2000, 'dimgray', 'solid', '2000 m')]
TRACK_STYLE = {'TD': ('yellowgreen', 'o', 'tropical depression'),
               'TS': ('mediumorchid', 'o', 'tropical storm'),
               'SS': ('firebrick', 'o', 'subtropical storm'),
               'HU': ('darkorange', 'o', 'hurricane'),
               'EX': ('slategray', 's', 'extratropical')}
GRID_STYLE = {'parent': ('black', 'solid', 1.0, '3 km grid'),
              'child': ('firebrick', 'solid', 1.6, '1 km grid')}

lon_r, lat_r = child.lon_rho.values, child.lat_rho.values
h_r, mask_r = child.h.values, child.mask_rho.values
lon_b, lat_b, h_b = ds.lon.values, ds.lat.values, ds.h.values


def outline(g):
    lo, la = g.lon_rho.values, g.lat_rho.values
    return (np.concatenate([lo[0], lo[:, -1], lo[-1, ::-1], lo[::-1, 0]]),
            np.concatenate([la[0], la[:, -1], la[-1, ::-1], la[::-1, 0]]))


def geo_axes(ax, label, left_labels=True):
    ax.set_aspect('auto')
    ax.set_extent([LON_LIM[0], LON_LIM[1], LAT_LIM[0], LAT_LIM[1]], crs=ccrs.PlateCarree())
    if NATURAL_EARTH:
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3,
                      color='gray', alpha=0.6, linestyle='--', zorder=11)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.xlines = gl.ylines = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {'size': 11.2, 'color': 'dimgray'}
    ps.panel(ax, label)


def extend_for(data, vmin, vmax):
    """The side towards which the data pass the colour limits, and no other."""
    lo = bool(np.nanmin(data) < vmin)
    hi = bool(np.nanmax(data) > vmax)
    return {(True, True): 'both', (True, False): 'min', (False, True): 'max'}.get((lo, hi),
                                                                                'neither')


def inset_colourbar(ax, cmap, norm, title, extend, ticks=None):
    """Horizontal colourbar along the top of the continent, its title below."""
    cax = ax.inset_axes(CBAR_BOX)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal',
                                   extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(title, fontsize=10, labelpad=2)
    cax.tick_params(labelsize=9.5, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    return cb


HALO = [pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()]


def add_moorings(ax, letters=False):
    for r in sta.itertuples():
        st = MOOR_STYLE.get(r.name, dict(marker='o', color='dimgray', ms=7))
        ax.plot(r.lon, r.lat, linestyle='none', marker=st['marker'], color=st['color'],
                markersize=st['ms']*0.85, markeredgecolor='black', markeredgewidth=0.6,
                zorder=12, transform=ccrs.PlateCarree())
        if letters:
            ax.text(r.lon + 0.22, r.lat + 0.10, r.name, fontsize=10.5, fontweight='bold', ha='left',
                    va='bottom', zorder=13, path_effects=HALO, transform=ccrs.PlateCarree())


# --------------------------------------------------------------------------
# layout. The maps keep the proportions of the region they show, about 1.05
# in height per unit of width once longitude is scaled by the cosine of the
# central latitude, and the profile of (c) shares their latitude axis.
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(11.0, 8.4))
gs1 = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.42], wspace=0.14,
               left=0.080, right=0.945, top=0.975, bottom=0.470)
gs2 = GridSpec(1, 1, figure=fig, left=0.080, right=0.910, top=0.290, bottom=0.100)
ax_a = fig.add_subplot(gs1[0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs1[1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs1[2])
ax_d = fig.add_subplot(gs2[0])

# --------------------------------------------------------------------------
# (a) the region
# --------------------------------------------------------------------------
cmap_h, norm_h = cmocean.cm.deep, mpl.colors.Normalize(vmin=0, vmax=2800)
h_ocean = np.where(mask_r == 1, h_r, np.nan)
ax_a.pcolormesh(lon_r, lat_r, h_ocean, cmap=cmap_h, norm=norm_h, shading='auto', zorder=0,
                rasterized=True, transform=ccrs.PlateCarree())
for lev, col, ls, _ in ISOBATHS:
    ax_a.contour(lon_r, lat_r, h_ocean, levels=[lev], colors=col, linestyles=ls,
                 linewidths=0.6, zorder=3, transform=ccrs.PlateCarree())
geo_axes(ax_a, 'a')
nrow = lon_r.shape[0]
for name, (lo, hi) in REGION_ETA.items():
    if 0 < lo < nrow:
        ax_a.plot(lon_r[lo], lat_r[lo], color='black', linestyle='dashed', linewidth=0.9,
                  alpha=0.75, zorder=4, transform=ccrs.PlateCarree())
    r = (lo + min(hi, nrow - 1))//2
    cols = np.where(mask_r[r] == 1)[0]
    if cols.size:
        c = cols[int(0.70*(cols.size - 1))]
        ax_a.text(lon_r[r, c], lat_r[r, c], name, fontsize=11.2, ha='center', va='center', color='dimgray',
                  zorder=14, path_effects=HALO, transform=ccrs.PlateCarree())
for name, (la, lo) in LANDMARKS.items():
    ax_a.plot(lo, la, marker='D', color='black', markersize=4, linestyle='none', zorder=13,
              transform=ccrs.PlateCarree())
    ax_a.text(lo - 0.25, la + 0.15, name, fontsize=9.8, fontweight='bold', ha='center',
              va='bottom', zorder=13, path_effects=HALO, transform=ccrs.PlateCarree())
ax_a.plot(track.lon, track.lat, color='black', linestyle='dashed', linewidth=0.8, alpha=0.7,
          zorder=6, transform=ccrs.PlateCarree())
for typ, (col, mk, _) in TRACK_STYLE.items():
    sel = track['type'] == typ
    if sel.any():
        ax_a.scatter(track.lon[sel], track.lat[sel], c=col, marker=mk, s=24, edgecolors='black',
                     linewidths=0.4, zorder=7, transform=ccrs.PlateCarree())
add_moorings(ax_a, letters=True)
inset_colourbar(ax_a, cmap_h, norm_h, 'depth (m)', extend_for(h_ocean, 0.0, 2800.0), ticks=[0, 1000, 2000])
hand_a = [Line2D([0], [0], marker=mk, color=col, linestyle='none', markersize=5,
                 markeredgecolor='black', markeredgewidth=0.4, label=lab)
          for typ, (col, mk, lab) in TRACK_STYLE.items() if (track['type'] == typ).any()]
hand_a += [Line2D([0], [0], linestyle=ls, color=col, linewidth=1, label=lab)
           for lev, col, ls, lab in ISOBATHS]
hand_a += [Line2D([0], [0], linestyle='none', marker='D', color='black', markersize=4, label='landmark')]
ax_a.legend(handles=hand_a, loc='upper center', bbox_to_anchor=(0.5, -0.065), ncol=3, fontsize=10,
            frameon=False, labelspacing=0.3, columnspacing=1.0, handlelength=1.6, handletextpad=0.4)

# the nesting, in the corner of the frame the 1 km domain leaves empty
plon, plat = outline(parent)
pad = 0.6
ax_in = ax_a.inset_axes([0.715, 0.015, 0.265, 0.235], projection=ccrs.PlateCarree())
ax_in.set_extent([plon.min() - pad, plon.max() + pad, plat.min() - pad, plat.max() + pad],
                 crs=ccrs.PlateCarree())
ax_in.set_aspect('auto')
ax_in.pcolormesh(parent.lon_rho.values, parent.lat_rho.values,
                 np.where(parent.mask_rho.values == 1, parent.h.values, np.nan),
                 cmap=cmap_h, norm=mpl.colors.Normalize(vmin=0, vmax=5000),
                 shading='auto', zorder=0, rasterized=True, transform=ccrs.PlateCarree())
if NATURAL_EARTH:
    ax_in.coastlines(linewidth=0.4)
    ax_in.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
for g, key in ((parent, 'parent'), (child, 'child')):
    col, ls, lw, lab = GRID_STYLE[key]
    elo, ela = outline(g)
    ax_in.plot(elo, ela, color=col, linestyle=ls, linewidth=lw, zorder=5, label=lab,
               transform=ccrs.PlateCarree())
ax_in.plot(track.lon, track.lat, color='black', linestyle='dashed', linewidth=0.5, zorder=6,
           transform=ccrs.PlateCarree())
ax_in.legend(loc='upper left', fontsize=7.7, frameon=True, framealpha=0.92, facecolor='white', edgecolor='none',
             borderpad=0.25, labelspacing=0.2, handlelength=1.4, handletextpad=0.3)
for sp in ax_in.spines.values():
    sp.set_linewidth(0.8)

# --------------------------------------------------------------------------
# (b) the detuning of the joint experiment in the storm window
# --------------------------------------------------------------------------
cmap_dc, norm_dc = cmocean.cm.balance, mpl.colors.Normalize(vmin=-DC_LIM, vmax=DC_LIM)
dc = np.where(ds.ocean.values, ds.dc.sel(run='FULL', epoch=MAP_EPOCH).values*RAD2CPH*1e3, np.nan)
zf = ds.zof.sel(run='FULL', epoch=MAP_EPOCH).values
ax_b.pcolormesh(lon_b, lat_b, dc, cmap=cmap_dc, norm=norm_dc, shading='auto', zorder=0,
                rasterized=True, transform=ccrs.PlateCarree())
ax_b.contour(lon_b, lat_b, dc, levels=[0.0], colors='black', linewidths=1.0, zorder=6,
             transform=ccrs.PlateCarree())
ax_b.plot(LON_LIM, [LAT_CRIT, LAT_CRIT], color='black', linestyle='dashed', linewidth=1.0,
          zorder=6, transform=ccrs.PlateCarree())
strained = (np.abs(0.5*zf) > ds.attrs['rossby_limit']).astype(float)
ax_b.contourf(lon_b, lat_b, strained, levels=[0.5, 1.5], colors='none', hatches=['///'],
              zorder=4, transform=ccrs.PlateCarree())
for lev, col, ls, _ in ISOBATHS:
    ax_b.contour(lon_b, lat_b, np.where(ds.ocean.values, h_b, np.nan), levels=[lev],
                 colors=col, linestyles=ls, linewidths=0.5, zorder=3,
                 transform=ccrs.PlateCarree())
geo_axes(ax_b, 'b', left_labels=False)
add_moorings(ax_b)
t0, t1 = EPOCHS[MAP_EPOCH]
inset_colourbar(ax_b, cmap_dc, norm_dc,
                r'$\Delta_c$ ' + f'{int(t0[8:10])} to {int(t1[8:10])} March' + r' ($10^{-3}$ cph)',
                extend_for(dc, -DC_LIM, DC_LIM), ticks=[-DC_LIM, -DC_LIM/2, 0, DC_LIM/2, DC_LIM])
hand_b = [Line2D([0], [0], color='black', linestyle='solid', linewidth=1.0, label=r'$\Delta_c = 0$'),
          Line2D([0], [0], color='black', linestyle='dashed', linewidth=1.0, label=r'$28.8^{\circ}$S'),
          Patch(facecolor='white', edgecolor='dimgray', hatch='///', label=r'$|\zeta/2f| > 0.25$'),
          Line2D([0], [0], linestyle='none', marker='o', color='dimgray', markersize=5, markeredgecolor='black',
                 markeredgewidth=0.5, label='moorings A to F')]
hand_b += [Line2D([0], [0], linestyle=ls, color=col, linewidth=0.8, label=lab) for lev, col, ls, lab in ISOBATHS]
ax_b.legend(handles=hand_b, loc='lower right', bbox_to_anchor=(0.985, 0.015), ncol=2,
            fontsize=9.1, frameon=True, framealpha=0.92, facecolor='white', edgecolor='none', borderpad=0.4,
            labelspacing=0.3, columnspacing=0.9, handlelength=1.8, handletextpad=0.4)

# --------------------------------------------------------------------------
# (c) where the slope is open to the subharmonic, by latitude and window
# --------------------------------------------------------------------------
lat_bin = ds.lat_bin.values
WIN_LINE = {'T1': ('steelblue', 'dashed'), 'T2': ('firebrick', 'solid'),
            'T3': ('darkgoldenrod', 'dotted')}
ax_c.plot(ds.free_plan.values, lat_bin, color='darkgray', linestyle='solid', linewidth=3.0,
          alpha=0.6, drawstyle='steps-mid', label='planetary', zorder=1)
for ep, (col, ls) in WIN_LINE.items():
    ax_c.plot(ds.free_prof.sel(run='FULL', epoch=ep).values, lat_bin, color=col, linestyle=ls,
              linewidth=1.5, label=ps.WINDOW_LABEL[ep], zorder=3)
ax_c.axhline(LAT_CRIT, color='black', linestyle='dashed', linewidth=0.9)
ax_c.set_ylim(LAT_LIM)
ax_c.set_xlim(-0.02, 1.02)
ax_c.set_xticks([0.0, 0.5, 1.0])
ax_c.set_yticks([-31.5, -30.0, -28.5, -27.0, -25.5, -24.0, -22.5])
ax_c.yaxis.tick_right()
ax_c.yaxis.set_major_formatter(LATITUDE_FORMATTER)
ax_c.tick_params(axis='y', labelsize=11.2, labelcolor='dimgray')
ax_c.set_xlabel('share of the slope\nwith ' + r'$\Delta_c < 0$', fontsize=11.2)
ax_c.grid(True, linestyle=':', alpha=0.4)
ps.panel(ax_c, 'c')
ax_c.spines[['top']].set_visible(False)
ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9.8, frameon=False,
            columnspacing=1.0, handlelength=2.0)

# --------------------------------------------------------------------------
# (d) the tide and the storm in time
# --------------------------------------------------------------------------
t = pd.to_datetime(ds.time.values)
for ep, (a0, a1) in EPOCHS.items():
    ax_d.axvspan(pd.Timestamp(a0), pd.Timestamp(a1), facecolor='none', edgecolor='silver',
                 hatch=ps.WINDOW_HATCH[ep], linewidth=0.0, zorder=0)
    ax_d.text(pd.Timestamp(a0) + (pd.Timestamp(a1) - pd.Timestamp(a0))/2, 1.02,
              ps.WINDOW_LABEL[ep], transform=ax_d.get_xaxis_transform(), ha='center',
              va='bottom', fontsize=11.2, color='dimgray')
ax_d.plot(t, ds.env.values, color='lightsteelblue', linestyle='solid', linewidth=0.8, zorder=2)
ax_d.plot(t, ds.env_smooth.values, color=ps.RUN_COLOUR['TIDE'], linestyle='solid',
          linewidth=1.8, label='semidiurnal envelope', zorder=3)
tn = mdates.date2num(t)
for key, marker, name in (('t_spring', '^', 'spring'), ('t_neap', 'v', 'neap')):
    tt = pd.to_datetime([s for s in ds.attrs[key].split(',') if s])
    if len(tt):
        ax_d.plot(tt, np.interp(mdates.date2num(tt), tn, ds.env_smooth.values), linestyle='none',
                  marker=marker, markersize=6.5, markerfacecolor='white', zorder=4,
                  markeredgecolor=ps.RUN_COLOUR['TIDE'], markeredgewidth=1.2, label=name)
ax_d.set_ylim(0.0, 1.15*float(np.nanmax(ds.env.values)))
ax_d.set_ylabel('semidiurnal\namplitude (m)')
ax_d.axvline(LANDFALL, color='black', linestyle='solid', linewidth=1.0, label='landfall')
if T_OFF is not None:
    ax_d.axvline(T_OFF, color='black', linestyle='dotted', linewidth=1.3, label='wind stops')
ax_tau = ax_d.twinx()
ax_tau.plot(t, ds.tau_foot_FULL.values, color='black', linestyle='dashdot', linewidth=0.9,
            label='wind stress over the southern slope')
ax_tau.set_yscale('log')
ax_tau.set_ylim(5e-3, 5.0)
ax_tau.set_ylabel('wind stress (Pa)')
ax_d.set_xlim(t[0], t[-1])
ax_d.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax_d.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(ps.english_date))
ps.panel(ax_d, 'd', x=0.006, y=0.96)
hd, ld = ax_d.get_legend_handles_labels()
ht, lt = ax_tau.get_legend_handles_labels()
ax_d.spines[['top']].set_visible(False)
ax_tau.spines[['top']].set_visible(False)
ax_d.legend(hd + ht, ld + lt, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=6,
            frameon=False, columnspacing=1.4)

fig.savefig('./figv6_1_setting.png', dpi=300)
print('saved ./figv6_1_setting.png')