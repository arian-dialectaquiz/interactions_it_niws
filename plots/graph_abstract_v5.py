"""
graphical_abstract_v5.py  (full_v5/new_figures)
-----------------------------------------------
Graphical abstract of "How a hurricane wake draws energy from the internal tide near the critical latitude".

One ocean section with no graphs. The four steps of the transfer are numbered on
the section and a box (seen from above) shows how the strain of the internal tide
works on the wake current and why the sign depends on the angle between them.

The numbers printed on the figure are typed in the block RESULTS below and come
from Section S10 and Figure 6 of the manuscript. Change them there if they change.
The tide drawn and named is the internal tide (the deviation of the semidiurnal
velocity from its depth mean). Its work on the wake is 354 J m-2. The tide as a
whole (barotropic and internal) did 319 and the barotropic tide -35.
Reads nothing. Saves graphical_abstract.png (Elsevier proportion 3984 by 1592 pixels).
Colours are named. Figure text carries no commas.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

OUT = './'

# --------------------------------------------------------------------------
# RESULTS (Section S10 and Figure 6 of the manuscript)
# --------------------------------------------------------------------------
TIDE_WORK = 354          # work of the internal tide on the wake (J m-2) (319 for the barotropic and internal tide)
SHARE_WIND = 7           # the same work as a share of the wind work (%) (354 of 4830)
LOSS_SHIFT_H = 5.4       # the tide this many hours earlier gives the largest loss
LATITUDE = '29°S'

# --------------------------------------------------------------------------
# colours (named)
# --------------------------------------------------------------------------
C_INK = 'darkslategray'
C_SKY_TOP, C_SKY_BOT = 'lightsteelblue', 'aliceblue'
C_SEA_TOP, C_SEA_BOT = 'lightcyan', 'steelblue'
C_SLOPE, C_SLOPE_EDGE = 'tan', 'saddlebrown'
C_TIDE = 'royalblue'
C_WAKE = 'black'
C_ENERGY, C_ENERGY_TXT = 'goldenrod', 'darkgoldenrod'
C_STORM = 'dimgray'
C_GAIN, C_LOSS = 'darkgoldenrod', 'firebrick'

# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
FIG_W, FIG_H, DPI = 13.28, 5.31, 300
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs_head = GridSpec(1, 1, figure=fig, left=0.012, right=0.988, top=0.985, bottom=0.895)
gs_sea = GridSpec(1, 1, figure=fig, left=0.012, right=0.988, top=0.880, bottom=0.030)
gs_how = GridSpec(1, 2, figure=fig, left=0.655, right=0.935, top=0.520, bottom=0.275, wspace=0.50)
ax_head = fig.add_subplot(gs_head[0])
ax = fig.add_subplot(gs_sea[0])
ax_gain = fig.add_subplot(gs_how[0])
ax_loss = fig.add_subplot(gs_how[1])


def arrow(a, p0, p1, colour, lw=2.0, scale=16, ls='-', z=6, alpha=1.0):
    a.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=scale, color=colour, linewidth=lw,
                                linestyle=ls, zorder=z, alpha=alpha, shrinkA=0, shrinkB=0))


def step(a, x, y, n, colour):
    a.text(x, y, str(n), ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=20,
           bbox=dict(boxstyle='circle,pad=0.22', facecolor=colour, edgecolor='white', linewidth=1.0))


# --------------------------------------------------------------------------
# the heading
# --------------------------------------------------------------------------
ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1); ax_head.axis('off')
ax_head.add_patch(Rectangle((0, 0), 1, 1, facecolor=C_INK, edgecolor='none'))
ax_head.text(0.012, 0.5, 'How a hurricane wake draws energy from the internal tide', ha='left', va='center',
             color='white', fontsize=15, fontweight='bold')
ax_head.text(0.988, 0.5, 'Hurricane Catarina (South Brazil Bight 2004)', ha='right', va='center', color='white',
             fontsize=11, fontstyle='italic')

# --------------------------------------------------------------------------
# the ocean section
# --------------------------------------------------------------------------
X0, X1, Z0, Z1 = 0.0, 20.0, -6.0, 2.2
ax.set_xlim(X0, X1); ax.set_ylim(Z0, Z1); ax.axis('off')
grad = np.linspace(0, 1, 200)[:, None]
ax.imshow(grad, extent=(X0, X1, 0, Z1), origin='lower', aspect='auto', zorder=0,
          cmap=LinearSegmentedColormap.from_list('sky', [C_SKY_BOT, C_SKY_TOP]))
ax.imshow(grad, extent=(X0, X1, Z0, 0), origin='lower', aspect='auto', zorder=0,
          cmap=LinearSegmentedColormap.from_list('sea', [C_SEA_BOT, C_SEA_TOP]))
xx = np.linspace(X0, X1, 900)
ax.plot(xx, 0.035*np.sin(7*xx), color=C_INK, linewidth=1.3, zorder=5)

# the continental slope
xs = np.linspace(X0, 4.2, 100)
zb = -0.55 - 5.6/(1 + np.exp(-(xs - 1.9)*2.4))
ax.add_patch(Polygon(np.r_[np.c_[xs, zb], [[4.2, Z0], [X0, Z0]]], closed=True, facecolor=C_SLOPE,
                     edgecolor=C_SLOPE_EDGE, linewidth=1.0, zorder=3))
ax.text(0.95, -4.7, 'continental\nslope', ha='center', va='center', fontsize=9.5, color=C_SLOPE_EDGE, zorder=4)

# the internal tide (isopycnals that heave and weaken past the wake)
for z_iso, amp, lw in ((-2.75, 0.30, 2.6), (-3.75, 0.36, 2.2), (-4.75, 0.30, 1.8)):
    xi = np.linspace(2.0, 11.9, 700)
    weak = 1.0 - 0.45/(1 + np.exp(-(xi - 8.0)*0.9))
    zi = z_iso + amp*weak*np.sin(2*np.pi*(xi - 2.0)/3.3)*np.tanh((xi - 2.0)/0.7)
    ok = zi > np.interp(xi, xs, zb, right=Z0 - 1) + 0.06
    ax.plot(xi[ok], zi[ok], color=C_TIDE, linewidth=lw, zorder=4, solid_capstyle='round')
arrow(ax, (2.55, -1.95), (4.35, -1.95), C_TIDE, lw=2.6, scale=18, z=7)
step(ax, 2.2, -1.36, 2, C_TIDE)
ax.text(2.55, -1.36, 'the internal tide\nleaves the slope', ha='left', va='center', fontsize=10.5, color=C_TIDE,
        fontweight='bold', zorder=8, linespacing=1.15)
ax.text(11.8, -4.25, 'the internal tide\nmoves on weaker', ha='right', va='center', fontsize=9.5, color=C_TIDE,
        fontstyle='italic', zorder=8)

# the hurricane and the wind
hx, hz = 5.2, 1.45
th = np.linspace(0.25, 2.2*np.pi, 200)
for rot in (0.0, np.pi):
    r = 0.075*th
    ax.plot(hx + 1.7*r*np.cos(th + rot), hz + 0.48*r*np.sin(th + rot), color=C_STORM, linewidth=3.0, zorder=6,
            solid_capstyle='round')
ax.add_patch(Ellipse((hx, hz), 0.30, 0.10, facecolor='white', edgecolor=C_STORM, linewidth=1.6, zorder=7))
arrow(ax, (11.6, 1.45), (6.6, 1.45), C_STORM, lw=1.4, ls=(0, (4, 3)), scale=14, z=5)
for xw in (4.4, 5.2, 6.0):
    arrow(ax, (xw, 0.86), (xw, 0.14), C_STORM, lw=1.4, scale=11, z=6)
step(ax, 6.75, 0.68, 1, C_STORM)
ax.text(7.1, 0.68, 'the hurricane wind sets the upper ocean turning', ha='left', va='center', fontsize=10.5,
        color=C_STORM, fontweight='bold', zorder=8)

# the wake (inertial currents that turn anticlockwise in the Southern Hemisphere)
for xc, rad in ((4.7, 0.58), (6.6, 0.68), (8.5, 0.58)):
    tt = np.linspace(0.10*np.pi, 1.88*np.pi, 140)
    ax.plot(xc + rad*np.cos(tt), -0.80 + 0.50*rad*np.sin(tt), color=C_WAKE, linewidth=2.5, zorder=7,
            solid_capstyle='round')
    arrow(ax, (xc + rad*np.cos(tt[-1] - 0.2), -0.80 + 0.50*rad*np.sin(tt[-1] - 0.2)),
          (xc + rad*np.cos(tt[-1] + 0.14), -0.80 + 0.50*rad*np.sin(tt[-1] + 0.14)), C_WAKE, lw=2.5, scale=17, z=8)
ax.text(9.45, -0.80, 'the wake\n(inertial currents)', ha='left', va='center', fontsize=10.5, color=C_WAKE,
        fontweight='bold', zorder=8, linespacing=1.15)

# the transfer from the tide to the wake
for xe in (5.65, 7.55):
    arrow(ax, (xe, -2.35), (xe, -1.38), C_ENERGY, lw=6.0, scale=27, z=9)
step(ax, 8.35, -1.50, 3, C_ENERGY_TXT)
ax.text(8.7, -1.50, 'the internal tide gives\nkinetic energy to the wake', ha='left', va='center', fontsize=10.5,
        color=C_ENERGY_TXT, fontweight='bold', zorder=10, linespacing=1.15)
ax.text(8.7, -2.17, f'({TIDE_WORK} J m$^{{-2}}$ or {SHARE_WIND}%\nof the wind work)', ha='left', va='center', linespacing=1.1,
        fontsize=9.5, color=C_ENERGY_TXT, zorder=10)

# the wake radiates downward
for xr in (4.4, 6.3, 8.2):
    arrow(ax, (xr, -3.0), (xr + 1.1, -5.25), C_WAKE, lw=1.7, ls=(0, (5, 3)), scale=15, z=6, alpha=0.9)
step(ax, 4.35, -5.62, 4, C_WAKE)
ax.text(4.7, -5.62, 'the wake carries the kinetic energy down within a week', ha='left', va='center',
        fontsize=10.5, color=C_WAKE, fontweight='bold', zorder=8)

# --------------------------------------------------------------------------
# how the internal tide works on the wake (seen from above)
# --------------------------------------------------------------------------
BX0, BX1, BZ0, BZ1 = 12.25, 19.78, -5.82, 0.32
ax.add_patch(FancyBboxPatch((BX0, BZ0), BX1 - BX0, BZ1 - BZ0, boxstyle='round,pad=0.0,rounding_size=0.25',
                            facecolor='white', edgecolor=C_INK, linewidth=1.2, alpha=0.95, zorder=13))
XM = 0.5*(BX0 + BX1)
ax.text(XM, -0.10, 'How the internal tide works on the wake (seen from above)', ha='center', va='center', fontsize=9.8,
        fontweight='bold', color=C_INK, zorder=14)
ax.text(XM, -0.58, 'its strain squeezes the water along one axis and stretches it along the other',
        ha='center', va='center', fontsize=9, color=C_TIDE, zorder=14)

ANG = np.deg2rad(35.0)
E_SQ = np.array([np.cos(ANG), np.sin(ANG)])              # the axis along which the strain of the internal tide squeezes
E_ST = np.array([-np.sin(ANG), np.cos(ANG)])             # the axis along which it stretches
for a_, e_cur, colour, head in ((ax_gain, E_SQ, C_GAIN, 'the wake gains'), (ax_loss, E_ST, C_LOSS, 'the wake loses')):
    a_.set_xlim(-1.5, 1.5); a_.set_ylim(-1.5, 1.5); a_.set_aspect('equal'); a_.axis('off'); a_.set_zorder(15)
    a_.add_patch(Circle((0, 0), 0.78, facecolor='aliceblue', edgecolor='gray', linewidth=0.9, linestyle=(0, (3, 3))))
    for sgn in (1, -1):
        arrow(a_, tuple(sgn*1.48*E_SQ), tuple(sgn*0.86*E_SQ), C_TIDE, lw=2.6, scale=15, z=5)
        arrow(a_, tuple(sgn*0.86*E_ST), tuple(sgn*1.48*E_ST), C_TIDE, lw=2.6, scale=15, z=5, ls=(0, (2.2, 1.4)))
    arrow(a_, (0, 0), tuple(0.72*e_cur), C_WAKE, lw=3.4, scale=19, z=8)
    a_.plot(0, 0, marker='o', markersize=4, color=C_WAKE, zorder=9)
    tt = np.linspace(np.deg2rad(200), np.deg2rad(262), 30)
    a_.plot(0.78*np.cos(tt), 0.78*np.sin(tt), color=C_WAKE, linewidth=1.6, zorder=7)
    arrow(a_, (0.78*np.cos(tt[-2]), 0.78*np.sin(tt[-2])), (0.78*np.cos(tt[-1] + 0.1), 0.78*np.sin(tt[-1] + 0.1)),
          C_WAKE, lw=1.6, scale=11, z=7)
    a_.text(0.5, 1.03, head, transform=a_.transAxes, ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=colour)
ax_gain.text(0.5, -0.04, 'wake current along the\nsqueezing axis (Catarina)', transform=ax_gain.transAxes,
             ha='center', va='top', fontsize=8.8, color=C_INK, linespacing=1.2)
ax_loss.text(0.5, -0.04, 'wake current along the\nstretching axis' + f' (tide {LOSS_SHIFT_H} h earlier)',
             transform=ax_loss.transAxes, ha='center', va='top', fontsize=8.8, color=C_INK, linespacing=1.2)

# the key of the box and the reason the push adds up
arrow(ax, (12.95, -4.68), (13.6, -4.68), C_WAKE, lw=3.0, scale=15, z=14)
ax.text(13.75, -4.68, 'wake current', ha='left', va='center', fontsize=8.8, color=C_INK, zorder=14)
arrow(ax, (16.05, -4.68), (15.45, -4.68), C_TIDE, lw=2.4, scale=13, z=14)
ax.text(16.2, -4.68, 'squeezing', ha='left', va='center', fontsize=8.8, color=C_TIDE, zorder=14)
arrow(ax, (17.75, -4.68), (18.35, -4.68), C_TIDE, lw=2.4, scale=13, z=14, ls=(0, (2.2, 1.4)))
ax.text(18.5, -4.68, 'stretching', ha='left', va='center', fontsize=8.8, color=C_TIDE, zorder=14)
ax.text(XM, -5.30, f'Near {LATITUDE} the axes of the strain turn with the wake current so the push\n'
        'repeats at every turn. The hour the wind subsides sets the angle (gain or loss)', ha='center', va='center', fontsize=8.6,
        color=C_INK, zorder=14, linespacing=1.3)

fig.savefig(OUT + 'graphical_abstract.png', dpi=DPI, facecolor='white')
print('saved', OUT + 'graphical_abstract.png')