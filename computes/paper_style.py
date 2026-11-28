"""
paper_style.py
--------------
One style for every figure of full_v5. It holds constants and three small
helpers only, so each plot script stays flat and every axis is still placed by
its own GridSpec.

"""

import matplotlib as mpl
import cmocean  # noqa: F401  registers the cmo.* colour maps with matplotlib
import matplotlib.dates as mdates

# ------------------------------------------------------------------ experiments
RUNS = ['BACK', 'TIDE', 'WIND', 'FULL']
RUN_COLOUR = {'BACK': 'grey', 'TIDE': 'royalblue', 'WIND': 'darkorange',
              'FULL': 'black', 'SUP': 'mediumorchid'}
RUN_STYLE = {'BACK': ':', 'TIDE': '--', 'WIND': '-.', 'FULL': '-',
             'SUP': (0, (5, 1, 1, 1, 1, 1))}
RUN_WIDTH = {'BACK': 1.4, 'TIDE': 1.3, 'WIND': 1.3, 'FULL': 1.6, 'SUP': 1.3}
RUN_MARKER = {'BACK': 's', 'TIDE': '^', 'WIND': 'D', 'FULL': 'o', 'SUP': 'X'}
RUN_LABEL = {'BACK': 'current only', 'TIDE': 'tide', 'WIND': 'storm',
             'FULL': 'tide and storm', 'SUP': 'superposition'}

# ------------------------------------------------- detuning classes and sectors
CLASS_COLOUR = {'free': 'teal', 'forbidden': 'sienna'}
CLASS_STYLE = {'free': '-', 'forbidden': '--'}
CLASS_HATCH = {'free': '', 'forbidden': '////'}
SECTOR_COLOUR = {'South': 'black', 'Central': 'steelblue', 'North': 'darkgoldenrod'}
SECTOR_STYLE = {'South': '-', 'Central': '--', 'North': ':'}
SECTOR_MARKER = {'South': 'o', 'Central': 's', 'North': '^'}

# ------------------------------------------------------------ analysis windows
WINDOW_HATCH = {'T1': '', 'T2': '////', 'T3': '....'}
WINDOW_LABEL = {'T1': 'before', 'T2': 'storm', 'T3': 'wake'}

# ------------------------------------------------------------------ colour maps
CMAP_DIV = 'PuOr_r'        # signed fields, orange positive and purple negative
CMAP_SEQ = 'cividis'       # fractions and probabilities
CMAP_ENERGY = 'terrain_r'  # kinetic energies on maps, light where small and saturated where large, limits from the data with extend both
CMAP_STRESS = 'magma_r'    # forcing
CMAP_PHASE = 'twilight'    # cyclic quantities

LAND = 'lightgrey'
ISOBATH = 'dimgrey'
REFERENCE = 'firebrick'    # planetary or theoretical reference curves

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def apply():
    """The rcParams shared by every figure."""
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.3, 'hatch.linewidth': 0.4, 'hatch.color': 'grey',
        'legend.frameon': False, 'legend.handlelength': 2.8,
        'figure.dpi': 110, 'savefig.dpi': 300,
        'mathtext.default': 'regular', 'pdf.fonttype': 42})


def english_date(x, pos=None):
    """Day and month in English whatever the locale of the machine."""
    d = mdates.num2date(x)
    return f'{d.day} {MONTHS[d.month - 1]}'


def panel(ax, letter, x=0.02, y=0.98, colour='black'):
    """The panel letter inside the upper left corner on a white patch."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='left', va='top',
            fontsize=9, fontweight='bold', color=colour, zorder=20,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.2))
