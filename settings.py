import matplotlib as mpl
import os
import sys

def init():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    # Matplotlib settings
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 600       
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.formatter.limits']=(-3, 3)
    mpl.rcParams['axes.formatter.use_mathtext']=True

    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
