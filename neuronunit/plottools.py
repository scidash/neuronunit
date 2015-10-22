#!/usr/bin/env python
"""DocString"""

from itertools import cycle
import matplotlib
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import colorsys
import numpy
import collections


def adjust_spines(ax, spines, color='k', d_out=10, d_down=[]):

    if d_down == []:
        d_down = d_out

    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    for loc, spine in ax.spines.items():
        if loc in spines:
            if loc == 'bottom':
                spine.set_position(('outward', d_down))  # outward by 10 points
            else:
                spine.set_position(('outward', d_out))  # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_visible(False) # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')

        if color is not 'k':

            ax.spines['left'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)


    elif 'right' not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')

        if color is not 'k':

            ax.spines['right'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)

    if 'bottom' in spines:
        pass
        ax.xaxis.set_ticks_position('bottom')
        #ax.axes.get_xaxis().set_visible(True)

    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)


def light_palette(color, n_colors=6, reverse=False, lumlight=0.8, light=None):

    rgb = mplcol.colorConverter.to_rgb(color)

    if light is not None:
        light = mplcol.colorConverter.to_rgb(light)
    else:
        vals = list(colorsys.rgb_to_hls(*rgb))
        vals[1] = lumlight
        light = colorsys.hls_to_rgb(*vals)

    colors = [color, light] if reverse else [light, color]
    pal = mplcol.LinearSegmentedColormap.from_list("blend", colors)

    palette = pal(numpy.linspace(0, 1, n_colors))

    return palette


def tiled_figure(figname='', frames=1, columns=2, figs=collections.OrderedDict(), axs=None, orientation='page', width_ratios=None, height_ratios=None, top=0.97, bottom=0.05, left=0.07, right=0.97, hspace=0.6, wspace=0.2, dirname=''):

    if figname not in figs:

        if axs == None:
            axs = []

        if orientation == 'landscape':
            figsize=(297/25.4, 210/25.4)
        elif orientation == 'page':
            figsize=(210/25.4, 297/25.4)

        # Some of these are not valid params in the current version
        # of matplotlib so I've commented them out.   
        params = {'backend': 'ps',
                  'axes.labelsize': 6,
                  'axes.linewidth' : 0.5,
                  #'title.fontsize': 8,
                  #'text.fontsize': 8,
                  'font.size': 8,
                  'axes.titlesize': 8,
                  'legend.fontsize': 8,
                  'xtick.labelsize': 6,
                  'ytick.labelsize': 6,
                  'legend.borderpad': 0.2,
                  #'legend.linewidth': 0.1, 
                  'legend.loc': 'best',
                  #'legend.ncol': 4, 
                  'text.usetex': False,
                  'pdf.fonttype': 42,
                  'figure.figsize': figsize}
        matplotlib.rcParams.update(params)

        fig = plt.figure(figname, facecolor='white')
        figs[figname] = {}
        figs[figname]['fig'] = fig
        figs[figname]['dirname'] = dirname

        if width_ratios is None:
            width_ratios=[1]*columns

        rows = int(numpy.ceil(frames/float(columns)))
        if height_ratios is None:
            height_ratios=[1]*rows

        gs = matplotlib.gridspec.GridSpec(rows, columns, height_ratios=height_ratios, width_ratios=width_ratios)
        gs.update(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

        for fi in range(frames):
            axs.append(fig.add_subplot(gs[int(fi/columns), int(fi%columns)]))
            adjust_spines(axs[-1], ['left', 'bottom'], d_out=0)

        figs[figname]['axs'] = axs

    else:
        axs = figs[figname]['axs']

    return axs
