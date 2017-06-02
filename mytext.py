import numpy as np

__all__ = ['textTR',
            'textTL',
            'textBR',
            'textBL']

def textTR(ax, s, padding=0.02, **kwargs):
    """Quick placement of a plot annotation in top-right corner of current axis"""
    xlim = ax.get_xlim()
    x = xlim[1] - padding * np.abs(xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    y = ylim[1]-padding*np.abs(ylim[1] - ylim[0])
    ax.text(x, y, s, ha='right', va='top', **kwargs)
    ax.figure.show()

def textTL(ax, s, padding=0.02, **kwargs):
    """Quick placement of a plot annotation in top-left corner of current axis"""
    xlim = ax.get_xlim()
    x = xlim[0] + padding*np.abs(xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    y = ylim[1] - padding*np.abs(ylim[1] - ylim[0])
    ax.text(x, y, s, ha='left', va='top', **kwargs)
    ax.figure.show()

def textBR(ax, s, padding=0.02, **kwargs):
    """Quick placement of a plot annotation in bottom-right corner of current axis"""
    xlim = ax.get_xlim()
    x = xlim[1] - padding*np.abs(xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    y = ylim[0] + padding*np.abs(ylim[1] - ylim[0])
    ax.text(x,y,s,ha='right',va='bottom',**kwargs)
    ax.figure.show()

def textBL(ax, s, padding=0.02, **kwargs):
    """Quick placement of a plot annotation in bottom-left corner of current axis"""
    xlim = ax.get_xlim()
    x = xlim[0] + padding*np.abs(xlim[1] - xlim[0])
    ylim = ax.get_ylim()
    y = ylim[0] + padding*np.abs(ylim[1] - ylim[0])
    ax.text(x, y, s, ha='left', va='bottom', **kwargs)
    ax.figure.show()